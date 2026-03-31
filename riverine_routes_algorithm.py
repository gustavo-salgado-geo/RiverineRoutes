import os
import gc
import math
import tempfile
import processing
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterDistance,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber,
    QgsProcessingParameterCrs,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsFeatureSink,
    QgsVectorLayer,
    QgsFeature,
)

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import calculate_default_transform, reproject, Resampling
from skimage.morphology import skeletonize, thin as skimage_thin, binary_closing, binary_opening, remove_small_objects, disk
from scipy.ndimage import label as nd_label, convolve as nd_convolve
from shapely.geometry import LineString
from shapely.ops import nearest_points


# ---------------------------------------------------------------------------
# Funções utilitárias de CRS
# ---------------------------------------------------------------------------

def _is_geographic(crs_wkt: str) -> bool:
    """Devolve True se o CRS está em graus (geográfico), False se já é métrico.
    Usa osgeo.osr (GDAL/OSGeo4W) em vez de pyproj para evitar conflitos de
    instalação em ambientes com múltiplos PROJ (ex: OSGeo4W + pip).
    """
    try:
        from osgeo import osr
        srs = osr.SpatialReference()
        ret = srs.ImportFromWkt(crs_wkt)
        if ret == 0:
            return bool(srs.IsGeographic())
    except Exception:
        pass
    # Fallback heurístico — sem dependência externa
    wkt_upper = crs_wkt.upper()
    return any(h in wkt_upper for h in ["GEOGCS", "GEOGRAPHIC", "DEGREE", "DECIMAL_DEGREE"])


def _auto_utm_epsg(lon: float, lat: float) -> int:
    """
    Calcula o EPSG UTM mais adequado para uma longitude/latitude dadas.
    Cobre zonas WGS84 norte (326xx) e sul (327xx).
    """
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone   # UTM Norte
    else:
        return 32700 + zone   # UTM Sul


def _get_raster_center_lonlat(raster_path: str):
    """Devolve (lon, lat) do centro do raster em WGS84.
    Usa osgeo.osr (GDAL/OSGeo4W) para a transformação de coordenadas,
    evitando dependência do pyproj instalado via pip.
    """
    from osgeo import osr, ogr as _ogr
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        cx = (bounds.left + bounds.right) / 2
        cy = (bounds.bottom + bounds.top) / 2
        crs_wkt = src.crs.to_wkt() if src.crs else None

    if crs_wkt is None:
        return cx, cy   # sem SRC — devolver como estão

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(crs_wkt)

    if src_srs.IsGeographic():
        return cx, cy   # já em graus

    dst_srs = osr.SpatialReference()
    # WKT do WGS84 GEOGCS embutido — não consulta o proj.db
    _wgs84_wkt = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433,'
        'AUTHORITY["EPSG","9122"]],'
        'AUTHORITY["EPSG","4326"]]'
    )
    dst_srs.ImportFromWkt(_wgs84_wkt)
    dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    ct = osr.CoordinateTransformation(src_srs, dst_srs)
    point = _ogr.CreateGeometryFromWkt(f"POINT ({cx} {cy})")
    point.Transform(ct)
    return point.GetX(), point.GetY()


def _reproject_raster_to_metric(raster_path: str, target_epsg: int, tmp_folder: str) -> str:
    """
    Reprojeta um raster para o EPSG métrico indicado.
    Devolve o caminho para o ficheiro reprojetado.
    """
    dst_crs = f"EPSG:{target_epsg}"
    out_path = os.path.join(tmp_folder, f"raster_reproj_{target_epsg}.tif")

    with rasterio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })
        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )
    return out_path


def _parse_vector_source(qgis_source: str):
    """
    O QGIS representa fontes vetoriais com a sintaxe:
        /caminho/arquivo.gpkg|layername=nome_da_camada
        /caminho/arquivo.shp          (sem pipe — shapefile simples)

    O GeoPandas/pyogrio nao entende o '|layername=...' — precisa receber
    o caminho e o nome da camada separados.

    Devolve: (caminho_do_ficheiro, nome_da_camada_ou_None)
    """
    if "|layername=" in qgis_source:
        parts = qgis_source.split("|layername=", 1)
        return parts[0], parts[1]
    # Outros parametros possiveis (|subset=, |geometrytype=, etc.) sao ignorados
    clean_path = qgis_source.split("|")[0]
    return clean_path, None


def _read_vector(qgis_source: str) -> "gpd.GeoDataFrame":
    """Le uma camada vetorial a partir de um caminho no formato QGIS."""
    path, layer = _parse_vector_source(qgis_source)
    if layer:
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)


def _reproject_vector_to_metric(qgis_source: str, target_epsg: int, tmp_folder: str) -> str:
    """
    Reprojeta uma camada vetorial para o EPSG metrico indicado.
    Aceita caminhos no formato QGIS (com |layername=...).
    Devolve o caminho para o ficheiro reprojetado (.gpkg).
    """
    gdf = _read_vector(qgis_source)
    gdf_reproj = gdf.to_crs(epsg=target_epsg)
    out_path = os.path.join(tmp_folder, f"vector_reproj_{target_epsg}.gpkg")
    gdf_reproj.to_file(out_path, driver="GPKG")
    return out_path


# ---------------------------------------------------------------------------
# Algoritmo principal
# ---------------------------------------------------------------------------

class RiverineRoutesAlgorithm(QgsProcessingAlgorithm):
    """
    Algoritmo para criação da rede completa de rotas fluviais:
    Centrais, Marginais e Transversais.

    Garante que todos os parâmetros de distância (buffer e transecto)
    são sempre interpretados em METROS, independentemente do SRC dos dados
    de entrada. Se necessário, os dados são reprojetados automaticamente
    para um sistema UTM métrico antes do processamento.
    """

    INPUT_RASTER     = "INPUT_RASTER"
    INPUT_VECTOR     = "INPUT_VECTOR"
    INPUT_LIMITS     = "INPUT_LIMITS"
    INPUT_GPS_TRACKS = "INPUT_GPS_TRACKS"
    BUFFER_DIST      = "BUFFER_DIST"
    TRANSECT_INTERVAL = "TRANSECT_INTERVAL"
    OUTPUT_CRS       = "OUTPUT_CRS"
    OUTPUT_NETWORK   = "OUTPUT_NETWORK"

    # ------------------------------------------------------------------
    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return RiverineRoutesAlgorithm()

    def name(self):
        return "riverineroutes"

    def displayName(self):
        return self.tr("2. RiverineRoutes (Gerar Rede Fluvial)")

    def group(self):
        return self.tr("Módulo Base")

    def groupId(self):
        return "modulo_base"

    def shortHelpString(self):
        return self.tr(
            "Cria uma rede topologicamente correta com rotas centrais (esqueleto), "
            "marginais (buffers) e transversais.\n\n"
            "DISTÂNCIAS EM METROS: Os parâmetros de buffer e transecto são sempre "
            "interpretados em metros. Se os dados de entrada estiverem em sistema "
            "geográfico (graus), o algoritmo reprojeta automaticamente para UTM "
            "antes de processar.\n\n"
            "SRC DE SAÍDA: Defina o EPSG desejado para a rede final. Se deixado "
            "em branco, a saída usa o mesmo SRC métrico de processamento."
        )

    # ------------------------------------------------------------------
    def initAlgorithm(self, config=None):

        # 1. Raster de máscara binária
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr("Máscara Binária de Água (Raster)"),
            )
        )

        # 2. Vetor de máscara binária
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VECTOR,
                self.tr("Máscara de Água (Vetor)"),
                [QgsProcessing.TypeVectorPolygon],
            )
        )

        # 3. Limites da área de estudo (opcional)
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_LIMITS,
                self.tr("Limites da Área de Estudo (Polígonos)"),
                [QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )

        # 4. Rastreio GPS (opcional)
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_GPS_TRACKS,
                self.tr("Ficheiro de Rastreio GPS (Opcional)"),
                [QgsProcessing.TypeVectorPoint],
                optional=True,
            )
        )

        # 5. Distância do Buffer Marginal — SEMPRE EM METROS
        param_buffer = QgsProcessingParameterNumber(
            self.BUFFER_DIST,
            self.tr("Distância do Buffer Marginal (metros)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=200.0,
            minValue=1.0,
        )
        param_buffer.setMetadata(
            {"widget_wrapper": {"decimals": 1}}
        )
        self.addParameter(param_buffer)

        # 6. Distância entre Rotas Transversais — SEMPRE EM METROS
        param_transect = QgsProcessingParameterNumber(
            self.TRANSECT_INTERVAL,
            self.tr("Distância entre Rotas Transversais (metros)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=1000.0,
            minValue=1.0,
        )
        param_transect.setMetadata(
            {"widget_wrapper": {"decimals": 1}}
        )
        self.addParameter(param_transect)

        # 7. SRC do dado de saída (opcional — deixar em branco para usar o SRC de processamento)
        self.addParameter(
            QgsProcessingParameterCrs(
                self.OUTPUT_CRS,
                self.tr("SRC do Dado de Saída (EPSG — opcional)"),
                optional=True,
                defaultValue=None,
            )
        )

        # OUTPUT: Rede final de linhas
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_NETWORK,
                self.tr("Rede Fluvial Completa (Linhas)"),
            )
        )

    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):

        # ── 1. Recuperar parâmetros ─────────────────────────────────────
        raster_layer  = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        vector_layer  = self.parameterAsVectorLayer(parameters, self.INPUT_VECTOR, context)
        gps_layer     = self.parameterAsVectorLayer(parameters, self.INPUT_GPS_TRACKS, context)
        limits_layer  = self.parameterAsVectorLayer(parameters, self.INPUT_LIMITS, context)

        buffer_dist_m    = self.parameterAsDouble(parameters, self.BUFFER_DIST, context)
        transect_interval_m = self.parameterAsDouble(parameters, self.TRANSECT_INTERVAL, context)

        output_crs_param = self.parameterAsCrs(parameters, self.OUTPUT_CRS, context)

        raster_path_orig = raster_layer.source()
        vector_path_orig = vector_layer.source()
        tmp = context.temporaryFolder()

        # ── 2. Detectar CRS e reprojetar se necessário ──────────────────
        #
        # FONTE DE VERDADE: os objetos QgsRasterLayer / QgsVectorLayer do QGIS.
        # O QGIS pode ter o SRC definido na camada em memória mesmo quando o
        # ficheiro físico não gravou os metadados de projeção. Usar rasterio/
        # geopandas diretamente no ficheiro falha nesses casos.
        # Rasterio e geopandas são usados apenas como fallback de último recurso.
        # -----------------------------------------------------------------
        feedback.pushInfo(self.tr("A verificar SRC dos dados de entrada..."))

        # ---- 2a. CRS do raster — fonte primária: QgsRasterLayer -----------
        qgs_raster_crs = raster_layer.crs()          # QgsCoordinateReferenceSystem
        raster_crs_wkt = None
        raster_is_geo  = None

        if qgs_raster_crs.isValid():
            raster_crs_wkt = qgs_raster_crs.toWkt()
            raster_is_geo  = _is_geographic(raster_crs_wkt)
            feedback.pushInfo(
                self.tr(
                    f"SRC do raster lido do QGIS: "
                    f"{qgs_raster_crs.authid()} — "
                    f"{'Geográfico (graus)' if raster_is_geo else 'Projetado (métrico)'}"
                )
            )
        else:
            # Fallback: tentar rasterio diretamente no ficheiro
            try:
                with rasterio.open(raster_path_orig) as src:
                    if src.crs is not None:
                        raster_crs_wkt = src.crs.to_wkt()
                        raster_is_geo  = _is_geographic(raster_crs_wkt)
                        feedback.pushInfo(
                            self.tr(
                                f"SRC do raster lido via rasterio: "
                                f"{'Geográfico' if raster_is_geo else 'Projetado'}"
                            )
                        )
            except Exception:
                pass
            if raster_crs_wkt is None:
                feedback.pushWarning(
                    self.tr(
                        "AVISO: O raster não tem SRC definido nem no QGIS nem no ficheiro. "
                        "O algoritmo tentará usar o SRC do vetor. "
                        "Para evitar este aviso, atribua o SRC ao raster no QGIS "
                        "(Raster → Projeções → Atribuir SRC)."
                    )
                )

        # ---- 2b. CRS do vetor — fonte primária: QgsVectorLayer ------------
        qgs_vector_crs = vector_layer.crs()
        vector_epsg    = None
        vector_is_geo  = None

        if qgs_vector_crs.isValid():
            vector_crs_wkt = qgs_vector_crs.toWkt()
            vector_is_geo  = _is_geographic(vector_crs_wkt)
            try:
                auth_id = qgs_vector_crs.authid()   # ex: "EPSG:31982"
                if auth_id.upper().startswith("EPSG:"):
                    vector_epsg = int(auth_id.split(":")[1])
            except Exception:
                vector_epsg = None
            feedback.pushInfo(
                self.tr(
                    f"SRC do vetor lido do QGIS: "
                    f"{qgs_vector_crs.authid()} — "
                    f"{'Geográfico (graus)' if vector_is_geo else 'Projetado (métrico)'}"
                )
            )
        else:
            # Fallback: tentar geopandas
            try:
                gdf_crs_check = _read_vector(vector_path_orig)
                if gdf_crs_check.crs is not None:
                    vector_epsg   = gdf_crs_check.crs.to_epsg()
                    vector_is_geo = gdf_crs_check.crs.is_geographic
                    feedback.pushInfo(
                        self.tr(
                            f"SRC do vetor lido via geopandas: EPSG:{vector_epsg}"
                        )
                    )
            except Exception:
                pass
            if vector_is_geo is None:
                feedback.pushWarning(
                    self.tr(
                        "AVISO: O vetor não tem SRC definido nem no QGIS nem no ficheiro."
                    )
                )

        # ---- 2c. Ambos sem SRC — erro fatal --------------------------------
        if raster_crs_wkt is None and vector_is_geo is None:
            feedback.reportError(
                self.tr(
                    "ERRO: Nao foi possivel determinar o SRC do raster nem do vetor. "
                    "Atribua o SRC correto a ambas as camadas no QGIS e tente novamente. "
                    "Raster: clique direito → Propriedades → SRC ou Raster → Projecoes → Atribuir SRC. "
                    "Vetor: clique direito na camada → Definir SRC da Camada."
                ),
                fatalError=True,
            )
            return {}

        # ---- 2d. Determinar EPSG de trabalho (métrico) --------------------
        #
        # Prioridade: (1) raster métrico → usa direto
        #             (2) raster geográfico → calcula UTM pelo centro do raster
        #             (3) raster sem SRC + vetor métrico → usa EPSG do vetor
        #             (4) raster sem SRC + vetor geográfico → calcula UTM pelo centroide do vetor

        work_epsg = None

        if raster_crs_wkt is not None and not raster_is_geo:
            # Caso 1: raster já métrico.
            # Fonte primária: authid do QgsCoordinateReferenceSystem (ex: "EPSG:32722").
            # O pyproj é usado apenas se o authid não tiver prefixo EPSG.
            work_epsg = None

            # Tentativa 1: authid do QGIS (mais fiável — é o que o QGIS mostra ao utilizador)
            try:
                auth_id = qgs_raster_crs.authid()   # ex: "EPSG:32722"
                if auth_id.upper().startswith("EPSG:"):
                    work_epsg = int(auth_id.split(":")[1])
            except Exception:
                pass

            # Tentativa 2: osr (fallback para SRCs não-EPSG)
            if work_epsg is None:
                try:
                    from osgeo import osr as _osr2
                    _srs2 = _osr2.SpatialReference()
                    _srs2.ImportFromWkt(raster_crs_wkt)
                    epsg_str = _srs2.GetAuthorityCode(None)
                    if epsg_str:
                        work_epsg = int(epsg_str)
                except Exception:
                    pass

            # Tentativa 3: postgisSrid do QGIS (último recurso)
            if work_epsg is None:
                try:
                    srid = qgs_raster_crs.postgisSrid()
                    if srid and srid > 0:
                        work_epsg = srid
                except Exception:
                    pass

            if work_epsg:
                feedback.pushInfo(
                    self.tr(f"SRC de trabalho: EPSG:{work_epsg} (raster métrico, sem reprojeção).")
                )
            else:
                # Nao conseguiu extrair EPSG numérico — usar o objecto QGIS directamente
                # (work_crs_qgs será definido abaixo a partir de qgs_raster_crs)
                feedback.pushWarning(
                    self.tr(
                        f"Nao foi possivel extrair o codigo EPSG numerico do SRC do raster "
                        f"({qgs_raster_crs.authid()}). O SRC QGIS sera usado directamente. "
                        f"Verifique os resultados."
                    )
                )
            raster_path = raster_path_orig

        elif raster_crs_wkt is not None and raster_is_geo:
            # Caso 2: raster geográfico → reprojetar para UTM
            lon, lat  = _get_raster_center_lonlat(raster_path_orig)
            work_epsg = _auto_utm_epsg(lon, lat)
            feedback.pushInfo(
                self.tr(
                    f"SRC do raster é geográfico. "
                    f"Reprojetando para EPSG:{work_epsg} (UTM)..."
                )
            )
            raster_path = _reproject_raster_to_metric(raster_path_orig, work_epsg, tmp)

        elif vector_is_geo is not None and not vector_is_geo and vector_epsg:
            # Caso 3: raster sem SRC, vetor já métrico
            work_epsg = vector_epsg
            feedback.pushWarning(
                self.tr(
                    f"Raster sem SRC. Usando SRC do vetor (EPSG:{work_epsg}) como SRC de trabalho. "
                    f"Atribua o SRC real ao raster para evitar este aviso."
                )
            )
            raster_path = raster_path_orig

        else:
            # Caso 4: raster sem SRC, vetor geográfico → UTM pelo centroide do vetor
            try:
                gdf_tmp = _read_vector(vector_path_orig)
                cx = gdf_tmp.geometry.unary_union.centroid.x
                cy = gdf_tmp.geometry.unary_union.centroid.y
                work_epsg = _auto_utm_epsg(cx, cy)
            except Exception:
                feedback.reportError(
                    self.tr(
                        "ERRO: Não foi possível calcular o UTM automático. "
                        "Atribua o SRC correto ao raster e tente novamente."
                    ),
                    fatalError=True,
                )
                return {}
            feedback.pushWarning(
                self.tr(
                    f"Raster sem SRC e vetor geográfico. "
                    f"Usando EPSG:{work_epsg} (UTM auto pelo centroide do vetor) "
                    f"como SRC de trabalho."
                )
            )
            raster_path = raster_path_orig

        # ---- 2e. Reprojetar vetor se necessário ---------------------------
        needs_vector_reproject = False
        if vector_is_geo is True:
            needs_vector_reproject = True
        elif work_epsg and vector_epsg and vector_epsg != work_epsg:
            needs_vector_reproject = True

        if needs_vector_reproject:
            feedback.pushInfo(
                self.tr(
                    f"SRC do vetor (EPSG:{vector_epsg}) difere do SRC de trabalho "
                    f"(EPSG:{work_epsg}). A reprojetar vetor..."
                )
            )
            vector_path = _reproject_vector_to_metric(vector_path_orig, work_epsg, tmp)
        else:
            feedback.pushInfo(
                self.tr(
                    f"SRC do vetor compativel com SRC de trabalho (EPSG:{work_epsg}). "
                    f"Sem necessidade de reprojecao."
                )
            )
            vector_path = vector_path_orig

        # Leitura do GeoDataFrame do vetor (para uso nas etapas seguintes)
        gdf_check = _read_vector(vector_path)

        feedback.pushInfo(
            self.tr(
                f"Parâmetros de distância — Buffer: {buffer_dist_m} m | "
                f"Transecto: {transect_interval_m} m"
            )
        )

        # SRC QGIS de trabalho (usado no merge e na saída)
        # Prioridade: (1) EPSG numérico extraído → constrói por código
        #             (2) QgsRasterLayer.crs() → já é um objecto válido
        #             (3) CRS do projecto QGIS → último recurso
        if work_epsg:
            work_crs_qgs = QgsCoordinateReferenceSystem(f"EPSG:{work_epsg}")
        elif qgs_raster_crs.isValid():
            work_crs_qgs = qgs_raster_crs
            # Tentar recuperar o EPSG a partir do objecto QGIS como inteiro
            try:
                srid = qgs_raster_crs.postgisSrid()
                if srid and srid > 0:
                    work_epsg = srid
                    work_crs_qgs = QgsCoordinateReferenceSystem(f"EPSG:{work_epsg}")
            except Exception:
                pass
        else:
            work_crs_qgs = context.project().crs()

        feedback.pushInfo(
            self.tr(f"SRC QGIS de trabalho definido: {work_crs_qgs.authid()}")
        )

        # ── A. ROTAS CENTRAIS ───────────────────────────────────────────
        #
        # ESTRATÉGIA DE MEMÓRIA:
        #
        # O skeletonize() precisa da imagem INTEIRA para gerar um esqueleto
        # contínuo. Processar em blocos independentes cria descontinuidades
        # nas bordas — por isso não é possível usar tiling directo.
        #
        # Estratégia adoptada (3 fases):
        #
        # FASE 1 — Quantização: ler o raster em blocos e gravar um raster
        #   temporário uint8 comprimido. Reduz o tamanho antes de carregar
        #   tudo na RAM (rasters float32/int16 passam para 1 byte/pixel).
        #
        # FASE 2 — Esqueletonização: carregar o raster uint8 comprimido
        #   inteiro em RAM como array bool (menor pegada possível),
        #   esqueletonizar, converter para uint8 e libertar o bool.
        #
        # FASE 3 — Gravação: gravar o esqueleto uint8 para disco em blocos
        #   e libertar o array da memória antes de avançar.
        #
        # Pegada máxima de RAM durante a fase 2 (pior caso):
        #   raster 500 MB × 1 byte/pixel (uint8 quantizado)
        #   + array bool = mesma dimensão (~500 MB)
        #   + array uint8 do esqueleto (~500 MB)
        #   Total estimado: ~1.5 GB — aceitável para 8+ GB de RAM.
        #
        # Para rasters > 1 GB recomenda-se usar um SRC com tiles menores
        # ou dividir a área de estudo em sub-regiões antes de processar.
        # ---------------------------------------------------------------
        feedback.pushInfo(self.tr("A gerar rotas centrais — fase 1/3: a quantizar raster..."))

        temp_uint8_path = os.path.join(tmp, "temp_uint8.tif")
        temp_skel_path  = os.path.join(tmp, "temp_skeleton.tif")

        # FASE 1: ler em blocos, gravar uint8 comprimido
        with rasterio.open(raster_path) as src:
            transform = src.transform
            crs       = src.crs
            height    = src.height
            width     = src.width

            # Blocos de leitura: linhas suficientes para ~64 MB por bloco
            # (seguro mesmo com pouca RAM disponível nesta fase)
            bytes_per_row = width  # uint8 = 1 byte/pixel após conversão
            rows_per_block = max(1, min(height, (64 * 1024 * 1024) // max(bytes_per_row, 1)))

            profile_uint8 = {
                "driver":    "GTiff",
                "dtype":     "uint8",
                "count":     1,
                "height":    height,
                "width":     width,
                "crs":       crs,
                "transform": transform,
                "compress":  "lzw",
                "tiled":     True,
                "blockxsize": 512,
                "blockysize": 512,
            }

            n_blocks = math.ceil(height / rows_per_block)
            with rasterio.open(temp_uint8_path, "w", **profile_uint8) as dst:
                for i, row_off in enumerate(range(0, height, rows_per_block)):
                    actual_rows = min(rows_per_block, height - row_off)
                    window = rasterio.windows.Window(0, row_off, width, actual_rows)
                    block  = src.read(1, window=window)
                    dst.write((block > 0).astype(np.uint8), 1, window=window)
                    del block
                    gc.collect()
                    feedback.setProgress(int((i + 1) / n_blocks * 20))  # 0-20%
                    if feedback.isCanceled():
                        return {}

        feedback.pushInfo(self.tr("Fase 2/3: a pre-processar e esqueletonizar (toda a imagem em RAM)..."))

        # FASE 2: carregar → pré-processar → skeletonize → uint8
        #
        # PRÉ-PROCESSAMENTO (ordem importa):
        #
        # 1. REMOÇÃO DE ILHAS PEQUENAS (remove_small_objects):
        #    Elimina manchas de pixels isolados (ruído, pixels soltos)
        #    menores que MIN_ISLAND_PX pixels. Sem isto, cada pixel isolado
        #    gera um esqueleto espúrio.
        #    MIN_ISLAND_PX = área em pixels equivalente a um círculo de
        #    raio = buffer_dist_m / pixel_size. Assim, apenas objectos com
        #    área relevante para navegação são mantidos.
        #
        # 2. FECHO MORFOLÓGICO (binary_closing):
        #    Preenche buracos pequenos e gaps na máscara (ilhas minúsculas,
        #    pixels pretos espúrios dentro do canal). Um rio contínuo deve
        #    ter uma máscara contínua — gaps causam descontinuidades no
        #    esqueleto. Kernel circular de raio CLOSE_R pixels.
        #
        # 3. SUAVIZAÇÃO DAS BORDAS (binary_opening):
        #    Remove protuberâncias finas nas bordas da máscara (efeito
        #    "dentes de serra" de pixelização). Reduz drasticamente as
        #    ramificações falsas no esqueleto. Kernel circular de raio
        #    OPEN_R pixels (menor que CLOSE_R para não destruir canais finos).
        #
        # Os raios são calculados em pixels a partir de buffer_dist_m,
        # garantindo que as operações morfológicas são proporcionais à
        # escala real do dado independentemente da resolução do raster.
        # ---------------------------------------------------------------
        with rasterio.open(temp_uint8_path) as src:
            data_uint8  = src.read(1)
            pixel_size_m = abs(src.transform.a)   # metros por pixel
        feedback.setProgress(22)

        data_bool = data_uint8 > 0
        del data_uint8
        gc.collect()

        # Calcular raios morfológicos em pixels
        # CLOSE_R: preenche gaps até ~10% do buffer marginal
        # OPEN_R:  suaviza bordas a ~5% do buffer marginal (metade do close)
        # MIN_ISLAND_PX: área mínima = círculo de raio = buffer/2
        close_r         = max(1, int(round((buffer_dist_m * 0.10) / max(pixel_size_m, 1e-6))))
        open_r          = max(1, int(round((buffer_dist_m * 0.05) / max(pixel_size_m, 1e-6))))
        min_island_px   = max(10, int(3.14159 * ((buffer_dist_m * 0.5) / max(pixel_size_m, 1e-6)) ** 2))

        feedback.pushInfo(self.tr(
            f"Pre-processamento morfologico: pixel={pixel_size_m:.2f}m | "
            f"close_r={close_r}px | open_r={open_r}px | min_island={min_island_px}px"
        ))

        # 1. Remover ilhas pequenas
        feedback.pushInfo(self.tr("  Removendo objectos pequenos (ruido)..."))
        data_bool = remove_small_objects(data_bool, min_size=min_island_px, connectivity=2)
        gc.collect()
        feedback.setProgress(25)

        # 2. Fecho morfológico — preencher gaps e buracos
        feedback.pushInfo(self.tr("  Fecho morfologico (preenchimento de gaps)..."))
        data_bool = binary_closing(data_bool, footprint=disk(close_r))
        gc.collect()
        feedback.setProgress(30)

        # 3. Abertura morfológica — suavizar bordas
        feedback.pushInfo(self.tr("  Abertura morfologica (suavizacao de bordas)..."))
        data_bool = binary_opening(data_bool, footprint=disk(open_r))
        gc.collect()
        feedback.setProgress(33)

        # Esqueletonizar a máscara pré-processada
        feedback.pushInfo(self.tr("  Esqueletonizando..."))
        skeleton_bool = skeletonize(data_bool)
        del data_bool
        gc.collect()
        feedback.setProgress(35)

        # ── AFINAMENTO ADICIONAL (thin) ─────────────────────────────────
        # skeletonize() pode deixar trechos com 2 pixels de largura,
        # especialmente em curvas e confluências. Isso gera duas linhas
        # paralelas no rastreio de grafo.
        # thin() aplica o algoritmo de Zhang-Suen iterativamente até
        # convergência, garantindo largura de exactamente 1 pixel.
        feedback.pushInfo(self.tr("  Afinamento (thin) para garantir 1 pixel de largura..."))
        skeleton_bool = skimage_thin(skeleton_bool)
        gc.collect()
        feedback.setProgress(37)

        # ── PODA DE RAMOS CURTOS (PRUNING) ─────────────────────────────
        #
        # O skeletonize gera ramos espúrios em:
        #   • Confluências (triângulo em vez de Y)
        #   • Bordas irregulares da máscara (pequenos "pelos")
        #   • Ilhas e concavidades
        #
        # Algoritmo de poda iterativa:
        #   Um pixel de esqueleto é "terminal" se tiver exactamente 1
        #   vizinho activo na vizinhança 3×3. Remover terminais iterativamente
        #   elimina ramos curtos sem afectar o canal principal.
        #
        # PRUNE_PX = comprimento máximo a podar em pixels
        #   = buffer_dist_m / pixel_size_m × 0.5
        #   (ramos menores que metade do buffer são considerados espúrios)
        # ---------------------------------------------------------------
        prune_px = max(3, int(round((buffer_dist_m * 0.5) / max(pixel_size_m, 1e-6))))
        feedback.pushInfo(self.tr(f"  Poda de ramos curtos: prune_px={prune_px}px ({prune_px * pixel_size_m:.1f}m)..."))

        skel = skeleton_bool.copy()
        del skeleton_bool
        gc.collect()

        # Kernel de vizinhança 3×3 para contar vizinhos
        neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_kernel[1, 1] = 0   # não contar o próprio pixel

        for _iter in range(prune_px):
            if not skel.any():
                break
            # Contar vizinhos activos de cada pixel
            neighbor_count = nd_convolve(skel.astype(np.uint8), neighbor_kernel, mode="constant", cval=0)
            # Terminal: pixel activo com exactamente 1 vizinho
            terminals = skel & (neighbor_count == 1)
            if not terminals.any():
                break
            skel = skel & ~terminals

        feedback.setProgress(40)

        skeleton_uint8 = skel.astype(np.uint8)
        del skel
        gc.collect()

        feedback.pushInfo(self.tr("Fase 3/3: a gravar esqueleto em disco..."))

        # FASE 3: gravar em blocos e libertar
        profile_skel = profile_uint8.copy()
        with rasterio.open(temp_skel_path, "w", **profile_skel) as dst:
            for i, row_off in enumerate(range(0, height, rows_per_block)):
                actual_rows = min(rows_per_block, height - row_off)
                window = rasterio.windows.Window(0, row_off, width, actual_rows)
                dst.write(skeleton_uint8[row_off:row_off + actual_rows, :], 1, window=window)
                feedback.setProgress(40 + int((i + 1) / n_blocks * 10))  # 40-50%
                if feedback.isCanceled():
                    return {}

        del skeleton_uint8
        gc.collect()

        # Remover o raster uint8 intermédio para libertar espaço em disco
        try:
            os.remove(temp_uint8_path)
        except OSError:
            pass

        feedback.pushInfo(self.tr("Esqueleto gravado. A vectorizar via rastreio de grafo de pixels..."))

        # ── VECTORIZAÇÃO DO ESQUELETO — RASTREIO DE GRAFO ──────────────
        #
        # PROBLEMA com rasterio.features.shapes() + boundary:
        #   shapes() agrupa pixels contíguos em POLÍGONOS (regiões de cor
        #   uniforme). O boundary desses polígonos é o CONTORNO da região,
        #   não a linha central. Para um esqueleto diagonal, o contorno é
        #   uma série de quadradinhos perpendiculares ao canal — exactamente
        #   os "traços" visíveis na imagem.
        #
        # SOLUÇÃO — rastreio de grafo de pixels (pixel graph tracing):
        #   1. Carregar o esqueleto inteiro como array bool.
        #   2. Para cada pixel activo, identificar os seus vizinhos activos
        #      na vizinhança 8-conexa (Moore neighbourhood).
        #   3. Os pixels com exactamente 2 vizinhos são "internos" (meio
        #      de uma linha). Os pixels com 1 vizinho são "terminais"
        #      (ponta de linha). Os pixels com 3+ vizinhos são "junções".
        #   4. Percorrer a cadeia de pixels internos entre junções/terminais,
        #      recolhendo as coordenadas reais (via transform) em ordem.
        #   5. Cada cadeia torna-se uma LineString. Depois simplify +
        #      Chaikin para suavizar os degraus da grade.
        #
        # MEMÓRIA: o esqueleto uint8 já está em memória (skeleton_uint8
        # foi gravado no ficheiro mas o array foi libertado). Relemos o
        # ficheiro comprimido — ocupa muito menos que o raster original.
        # ---------------------------------------------------------------
        from shapely.geometry import LineString as ShpLineString
        from osgeo            import ogr, osr
        from scipy.ndimage    import label as nd_label2

        feedback.pushInfo(self.tr("A carregar esqueleto para rastreio de grafo..."))

        with rasterio.open(temp_skel_path) as src:
            skel_arr       = src.read(1).astype(bool)
            skel_transform = src.transform
            pixel_size     = abs(skel_transform.a)

        feedback.setProgress(51)

        # ── Calcular número de vizinhos 8-conexos para cada pixel ──────
        # neighbor_kernel já definido acima (nd_convolve + ones 3x3 sem centro)
        n_neighbors = nd_convolve(
            skel_arr.astype(np.uint8), neighbor_kernel, mode="constant", cval=0
        )

        # Classificar pixels
        terminals  = skel_arr & (n_neighbors == 1)   # ponta
        internals  = skel_arr & (n_neighbors == 2)   # meio de linha
        junctions  = skel_arr & (n_neighbors >= 3)   # bifurcação/cruzamento
        # Pixels isolados (0 vizinhos) — ignorar
        skel_arr = skel_arr & (n_neighbors >= 1)

        feedback.setProgress(53)

        # ── Função para converter (row, col) → (x, y) em coordenadas reais
        def _rc_to_xy(r, c):
            x, y = rasterio.transform.xy(skel_transform, r, c, offset="center")
            return float(x), float(y)

        # ── Função de suavização de Chaikin ──────────────────────────
        def _chaikin(coords, iters=3):
            for _ in range(iters):
                if len(coords) < 3:
                    break
                nc = [coords[0]]
                for j in range(len(coords) - 1):
                    x0, y0 = coords[j];  x1, y1 = coords[j+1]
                    nc.append((0.75*x0 + 0.25*x1, 0.75*y0 + 0.25*y1))
                    nc.append((0.25*x0 + 0.75*x1, 0.25*y0 + 0.75*y1))
                nc.append(coords[-1])
                coords = nc
            return coords

        # ── Rastreio: percorrer cada cadeia de pixels ───────────────────
        # Estratégia:
        #   • Marcar pixels visitados num array separado.
        #   • Para cada pixel de arranque (terminal ou junção), percorrer
        #     os vizinhos não visitados até atingir outro terminal/junção
        #     ou um pixel sem saída.
        #   • Cada cadeia completa → uma LineString.

        visited  = np.zeros_like(skel_arr, dtype=bool)
        lines_rc = []   # lista de listas de (row, col)

        # Offsets dos 8 vizinhos (Moore neighbourhood)
        NEIGHBORS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        def _get_active_neighbors(r, c):
            nb = []
            for dr, dc in NEIGHBORS_8:
                nr, nc_ = r+dr, c+dc
                if 0 <= nr < skel_arr.shape[0] and 0 <= nc_ < skel_arr.shape[1]:
                    if skel_arr[nr, nc_] and not visited[nr, nc_]:
                        nb.append((nr, nc_))
            return nb

        # Pontos de arranque: terminais primeiro, depois junções
        start_mask = terminals | junctions
        start_rows, start_cols = np.where(start_mask)

        # Se não há terminais/junções (linha circular fechada), usar qualquer pixel
        if len(start_rows) == 0:
            start_rows, start_cols = np.where(skel_arr)

        total_starts = len(start_rows)
        for si, (sr, sc) in enumerate(zip(start_rows, start_cols)):
            if visited[sr, sc]:
                continue

            # Percorrer cada ramo saindo deste ponto de arranque
            neighbors = _get_active_neighbors(sr, sc)
            if not neighbors and not visited[sr, sc]:
                # Pixel isolado (já podado mas ficou) — ignorar
                visited[sr, sc] = True
                continue

            visited[sr, sc] = True

            for (nr, nc_) in neighbors:
                if visited[nr, nc_]:
                    continue
                # Seguir a cadeia
                chain = [(sr, sc), (nr, nc_)]
                visited[nr, nc_] = True
                cur_r, cur_c = nr, nc_

                while True:
                    nbs = _get_active_neighbors(cur_r, cur_c)
                    if not nbs:
                        break
                    # Se há múltiplos vizinhos não visitados: parar (chegou a junção)
                    if len(nbs) > 1:
                        # Marcar como ponto de chegada mas não marcar como visitado
                        # (outras cadeias arrancarão daqui)
                        chain.append(nbs[0])
                        break
                    next_r, next_c = nbs[0]
                    chain.append((next_r, next_c))
                    visited[next_r, next_c] = True
                    cur_r, cur_c = next_r, next_c
                    # Parar se chegou a terminal ou junção
                    if terminals[cur_r, cur_c] or junctions[cur_r, cur_c]:
                        break

                if len(chain) >= 2:
                    lines_rc.append(chain)

            if si % 5000 == 0:
                feedback.setProgress(53 + int(si / max(total_starts, 1) * 5))
                if feedback.isCanceled():
                    return {}

        del skel_arr, visited, terminals, internals, junctions, n_neighbors
        gc.collect()
        feedback.pushInfo(self.tr(f"Rastreio concluido: {len(lines_rc)} cadeias encontradas."))
        feedback.setProgress(58)

        # ── Converter cadeias (row,col) → LineStrings Shapely ──────────
        # Fase 1: converter todas as cadeias para LineStrings reais,
        #         aplicando simplify + Chaikin para suavização.
        # Fase 2: linemerge para unir segmentos que se tocam nas pontas.
        # Fase 3: deduplicação por distância média (Hausdorff aproximado)
        #         para eliminar linhas paralelas remanescentes.
        # Fase 4: gravar via OGR.

        from shapely.ops import linemerge, unary_union as shp_unary_union
        from shapely.geometry import MultiLineString as ShpMultiLine

        simplify_tol = pixel_size * 1.5
        shapely_lines = []

        feedback.pushInfo(self.tr("  Convertendo cadeias para LineStrings e suavizando..."))
        for chain in lines_rc:
            coords = [_rc_to_xy(r, c) for r, c in chain]
            if len(coords) < 2:
                continue
            line = ShpLineString(coords).simplify(simplify_tol, preserve_topology=False)
            if line.is_empty or line.length == 0:
                continue
            smooth_coords = _chaikin(list(line.coords), iters=3)
            if len(smooth_coords) < 2:
                continue
            sl = ShpLineString(smooth_coords)
            if not sl.is_empty and sl.length > 0:
                shapely_lines.append(sl)

        del lines_rc
        gc.collect()
        feedback.pushInfo(self.tr(f"  {len(shapely_lines)} segmentos antes do merge."))

        # ── LINEMERGE: unir segmentos que se tocam nas extremidades ─────
        # Elimina a fragmentação — segmentos contíguos tornam-se uma linha
        # única contínua, o que melhora drasticamente a qualidade das
        # transversais (interpolate opera sobre o comprimento total).
        feedback.pushInfo(self.tr("  Unindo segmentos contiguos (linemerge)..."))
        merged = linemerge(shapely_lines)
        del shapely_lines
        gc.collect()

        if merged.geom_type == "LineString":
            merged_lines = [merged]
        elif merged.geom_type == "MultiLineString":
            merged_lines = list(merged.geoms)
        else:
            # GeometryCollection — extrair apenas LineStrings
            merged_lines = [g for g in merged.geoms if g.geom_type == "LineString"]

        feedback.pushInfo(self.tr(f"  {len(merged_lines)} linhas apos linemerge."))

        # ── DEDUPLICAÇÃO POR DISTÂNCIA MÉDIA (Hausdorff aproximado) ────
        # Compara amostras de pontos ao longo de cada par de linhas.
        # Duas linhas são paralelas se a distância média entre amostras
        # for menor que DEDUP_DIST (3 pixels). Mantém a mais longa.
        # Muito mais robusto que comparar apenas o ponto médio.
        DEDUP_DIST  = pixel_size * 3.0
        N_SAMPLES   = 10   # pontos amostrados por linha para comparação

        def _avg_dist(la, lb):
            """Distância média entre N_SAMPLES pontos de la e lb."""
            total = 0.0
            for k in range(N_SAMPLES):
                t = k / max(N_SAMPLES - 1, 1)
                pa = la.interpolate(t, normalized=True)
                dist = lb.distance(pa)
                total += dist
            return total / N_SAMPLES

        if len(merged_lines) > 1:
            feedback.pushInfo(self.tr(f"  Deduplicando {len(merged_lines)} linhas por distancia media..."))
            keep = [True] * len(merged_lines)
            lengths = [l.length for l in merged_lines]
            for i in range(len(merged_lines)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(merged_lines)):
                    if not keep[j]:
                        continue
                    # Verificação rápida por bounding box antes do cálculo completo
                    bi = merged_lines[i].bounds
                    bj = merged_lines[j].bounds
                    bb_dist = max(
                        max(bi[0], bj[0]) - min(bi[2], bj[2]),
                        max(bi[1], bj[1]) - min(bi[3], bj[3]),
                        0
                    )
                    if bb_dist > DEDUP_DIST * 2:
                        continue   # bounding boxes distantes — não são paralelas
                    if _avg_dist(merged_lines[i], merged_lines[j]) < DEDUP_DIST:
                        if lengths[i] >= lengths[j]:
                            keep[j] = False
                        else:
                            keep[i] = False
                            break
            merged_lines = [l for l, k in zip(merged_lines, keep) if k]
            feedback.pushInfo(self.tr(f"  {len(merged_lines)} linhas apos deduplicacao."))

        # ── Gravar via OGR ──────────────────────────────────────────────
        temp_central_path = os.path.join(tmp, "temp_central.gpkg")

        srs = osr.SpatialReference()
        srs_ok = False
        try:
            ret = srs.ImportFromWkt(work_crs_qgs.toWkt())
            srs_ok = (ret == 0)
        except Exception:
            pass
        if not srs_ok:
            srs = None

        drv     = ogr.GetDriverByName("GPKG")
        ds_cent = drv.CreateDataSource(temp_central_path)
        lyr     = ds_cent.CreateLayer("central_routes", srs=srs, geom_type=ogr.wkbLineString)
        fdef    = ogr.FieldDefn("source", ogr.OFTString); fdef.SetWidth(32)
        lyr.CreateField(fdef)
        lyr_defn = lyr.GetLayerDefn()

        feat_count = 0
        BATCH_SIZE = 20_000
        lyr.StartTransaction()

        for smooth_line in merged_lines:
            if smooth_line.is_empty or smooth_line.length == 0:
                continue
            ogr_geom = ogr.CreateGeometryFromWkt(smooth_line.wkt)
            if ogr_geom is None:
                continue

            feat = ogr.Feature(lyr_defn)
            feat.SetGeometry(ogr_geom)
            feat.SetField("source", "central")
            lyr.CreateFeature(feat)
            feat = None
            feat_count += 1

            if feat_count % BATCH_SIZE == 0:
                lyr.CommitTransaction()
                lyr.StartTransaction()
                gc.collect()

        lyr.CommitTransaction()
        ds_cent = None
        del merged_lines
        gc.collect()
        feedback.pushInfo(self.tr(f"Vectorizacao concluida: {feat_count} segmentos gravados."))
        feedback.setProgress(58)

        # Dissolver linhas contíguas (reduz número de features e melhora topologia)
        feedback.pushInfo(self.tr("A dissolver e simplificar rotas centrais..."))
        central_routes = processing.run(
            "native:dissolve",
            {
                "INPUT":  temp_central_path,
                "FIELD":  [],
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context=context,
            feedback=feedback,
        )["OUTPUT"]

        try:
            os.remove(temp_central_path)
        except OSError:
            pass
        gc.collect()
        feedback.setProgress(62)

        # ── B. ROTAS MARGINAIS ──────────────────────────────────────────
        # Otimização: processar feature a feature em vez de carregar tudo
        # de uma vez; gravar directamente para ficheiro temporário.
        # ---------------------------------------------------------------
        feedback.pushInfo(self.tr("A gerar rotas marginais..."))

        nav_mediana = buffer_dist_m
        if gps_layer:
            feedback.pushInfo(
                self.tr(
                    "Camada GPS encontrada. "
                    "A usar valor de buffer como fallback (logica GPS nao implementada)."
                )
            )

        # Ler apenas as colunas geometry (evita carregar atributos desnecessários)
        water_gdf = _read_vector(vector_path)[["geometry"]].copy()
        water_gdf["geometry"] = (
            water_gdf.geometry
            .buffer(-abs(nav_mediana))   # buffer negativo → interior
            .boundary                    # fronteira = linha marginal
        )
        # Remover geometrias vazias (buffer negativo pode colapsar polígonos pequenos)
        water_gdf = water_gdf[~water_gdf.geometry.is_empty].reset_index(drop=True)

        temp_marg_path = os.path.join(tmp, "temp_marginais.gpkg")
        water_gdf.to_file(temp_marg_path, driver="GPKG")

        del water_gdf
        gc.collect()
        feedback.setProgress(65)

        # ── C. ROTAS TRANSVERSAIS ───────────────────────────────────────
        #
        # PROBLEMA com native:transect:
        #   • O parâmetro DISTANCE opera sobre cada segmento elementar da
        #     geometria, não sobre o comprimento total da linha. Após o
        #     dissolve, o esqueleto é uma MultiLineString com milhares de
        #     segmentos pequenos — o resultado é transectos a cada ~pixel,
        #     não a cada TRANSECT_INTERVAL metros.
        #   • O comprimento é fixo (LENGTH), não chega à margem real.
        #
        # SOLUÇÃO — implementação própria em Shapely:
        #   1. Iterar sobre todas as geometrias da rota central.
        #   2. Para cada geometry, percorrer o comprimento total em passos
        #      de TRANSECT_INTERVAL metros usando interpolate().
        #   3. Em cada ponto, calcular a direcção tangente da linha e
        #      gerar uma perpendicular longa (2× a largura do rio).
        #   4. Intersectar a perpendicular com o polígono da máscara
        #      de água para obter o segmento exacto dentro do rio.
        #   5. Gravar via OGR (sem aceder ao proj.db).
        # ---------------------------------------------------------------
        feedback.pushInfo(self.tr("A gerar rotas transversais (implementacao Shapely)..."))

        from rasterio.features  import shapes as rasterio_shapes
        from shapely.geometry   import LineString as ShpLineString, MultiLineString
        from shapely.ops        import unary_union
        from shapely.affinity   import rotate
        from qgis.core          import QgsVectorLayer as _QgsVL

        # Carregar a rota central como GeoDataFrame (já em memória QGIS)
        if isinstance(central_routes, str):
            cent_gdf = gpd.read_file(central_routes)
        else:
            # É uma camada QGIS em memória — exportar para gpkg temporário
            temp_cent_exp = os.path.join(tmp, "cent_export.gpkg")
            processing.run("native:savefeatures", {
                "INPUT": central_routes, "OUTPUT": temp_cent_exp,
            }, context=context, feedback=feedback)
            cent_gdf = gpd.read_file(temp_cent_exp)

        # Carregar o polígono da máscara de água para clip
        water_mask_gdf = _read_vector(vector_path)[["geometry"]]
        water_union    = water_mask_gdf.geometry.unary_union
        del water_mask_gdf

        # Estimar a largura máxima do rio para o comprimento da perpendicular
        # (usa a bbox do polígono de água como estimativa conservadora)
        water_bbox_diag = (
            (water_union.bounds[2] - water_union.bounds[0]) ** 2
            + (water_union.bounds[3] - water_union.bounds[1]) ** 2
        ) ** 0.5
        perp_half_len = min(water_bbox_diag, nav_mediana * 10)

        # Criar datasource OGR para os transectos
        temp_transect_path = os.path.join(tmp, "temp_transects.gpkg")
        drv_t   = ogr.GetDriverByName("GPKG")
        ds_t    = drv_t.CreateDataSource(temp_transect_path)
        srs_t   = osr.SpatialReference()
        srs_t.ImportFromWkt(work_crs_qgs.toWkt())
        lyr_t   = ds_t.CreateLayer("transects", srs=srs_t, geom_type=ogr.wkbLineString)
        fdef_t  = ogr.FieldDefn("source", ogr.OFTString)
        fdef_t.SetWidth(32)
        lyr_t.CreateField(fdef_t)
        lyr_defn_t = lyr_t.GetLayerDefn()
        lyr_t.StartTransaction()
        t_count = 0

        def _make_perpendicular(line_geom, dist_along, half_len):
            """
            Cria uma perpendicular à linha no ponto a dist_along do início.
            Usa dois pontos próximos para estimar a tangente local.
            Devolve um ShpLineString longo o suficiente para cruzar o rio,
            ou None se não for possível calcular a tangente.
            """
            delta = max(0.1, half_len * 0.001)   # incremento para tangente
            p0 = line_geom.interpolate(max(0.0, dist_along - delta))
            p1 = line_geom.interpolate(min(line_geom.length, dist_along + delta))
            dx = p1.x - p0.x
            dy = p1.y - p0.y
            norm = (dx**2 + dy**2) ** 0.5
            if norm < 1e-10:
                return None
            # Perpendicular: rodar 90° → (-dy, dx)
            px, py = -dy / norm, dx / norm
            cx = line_geom.interpolate(dist_along).x
            cy = line_geom.interpolate(dist_along).y
            return ShpLineString([
                (cx - px * half_len, cy - py * half_len),
                (cx + px * half_len, cy + py * half_len),
            ])

        total_geoms = len(cent_gdf)
        for gi, row in enumerate(cent_gdf.itertuples()):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            # Explodir MultiLineString em linhas simples
            parts = (
                list(geom.geoms)
                if geom.geom_type == "MultiLineString"
                else [geom]
            )
            for part in parts:
                length = part.length
                if length < transect_interval_m:
                    continue
                dist = 0.0
                while dist <= length:
                    perp = _make_perpendicular(part, dist, perp_half_len)
                    if perp is not None:
                        # Clipar com o polígono de água
                        clipped = perp.intersection(water_union)
                        if not clipped.is_empty and clipped.length > 1.0:
                            segs = (
                                list(clipped.geoms)
                                if clipped.geom_type in ("MultiLineString", "GeometryCollection")
                                else [clipped]
                            )
                            for seg in segs:
                                if seg.geom_type == "LineString" and seg.length > 1.0:
                                    ogr_g = ogr.CreateGeometryFromWkt(seg.wkt)
                                    if ogr_g:
                                        feat_t = ogr.Feature(lyr_defn_t)
                                        feat_t.SetGeometry(ogr_g)
                                        feat_t.SetField("source", "transversal")
                                        lyr_t.CreateFeature(feat_t)
                                        feat_t = None
                                        t_count += 1
                    dist += transect_interval_m

                    if t_count % 500 == 0 and t_count > 0:
                        lyr_t.CommitTransaction()
                        lyr_t.StartTransaction()
                        gc.collect()

            feedback.setProgress(65 + int((gi + 1) / max(total_geoms, 1) * 17))
            if feedback.isCanceled():
                lyr_t.CommitTransaction()
                ds_t = None
                return {}

        lyr_t.CommitTransaction()
        ds_t = None
        gc.collect()

        feedback.pushInfo(self.tr(f"Transversais concluidas: {t_count} rotas geradas."))
        clipped_transects = temp_transect_path
        feedback.setProgress(82)

        # ── D. MESCLAR REDES ────────────────────────────────────────────
        feedback.pushInfo(self.tr("A compilar a rede final..."))

        merged_network = processing.run(
            "native:mergevectorlayers",
            {
                "LAYERS": [central_routes, temp_marg_path, clipped_transects],
                "CRS":    work_crs_qgs,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context=context,
            feedback=feedback,
        )["OUTPUT"]

        # Remover geometrias com coordenadas inválidas (NaN/Inf) que causam
        # "Coordinates with non-finite values" ao gravar em shapefile/gpkg.
        # Isto pode ocorrer quando features de diferentes SRCs são mescladas
        # ou quando a reprojeção falha em geometrias degeneradas.
        feedback.pushInfo(self.tr("A validar geometrias da rede final..."))
        merged_network = processing.run(
            "native:fixgeometries",
            {
                "INPUT":  merged_network,
                "METHOD": 1,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context=context,
            feedback=feedback,
        )["OUTPUT"]

        del central_routes, clipped_transects
        gc.collect()
        feedback.setProgress(90)

        # ── E. REPROJETAR PARA O SRC DE SAÍDA (TEMPORARY_OUTPUT) ────────
        # O native:reprojectlayer com destino ficheiro preserva os FIDs
        # originais — quando há features de 3 camadas mescladas os FIDs
        # ficam duplicados e o GPKG falha com "UNIQUE constraint failed: fid".
        # Solução: reprojetar sempre para TEMPORARY_OUTPUT (camada em memória).
        # O destino final é controlado pelo OUTPUT_NETWORK do Processing,
        # que atribui FIDs novos e sequenciais automaticamente.

        if output_crs_param and output_crs_param.isValid():
            target_crs  = output_crs_param
            target_desc = output_crs_param.authid()
        else:
            target_crs  = work_crs_qgs
            target_desc = work_crs_qgs.authid()

        feedback.pushInfo(self.tr(f"A reprojetar a rede final para {target_desc}..."))

        reprojected = processing.run(
            "native:reprojectlayer",
            {
                "INPUT":      merged_network,
                "TARGET_CRS": target_crs,
                "OUTPUT":     "TEMPORARY_OUTPUT",   # evita FIDs duplicados no GPKG
            },
            context=context,
            feedback=feedback,
        )["OUTPUT"]

        del merged_network
        gc.collect()
        feedback.setProgress(95)

        # ── F. GRAVAR NA SAÍDA FINAL VIA FEATURESINK ──────────────────
        # O FeatureSink do QGIS Processing gere os FIDs automaticamente,
        # garantindo unicidade independentemente do formato de saída
        # (GPKG, SHP, GeoJSON, memória, etc.).
        feedback.pushInfo(self.tr("A gravar a rede final..."))

        # Abrir a camada reprojetada
        if isinstance(reprojected, str):
            lyr_final = QgsVectorLayer(reprojected, "rede_final", "ogr")
        else:
            lyr_final = reprojected   # já é QgsVectorLayer (TEMPORARY_OUTPUT)

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT_NETWORK, context,
            lyr_final.fields(), lyr_final.wkbType(), lyr_final.crs()
        )

        if sink is None:
            feedback.reportError(self.tr("Nao foi possivel criar a camada de saida."), fatalError=True)
            return {}

        total = lyr_final.featureCount()
        for n, feat in enumerate(lyr_final.getFeatures()):
            sink.addFeature(feat, QgsFeatureSink.FastInsert)
            if n % 5000 == 0:
                feedback.setProgress(95 + int(n / max(total, 1) * 5))
            if feedback.isCanceled():
                return {}

        del sink, lyr_final
        gc.collect()
        feedback.setProgress(100)
        feedback.pushInfo(self.tr("Rede fluvial concluida com sucesso."))

        return {self.OUTPUT_NETWORK: dest_id}
