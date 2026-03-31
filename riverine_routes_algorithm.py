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
)

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS as ProjCRS
from skimage.morphology import skeletonize
from shapely.geometry import LineString
from shapely.ops import nearest_points


# ---------------------------------------------------------------------------
# Funções utilitárias de CRS
# ---------------------------------------------------------------------------

def _is_geographic(crs_wkt: str) -> bool:
    """Devolve True se o CRS está em graus (geográfico), False se já é métrico."""
    try:
        proj_crs = ProjCRS.from_wkt(crs_wkt)
        return proj_crs.is_geographic
    except Exception:
        # Fallback heurístico: EPSG:4326 e variantes comuns
        wkt_upper = crs_wkt.upper()
        geographic_hints = ["GEOGCS", "GEOGRAPHIC", "DEGREE", "DECIMAL_DEGREE"]
        return any(hint in wkt_upper for hint in geographic_hints)


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
    """Devolve (lon, lat) do centro do raster no CRS geográfico WGS84."""
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        cx = (bounds.left + bounds.right) / 2
        cy = (bounds.bottom + bounds.top) / 2
        raster_crs = ProjCRS.from_wkt(src.crs.to_wkt())
        if raster_crs.is_geographic:
            return cx, cy
        # Converter centro para WGS84
        from pyproj import Transformer
        transformer = Transformer.from_crs(raster_crs, ProjCRS.from_epsg(4326), always_xy=True)
        return transformer.transform(cx, cy)


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
            defaultValue=50.0,
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
            defaultValue=100.0,
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

            # Tentativa 2: pyproj (fallback para SRCs não-EPSG)
            if work_epsg is None:
                try:
                    proj_crs = ProjCRS.from_wkt(raster_crs_wkt)
                    auth     = proj_crs.to_authority()
                    if auth:
                        work_epsg = int(auth[1])
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

        feedback.pushInfo(self.tr("Fase 2/3: a esqueletonizar (toda a imagem em RAM)..."))

        # FASE 2: carregar uint8 comprimido → bool → skeletonize → uint8
        with rasterio.open(temp_uint8_path) as src:
            data_uint8 = src.read(1)                    # uint8, menor footprint
        feedback.setProgress(30)

        data_bool  = data_uint8 > 0                     # bool (1 bit lógico, 1 byte real)
        del data_uint8
        gc.collect()

        skeleton_bool = skeletonize(data_bool)           # bool
        del data_bool
        gc.collect()
        feedback.setProgress(40)

        skeleton_uint8 = skeleton_bool.astype(np.uint8)  # uint8
        del skeleton_bool
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

        feedback.pushInfo(self.tr("Esqueleto gravado. A vectorizar via rasterio.features (streaming)..."))

        # ── VECTORIZAÇÃO DO ESQUELETO ───────────────────────────────────
        #
        # PROBLEMA com native:pixelstopolygons:
        #   Gera um polígono por pixel — para um raster de 500 MB pode
        #   produzir dezenas de milhões de polígonos em memória RAM de
        #   uma só vez, travando o sistema.
        #
        # SOLUÇÃO — rasterio.features.shapes() em streaming:
        #   • Lê o raster em blocos de linhas (~32 MB por bloco).
        #   • O gerador shapes() emite geometrias uma a uma — nunca
        #     guarda todas na RAM simultaneamente.
        #   • Apenas os pixels com valor > 0 (esqueleto) geram geometrias.
        #   • As geometrias são acumuladas numa lista leve de dicts e
        #     gravadas em disco via Fiona em modo append, batch a batch.
        #   • Depois de gravar, a lista é limpa e a RAM libertada.
        #   • simplify() com tolerância = 1 pixel remove vértices
        #     redundantes dos quadrados de pixels antes de gravar,
        #     reduzindo drasticamente o tamanho do ficheiro de saída.
        # ---------------------------------------------------------------
        import fiona
        import fiona.crs
        from rasterio.features import shapes as rasterio_shapes
        from shapely.geometry import shape as shapely_shape, mapping

        temp_central_path = os.path.join(tmp, "temp_central.gpkg")

        with rasterio.open(temp_skel_path) as src:
            skel_transform = src.transform
            skel_crs       = src.crs
            skel_height    = src.height
            skel_width     = src.width
            pixel_size     = abs(skel_transform.a)   # tamanho do pixel em unidades do SRC

            # Tolerância de simplificação: 1 pixel (remove artefactos de grade)
            simplify_tol = pixel_size * 1.0

            # Linhas por bloco: ~32 MB de RAM por leitura
            rows_per_block = max(1, min(skel_height, (32 * 1024 * 1024) // max(skel_width, 1)))
            n_blocks = math.ceil(skel_height / rows_per_block)

            # Schema Fiona: apenas a geometria (LineString/MultiLineString)
            schema = {
                "geometry": "LineString",
                "properties": {"source": "str"},
            }
            # CRS para o Fiona: preferir EPSG numérico, fallback para WKT do QGIS
            if work_epsg:
                fiona_crs = fiona.crs.from_epsg(work_epsg)
            else:
                try:
                    fiona_crs = fiona.crs.from_wkt(work_crs_qgs.toWkt())
                except Exception:
                    fiona_crs = {}

            # Gravar em streaming — abrir ficheiro uma vez, escrever batch a batch
            BATCH_SIZE = 50_000   # features por flush para disco
            batch = []

            with fiona.open(
                temp_central_path, "w",
                driver="GPKG",
                crs=fiona_crs,
                schema=schema,
            ) as dst_fiona:

                for i, row_off in enumerate(range(0, skel_height, rows_per_block)):
                    actual_rows = min(rows_per_block, skel_height - row_off)
                    window      = rasterio.windows.Window(0, row_off, skel_width, actual_rows)
                    block       = src.read(1, window=window)

                    if block.max() == 0:
                        # Bloco vazio — saltar sem processar
                        del block
                        continue

                    # Calcular o transform do bloco (offset Y correcto)
                    block_transform = rasterio.windows.transform(window, skel_transform)

                    # Gerar geometrias via gerador (streaming — sem lista intermédia)
                    for geom_dict, val in rasterio_shapes(block, transform=block_transform):
                        if val == 0:
                            continue
                        geom = shapely_shape(geom_dict)
                        # Converter polígono de pixel → linha de contorno e simplificar
                        line = geom.boundary.simplify(simplify_tol, preserve_topology=False)
                        if line.is_empty:
                            continue
                        # Explodir MultiLineString em LineStrings individuais
                        parts = list(line.geoms) if line.geom_type == "MultiLineString" else [line]
                        for part in parts:
                            if not part.is_empty and part.length > 0:
                                batch.append({
                                    "geometry": mapping(part),
                                    "properties": {"source": "central"},
                                })
                        # Flush para disco quando o batch atinge o limite
                        if len(batch) >= BATCH_SIZE:
                            dst_fiona.writerecords(batch)
                            batch.clear()
                            gc.collect()

                    del block
                    gc.collect()
                    feedback.setProgress(50 + int((i + 1) / n_blocks * 8))  # 50-58%
                    if feedback.isCanceled():
                        return {}

                # Flush final
                if batch:
                    dst_fiona.writerecords(batch)
                    batch.clear()

        del batch
        gc.collect()
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
        feedback.pushInfo(self.tr("A gerar rotas transversais (Transectos)..."))

        transects = processing.run(
            "native:transect",
            {
                "INPUT":    central_routes,
                "LENGTH":   nav_mediana * 1.5,
                "DISTANCE": transect_interval_m,
                "SIDE":     2,
                "OUTPUT":   "TEMPORARY_OUTPUT",
            },
            context=context,
            feedback=feedback,
        )["OUTPUT"]
        feedback.setProgress(75)

        clipped_transects = processing.run(
            "native:clip",
            {
                "INPUT":   transects,
                "OVERLAY": vector_path,
                "OUTPUT":  "TEMPORARY_OUTPUT",
            },
            context=context,
            feedback=feedback,
        )["OUTPUT"]

        del transects
        gc.collect()
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

        # ── E. REPROJETAR SAÍDA (se definido pelo utilizador) ───────────
        if output_crs_param and output_crs_param.isValid():
            out_epsg = output_crs_param.postgisSrid()
            feedback.pushInfo(
                self.tr(
                    f"A reprojetar a rede final para EPSG:{out_epsg}..."
                )
            )
            final_network = processing.run(
                "native:reprojectlayer",
                {
                    "INPUT":      merged_network,
                    "TARGET_CRS": output_crs_param,
                    final_path = os.path.join(tmp, "final_network.gpkg")  "OUTPUT": final_path,
                },
                context=context,
                feedback=feedback,
            )["OUTPUT"]
        else:
            feedback.pushInfo(
                self.tr(
                    f"Nenhum SRC de saida definido. "
                    f"Rede gerada no SRC de processamento (EPSG:{work_epsg})."
                )
            )
            final_network = processing.run(
                "native:reprojectlayer",
                {
                    "INPUT":      merged_network,
                    "TARGET_CRS": work_crs_qgs,
                    final_path = os.path.join(tmp, "final_network.gpkg")  "OUTPUT": final_path,
                },
                context=context,
                feedback=feedback,
            )["OUTPUT"]

        feedback.setProgress(100)
        feedback.pushInfo(self.tr("Rede fluvial concluida com sucesso."))
        return {self.OUTPUT_NETWORK: final_network}
