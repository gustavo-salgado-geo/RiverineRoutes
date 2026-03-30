import os
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


def _reproject_vector_to_metric(vector_path: str, target_epsg: int, tmp_folder: str) -> str:
    """
    Reprojeta uma camada vetorial para o EPSG métrico indicado.
    Devolve o caminho para o ficheiro reprojetado (.gpkg).
    """
    gdf = gpd.read_file(vector_path)
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
        feedback.pushInfo(self.tr("A verificar SRC dos dados de entrada..."))

        # ---- 2a. Ler CRS do raster ----------------------------------------
        with rasterio.open(raster_path_orig) as src:
            raw_raster_crs = src.crs          # pode ser None se o ficheiro não tiver SRC
            raster_bounds  = src.bounds
            raster_transform = src.transform

        if raw_raster_crs is None:
            # Sem SRC no raster: tentar ler do vetor para decidir o UTM.
            # Se o vetor também não tiver SRC, o processamento é interrompido.
            feedback.pushWarning(
                self.tr(
                    "AVISO: O raster não tem SRC definido (CRS ausente no ficheiro). "
                    "O algoritmo tentará usar o SRC do vetor para determinar a zona UTM. "
                    "Se o resultado não for correto, defina o SRC do raster no QGIS "
                    "(Raster → Projeções → Atribuir SRC) e execute novamente."
                )
            )
            raster_crs_wkt = None
            raster_is_geo  = None       # indeterminado — será resolvido via vetor
        else:
            raster_crs_wkt = raw_raster_crs.to_wkt()
            raster_is_geo  = _is_geographic(raster_crs_wkt)

        # ---- 2b. Ler CRS do vetor -----------------------------------------
        gdf_check = gpd.read_file(vector_path_orig)

        if gdf_check.crs is None:
            if raster_crs_wkt is None:
                # Nenhum dos dois tem SRC — não é possível continuar
                feedback.reportError(
                    self.tr(
                        "ERRO: Nem o raster nem o vetor têm SRC definido. "
                        "Atribua o SRC correto a ambas as camadas no QGIS e tente novamente."
                    ),
                    fatalError=True,
                )
                return {}
            # Vetor sem SRC mas raster tem: assumir mesmo SRC que o raster
            feedback.pushWarning(
                self.tr(
                    "AVISO: O vetor não tem SRC definido. "
                    "Assumindo o mesmo SRC do raster. "
                    "Verifique se este comportamento está correto para os seus dados."
                )
            )
            vector_epsg   = None
            vector_is_geo = raster_is_geo
        else:
            vector_epsg   = gdf_check.crs.to_epsg()
            vector_is_geo = gdf_check.crs.is_geographic

        # ---- 2c. Determinar EPSG de trabalho (métrico) --------------------
        #
        # Ordem de prioridade:
        #   1. CRS do raster, se métrico
        #   2. CRS do vetor, se métrico
        #   3. Auto-UTM calculado a partir do centro do raster (se geográfico)
        #   4. Auto-UTM calculado a partir do centro do vetor (se geográfico)

        work_epsg = None

        if raster_crs_wkt is not None and not raster_is_geo:
            try:
                proj_crs = ProjCRS.from_wkt(raster_crs_wkt)
                auth     = proj_crs.to_authority()
                work_epsg = int(auth[1]) if auth else None
            except Exception:
                work_epsg = None
            feedback.pushInfo(
                self.tr(
                    f"SRC do raster já é métrico "
                    f"{'(EPSG:' + str(work_epsg) + ')' if work_epsg else '(EPSG desconhecido)'}. "
                    f"Usando como SRC de trabalho."
                )
            )
            raster_path = raster_path_orig

        elif raster_crs_wkt is not None and raster_is_geo:
            lon, lat  = _get_raster_center_lonlat(raster_path_orig)
            work_epsg = _auto_utm_epsg(lon, lat)
            feedback.pushInfo(
                self.tr(
                    f"SRC do raster é geográfico (graus). "
                    f"Reprojetando automaticamente para EPSG:{work_epsg} (UTM)..."
                )
            )
            raster_path = _reproject_raster_to_metric(raster_path_orig, work_epsg, tmp)

        else:
            # raster sem SRC — tentar via vetor
            if gdf_check.crs is not None:
                if vector_is_geo:
                    cx = gdf_check.geometry.unary_union.centroid.x
                    cy = gdf_check.geometry.unary_union.centroid.y
                    work_epsg = _auto_utm_epsg(cx, cy)
                    feedback.pushInfo(
                        self.tr(
                            f"Raster sem SRC. SRC do vetor é geográfico. "
                            f"Usando EPSG:{work_epsg} (UTM auto) como SRC de trabalho."
                        )
                    )
                else:
                    work_epsg = vector_epsg
                    feedback.pushInfo(
                        self.tr(
                            f"Raster sem SRC. Usando SRC do vetor "
                            f"(EPSG:{work_epsg}) como SRC de trabalho."
                        )
                    )
            else:
                feedback.reportError(
                    self.tr(
                        "ERRO: Não foi possível determinar um SRC de trabalho. "
                        "Atribua o SRC correto ao raster ou ao vetor e tente novamente."
                    ),
                    fatalError=True,
                )
                return {}

            # Reprojetar raster sem SRC: atribuir o work_epsg primeiro,
            # depois reprojetar se necessário (aqui apenas atribuímos pois
            # não sabemos a projeção original — o utilizador foi avisado).
            feedback.pushWarning(
                self.tr(
                    f"Raster sem SRC: será processado assumindo EPSG:{work_epsg}. "
                    f"Se os resultados forem incorretos, atribua o SRC real ao raster."
                )
            )
            raster_path = raster_path_orig

        # ---- 2d. Reprojetar vetor se necessário ---------------------------
        needs_vector_reproject = False
        if gdf_check.crs is None:
            needs_vector_reproject = False   # sem CRS — usamos como está, já avisamos
        elif vector_is_geo:
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
            feedback.pushInfo(self.tr("SRC do vetor compatível. Sem necessidade de reprojeção."))
            vector_path = vector_path_orig

        feedback.pushInfo(
            self.tr(
                f"Parâmetros de distância — Buffer: {buffer_dist_m} m | "
                f"Transecto: {transect_interval_m} m"
            )
        )

        # ── A. ROTAS CENTRAIS ───────────────────────────────────────────
        feedback.pushInfo(self.tr("A gerar rotas centrais (esqueletonização)..."))

        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            transform   = src.transform
            crs         = src.crs

        skeleton      = skeletonize(raster_data)
        skeleton_uint8 = (skeleton > 0).astype(np.uint8)

        temp_skel_path = os.path.join(tmp, "temp_skeleton.tif")
        with rasterio.open(
            temp_skel_path, "w", driver="GTiff",
            height=skeleton_uint8.shape[0],
            width=skeleton_uint8.shape[1],
            count=1,
            dtype=skeleton_uint8.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(skeleton_uint8, 1)

        central_lines = processing.run(
            "native:pixelstopolygons",
            {
                "INPUT_RASTER": temp_skel_path,
                "RASTER_BAND": 1,
                "FIELD_NAME": "is_water",
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context=context,
            feedback=feedback,
        )["OUTPUT"]

        central_routes = processing.run(
            "native:polygonstolines",
            {"INPUT": central_lines, "OUTPUT": "TEMPORARY_OUTPUT"},
            context=context,
            feedback=feedback,
        )["OUTPUT"]

        # ── B. ROTAS MARGINAIS ──────────────────────────────────────────
        feedback.pushInfo(self.tr("A gerar rotas marginais..."))

        water_gdf = gpd.read_file(vector_path)

        nav_mediana = buffer_dist_m  # já em metros
        if gps_layer:
            feedback.pushInfo(
                self.tr(
                    "Camada GPS encontrada. "
                    "Implementar aqui o cálculo da mediana navegável se desejado. "
                    "A usar valor de buffer definido como fallback."
                )
            )
            # Placeholder: substituir pela lógica do script A.2 quando disponível
            nav_mediana = buffer_dist_m

        water_gdf["marginal_geom"] = water_gdf.geometry.buffer(-abs(nav_mediana))
        water_gdf["geometry"]      = water_gdf["marginal_geom"].boundary

        temp_marg_path = os.path.join(tmp, "temp_marginais.gpkg")
        water_gdf.drop(columns=["marginal_geom"]).to_file(temp_marg_path, driver="GPKG")

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

        # ── D. MESCLAR REDES ────────────────────────────────────────────
        feedback.pushInfo(self.tr("A compilar a rede final..."))

        # Definir o SRC de trabalho para o merge
        work_crs_qgs = QgsCoordinateReferenceSystem(f"EPSG:{work_epsg}") if work_epsg else context.project().crs()

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

        # ── E. REPROJETAR SAÍDA (se definido pelo utilizador) ───────────
        if output_crs_param and output_crs_param.isValid():
            out_epsg = output_crs_param.postgisSrid()
            feedback.pushInfo(
                self.tr(f"A reprojetar a rede final para o SRC de saída definido (EPSG:{out_epsg})...")
            )
            final_network = processing.run(
                "native:reprojectlayer",
                {
                    "INPUT":     merged_network,
                    "TARGET_CRS": output_crs_param,
                    "OUTPUT":    parameters[self.OUTPUT_NETWORK],
                },
                context=context,
                feedback=feedback,
            )["OUTPUT"]
        else:
            # Sem SRC de saída definido: guardar directamente no SRC de trabalho
            feedback.pushInfo(
                self.tr(
                    "Nenhum SRC de saída definido. "
                    f"A rede será gerada no SRC de processamento (EPSG:{work_epsg})."
                )
            )
            final_network = processing.run(
                "native:reprojectlayer",
                {
                    "INPUT":     merged_network,
                    "TARGET_CRS": work_crs_qgs,
                    "OUTPUT":    parameters[self.OUTPUT_NETWORK],
                },
                context=context,
                feedback=feedback,
            )["OUTPUT"]

        return {self.OUTPUT_NETWORK: final_network}
