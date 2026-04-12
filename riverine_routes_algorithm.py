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
    QgsProcessingParameterRasterDestination,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsFeatureSink,
    QgsVectorLayer,
    QgsFeature,
    QgsFields,
    QgsField,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant

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
    """Devolve True se o CRS está em graus (geográfico), False se já é métrico."""
    try:
        from osgeo import osr
        srs = osr.SpatialReference()
        ret = srs.ImportFromWkt(crs_wkt)
        if ret == 0:
            return bool(srs.IsGeographic())
    except Exception:
        pass
    wkt_upper = crs_wkt.upper()
    return any(h in wkt_upper for h in ["GEOGCS", "GEOGRAPHIC", "DEGREE", "DECIMAL_DEGREE"])


def _auto_utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def _get_raster_center_lonlat(raster_path: str):
    from osgeo import osr, ogr as _ogr
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        cx = (bounds.left + bounds.right) / 2
        cy = (bounds.bottom + bounds.top) / 2
        crs_wkt = src.crs.to_wkt() if src.crs else None

    if crs_wkt is None:
        return cx, cy

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(crs_wkt)

    if src_srs.IsGeographic():
        return cx, cy

    dst_srs = osr.SpatialReference()
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
    if "|layername=" in qgis_source:
        parts = qgis_source.split("|layername=", 1)
        return parts[0], parts[1]
    clean_path = qgis_source.split("|")[0]
    return clean_path, None


def _read_vector(qgis_source: str) -> "gpd.GeoDataFrame":
    path, layer = _parse_vector_source(qgis_source)
    if layer:
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)


def _reproject_vector_to_metric(qgis_source: str, target_epsg: int, tmp_folder: str) -> str:
    gdf = _read_vector(qgis_source)
    gdf_reproj = gdf.to_crs(epsg=target_epsg)
    out_path = os.path.join(tmp_folder, f"vector_reproj_{target_epsg}.gpkg")
    gdf_reproj.to_file(out_path, driver="GPKG")
    return out_path


# ---------------------------------------------------------------------------
# Funções auxiliares para centerlines internas (uso nas transversais)
# ---------------------------------------------------------------------------

def _rasterize_vector_to_tiff(vector_path: str, reference_raster_path: str,
                               out_tiff_path: str, burn_value: int = 1) -> str:
    """
    Rasteriza um vetor de polígonos usando as mesmas dimensões/transform
    do raster de referência. Devolve o caminho do tiff gerado.
    Usa rasterio.features.rasterize — sem dependência do GDAL CLI.
    """
    import rasterio.features

    gdf = _read_vector(vector_path)

    with rasterio.open(reference_raster_path) as ref:
        ref_transform = ref.transform
        ref_shape     = (ref.height, ref.width)
        ref_crs       = ref.crs
        profile = {
            "driver":    "GTiff",
            "dtype":     "uint8",
            "count":     1,
            "height":    ref.height,
            "width":     ref.width,
            "crs":       ref_crs,
            "transform": ref_transform,
            "compress":  "lzw",
        }

    # Reprojetar vetor para o CRS do raster de referência se necessário
    if gdf.crs is not None and ref_crs is not None:
        import pyproj
        from pyproj import CRS as ProjCRS
        ref_crs_str = ref_crs.to_wkt() if hasattr(ref_crs, "to_wkt") else str(ref_crs)
        gdf = gdf.to_crs(ref_crs_str)

    shapes = ((geom, burn_value) for geom in gdf.geometry if geom is not None and not geom.is_empty)

    burned = rasterio.features.rasterize(
        shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype="uint8",
    )

    with rasterio.open(out_tiff_path, "w", **profile) as dst:
        dst.write(burned, 1)

    return out_tiff_path


def _compute_internal_centerlines(
    vector_path: str,
    reference_raster_path: str,
    buffer_dist_m: float,
    pixel_size_m: float,
    tmp_folder: str,
    feedback=None,
) -> list:
    """
    Gera centerlines INTERNAS para uso exclusivo no posicionamento das
    rotas transversais. Estas centerlines NÃO são entregues como output.

    Pipeline interno (totalmente temporário):
      1. Ler vetor de água
      2. Buffer POSITIVO com buffer_dist_m  → polígono "zona navegável interna"
      3. Buffer NEGATIVO com buffer_dist_m  → colapso para canal central
      4. Rasterizar o resultado
      5. Skeletonize + thin + pruning (igual ao pipeline das rotas centrais)
      6. Rastreio de grafo → lista de ShpLineString

    Devolve lista de shapely.geometry.LineString já em coordenadas reais.
    """
    from shapely.geometry import LineString as ShpLS
    from shapely.ops      import linemerge as shp_lm
    from scipy.ndimage    import label as ndl, convolve as ndc
    import rasterio.features

    def _log(msg):
        if feedback:
            feedback.pushInfo(msg)

    # ── 1. Ler vetor e aplicar buffer positivo → negativo ──────────────
    _log("  [Transversais] Lendo vetor de agua para centerlines internas...")
    water_gdf = _read_vector(vector_path)[["geometry"]]
    water_union_raw = water_gdf.geometry.unary_union
    del water_gdf
    gc.collect()

    _log(f"  [Transversais] Buffer positivo de {buffer_dist_m:.1f}m...")
    buffered_pos = water_union_raw.buffer(buffer_dist_m)

    _log(f"  [Transversais] Buffer negativo de {buffer_dist_m:.1f}m (colapso para canal)...")
    buffered_neg = buffered_pos.buffer(-buffer_dist_m)
    del buffered_pos
    gc.collect()

    if buffered_neg.is_empty:
        _log("  [Transversais] AVISO: buffer negativo resultou em geometria vazia. "
             "Usando uniao original como fallback.")
        buffered_neg = water_union_raw

    del water_union_raw
    gc.collect()

    # ── 2. Rasterizar para tiff temporário ─────────────────────────────
    _log("  [Transversais] Rasterizando canal colapsado...")
    tmp_vec_path  = os.path.join(tmp_folder, "_internal_canal_collapsed.gpkg")
    tmp_rast_path = os.path.join(tmp_folder, "_internal_canal_raster.tif")

    # Gravar geometria colapsada num gpkg temporário
    import geopandas as _gpd2
    from shapely.geometry import mapping as shp_mapping
    gdf_collapsed = _gpd2.GeoDataFrame(
        geometry=[buffered_neg] if buffered_neg.geom_type != "GeometryCollection"
                  else list(buffered_neg.geoms),
        crs=None,
    )
    # Atribuir CRS a partir do raster de referência
    with rasterio.open(reference_raster_path) as _ref:
        _ref_crs = _ref.crs
        _ref_transform = _ref.transform
        _ref_shape = (_ref.height, _ref.width)
        _ref_profile = {
            "driver":    "GTiff",
            "dtype":     "uint8",
            "count":     1,
            "height":    _ref.height,
            "width":     _ref.width,
            "crs":       _ref.crs,
            "transform": _ref.transform,
            "compress":  "lzw",
        }

    if _ref_crs:
        gdf_collapsed = gdf_collapsed.set_crs(_ref_crs, allow_override=True)

    shapes_burn = (
        (geom, 1) for geom in gdf_collapsed.geometry
        if geom is not None and not geom.is_empty
    )
    burned_arr = rasterio.features.rasterize(
        shapes_burn,
        out_shape=_ref_shape,
        transform=_ref_transform,
        fill=0,
        dtype="uint8",
    )
    with rasterio.open(tmp_rast_path, "w", **_ref_profile) as _dst:
        _dst.write(burned_arr, 1)
    del gdf_collapsed, burned_arr
    gc.collect()

    # ── 3. Pré-processamento morfológico + skeletonize ─────────────────
    _log("  [Transversais] Skeletonizando canal colapsado...")
    with rasterio.open(tmp_rast_path) as _src:
        canal_arr   = _src.read(1).astype(bool)
        skel_transform = _src.transform
        skel_pixel_size = abs(_src.transform.a)

    close_r       = max(1, int(round((buffer_dist_m * 0.10) / max(skel_pixel_size, 1e-6))))
    open_r        = max(1, int(round((buffer_dist_m * 0.05) / max(skel_pixel_size, 1e-6))))
    min_island_px = max(10, int(3.14159 * ((buffer_dist_m * 0.5) / max(skel_pixel_size, 1e-6)) ** 2))

    canal_arr = remove_small_objects(canal_arr, min_size=min_island_px, connectivity=2)
    canal_arr = binary_closing(canal_arr, footprint=disk(close_r))
    canal_arr = binary_opening(canal_arr, footprint=disk(open_r))
    gc.collect()

    skel_bool = skeletonize(canal_arr)
    del canal_arr
    gc.collect()

    skel_bool = skimage_thin(skel_bool)
    gc.collect()

    # Pruning
    prune_px       = max(3, int(round((buffer_dist_m * 0.5) / max(skel_pixel_size, 1e-6))))
    nb_kernel      = np.ones((3, 3), dtype=np.uint8)
    nb_kernel[1,1] = 0
    skel_work = skel_bool.copy()
    del skel_bool
    gc.collect()

    for _ in range(prune_px):
        if not skel_work.any():
            break
        nb_count = nd_convolve(skel_work.astype(np.uint8), nb_kernel, mode="constant", cval=0)
        terminals_p = skel_work & (nb_count == 1)
        if not terminals_p.any():
            break
        skel_work = skel_work & ~terminals_p

    # ── 4. Rastreio de grafo → LineStrings ─────────────────────────────
    _log("  [Transversais] Rastreando grafo do esqueleto interno...")

    n_neighbors_int = nd_convolve(
        skel_work.astype(np.uint8), nb_kernel, mode="constant", cval=0
    )
    terminals_int = skel_work & (n_neighbors_int == 1)
    junctions_int = skel_work & (n_neighbors_int >= 3)
    skel_work     = skel_work & (n_neighbors_int >= 1)
    node_mask_int = terminals_int | junctions_int

    NEIGHBORS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def _rc_to_xy_int(r, c):
        x, y = rasterio.transform.xy(skel_transform, r, c, offset="center")
        return float(x), float(y)

    visited_edges_int = set()
    lines_rc_int = []
    node_rows_int, node_cols_int = np.where(node_mask_int)

    def _all_nb_int(r, c):
        nb = []
        for dr, dc in NEIGHBORS_8:
            nr, nc_ = r+dr, c+dc
            if 0 <= nr < skel_work.shape[0] and 0 <= nc_ < skel_work.shape[1]:
                if skel_work[nr, nc_]:
                    nb.append((nr, nc_))
        return nb

    for nr0, nc0 in zip(node_rows_int.tolist(), node_cols_int.tolist()):
        for (vr, vc) in _all_nb_int(nr0, nc0):
            edge_key = (min((nr0,nc0),(vr,vc)), max((nr0,nc0),(vr,vc)))
            if edge_key in visited_edges_int:
                continue
            chain = [(nr0, nc0), (vr, vc)]
            visited_edges_int.add(edge_key)
            cur_r, cur_c = vr, vc
            if node_mask_int[cur_r, cur_c]:
                if len(chain) >= 2:
                    lines_rc_int.append(chain)
                continue
            prev_r, prev_c = nr0, nc0
            while True:
                nbs = [(r, c) for r, c in _all_nb_int(cur_r, cur_c)
                       if (r, c) != (prev_r, prev_c)]
                if not nbs:
                    break
                if len(nbs) == 1:
                    next_r, next_c = nbs[0]
                    e2 = (min((cur_r,cur_c),(next_r,next_c)),
                          max((cur_r,cur_c),(next_r,next_c)))
                    chain.append((next_r, next_c))
                    visited_edges_int.add(e2)
                    prev_r, prev_c = cur_r, cur_c
                    cur_r, cur_c = next_r, next_c
                    if node_mask_int[cur_r, cur_c]:
                        break
                else:
                    break
            if len(chain) >= 2:
                lines_rc_int.append(chain)

    # Segunda passagem: pixels orphaned
    visited_flat_int = set()
    for chain in lines_rc_int:
        for px in chain:
            visited_flat_int.add(px)

    orphaned_int = skel_work.copy()
    for r, c in visited_flat_int:
        orphaned_int[r, c] = False

    if orphaned_int.any():
        struct_8_int = np.ones((3, 3), dtype=bool)
        labeled_orp, n_grps = nd_label(orphaned_int, structure=struct_8_int)
        for grp in range(1, n_grps + 1):
            grp_mask = labeled_orp == grp
            grp_rows, grp_cols = np.where(grp_mask)
            if len(grp_rows) < 2:
                continue
            sr2, sc2 = int(grp_rows[0]), int(grp_cols[0])
            best_n = 9
            for ri, ci in zip(grp_rows.tolist(), grp_cols.tolist()):
                n_nb = sum(1 for dr, dc in NEIGHBORS_8
                           if 0 <= ri+dr < grp_mask.shape[0]
                           and 0 <= ci+dc < grp_mask.shape[1]
                           and grp_mask[ri+dr, ci+dc])
                if n_nb < best_n:
                    best_n = n_nb
                    sr2, sc2 = int(ri), int(ci)
                if best_n == 1:
                    break
            chain2, v2 = [], np.zeros(skel_work.shape, dtype=bool)
            cur_r2, cur_c2 = sr2, sc2
            while True:
                v2[cur_r2, cur_c2] = True
                chain2.append((cur_r2, cur_c2))
                next_r2, next_c2 = -1, -1
                for dr, dc in NEIGHBORS_8:
                    nr2, nc2 = cur_r2+dr, cur_c2+dc
                    if (0 <= nr2 < skel_work.shape[0] and 0 <= nc2 < skel_work.shape[1]
                            and grp_mask[nr2, nc2] and not v2[nr2, nc2]):
                        next_r2, next_c2 = nr2, nc2
                        break
                if next_r2 == -1:
                    break
                cur_r2, cur_c2 = next_r2, next_c2
            if len(chain2) >= 2:
                lines_rc_int.append(chain2)

    del skel_work, orphaned_int, node_mask_int
    del terminals_int, junctions_int, n_neighbors_int
    gc.collect()

    # ── 5. Converter cadeias → LineStrings ──────────────────────────────
    smooth_window_int = max(2, int(round((buffer_dist_m * 0.25) / max(skel_pixel_size, 1e-6))))

    def _ma_smooth_int(coords, window, pt_start, pt_end):
        n = len(coords)
        if n < 3:
            return coords
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        sx, sy = [], []
        for i in range(n):
            lo = max(0, i - window)
            hi = min(n - 1, i + window)
            count = hi - lo + 1
            sx.append(sum(xs[lo:hi+1]) / count)
            sy.append(sum(ys[lo:hi+1]) / count)
        sx[0],  sy[0]  = pt_start[0], pt_start[1]
        sx[-1], sy[-1] = pt_end[0],   pt_end[1]
        return list(zip(sx, sy))

    shapely_lines_int = []
    for chain in lines_rc_int:
        if len(chain) < 2:
            continue
        coords = [_rc_to_xy_int(r, c) for r, c in chain]
        if len(coords) < 2:
            continue
        pt_start = coords[0]
        pt_end   = coords[-1]
        smoothed = _ma_smooth_int(coords, smooth_window_int, pt_start, pt_end)
        if len(smoothed) < 2:
            continue
        line = ShpLS(smoothed)
        if not line.is_empty and line.length > 0:
            shapely_lines_int.append(line)

    del lines_rc_int
    gc.collect()

    # Linemerge final
    merged_int = shp_lm(shapely_lines_int)
    del shapely_lines_int
    gc.collect()

    if merged_int.geom_type == "LineString":
        result_lines = [merged_int]
    elif merged_int.geom_type == "MultiLineString":
        result_lines = list(merged_int.geoms)
    else:
        result_lines = [g for g in getattr(merged_int, "geoms", [])
                        if g.geom_type == "LineString"]

    # Limpar ficheiros temporários internos
    for _tmp_f in [tmp_vec_path, tmp_rast_path]:
        try:
            os.remove(_tmp_f)
        except OSError:
            pass

    _log(f"  [Transversais] Centerlines internas: {len(result_lines)} linhas geradas.")
    return result_lines


# ---------------------------------------------------------------------------
# Algoritmo principal
# ---------------------------------------------------------------------------

class RiverineRoutesAlgorithm(QgsProcessingAlgorithm):
    """
    Algoritmo para criação da rede completa de rotas fluviais:
    Centrais, Marginais e Transversais.
    """

    INPUT_RASTER      = "INPUT_RASTER"
    INPUT_VECTOR      = "INPUT_VECTOR"
    INPUT_LIMITS      = "INPUT_LIMITS"
    INPUT_GPS_TRACKS  = "INPUT_GPS_TRACKS"
    BUFFER_DIST       = "BUFFER_DIST"
    TRANSECT_INTERVAL = "TRANSECT_INTERVAL"
    OUTPUT_CRS        = "OUTPUT_CRS"
    OUTPUT_SKELETON   = "OUTPUT_SKELETON"
    OUTPUT_CENTRAL    = "OUTPUT_CENTRAL"
    OUTPUT_NETWORK    = "OUTPUT_NETWORK"

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

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr("Máscara Binária de Água (Raster)"),
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VECTOR,
                self.tr("Máscara de Água (Vetor)"),
                [QgsProcessing.TypeVectorPolygon],
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_LIMITS,
                self.tr("Limites da Área de Estudo (Polígonos)"),
                [QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_GPS_TRACKS,
                self.tr("Ficheiro de Rastreio GPS (Opcional)"),
                [QgsProcessing.TypeVectorPoint],
                optional=True,
            )
        )

        param_buffer = QgsProcessingParameterNumber(
            self.BUFFER_DIST,
            self.tr("Distância do Buffer Marginal (metros)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=200.0,
            minValue=1.0,
        )
        param_buffer.setMetadata({"widget_wrapper": {"decimals": 1}})
        self.addParameter(param_buffer)

        param_transect = QgsProcessingParameterNumber(
            self.TRANSECT_INTERVAL,
            self.tr("Distância entre Rotas Transversais (metros)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=1000.0,
            minValue=1.0,
        )
        param_transect.setMetadata({"widget_wrapper": {"decimals": 1}})
        self.addParameter(param_transect)

        self.addParameter(
            QgsProcessingParameterCrs(
                self.OUTPUT_CRS,
                self.tr("SRC do Dado de Saída (EPSG — opcional)"),
                optional=True,
                defaultValue=None,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_SKELETON,
                self.tr("Raster do Esqueleto (inspeção)"),
                optional=True,
                createByDefault=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_CENTRAL,
                self.tr("Rotas Centrais (Polylines)"),
                optional=True,
                createByDefault=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_NETWORK,
                self.tr("Rede Fluvial Completa (Linhas)"),
            )
        )

    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):

        # ── 0. FIXAR PROJ_DATA ──────────────────────────────────────────
        import os as _os
        try:
            from osgeo import gdal as _gdal_probe, osr as _osr_probe
            _gdal_path = _gdal_probe.__file__
            _osgeo4w_root = _gdal_path
            for _ in range(6):
                _osgeo4w_root = _os.path.dirname(_osgeo4w_root)
                _proj_db_candidate = _os.path.join(_osgeo4w_root, "share", "proj", "proj.db")
                if _os.path.isfile(_proj_db_candidate):
                    _proj_data_dir = _os.path.join(_osgeo4w_root, "share", "proj")
                    _os.environ["PROJ_DATA"] = _proj_data_dir
                    _os.environ["PROJ_LIB"]  = _proj_data_dir
                    try:
                        _osr_probe.SetPROJSearchPaths([_proj_data_dir])
                    except Exception:
                        pass
                    feedback.pushInfo(self.tr(f"PROJ_DATA fixado: {_proj_data_dir}"))
                    break
            else:
                feedback.pushWarning(self.tr(
                    "Nao foi possivel localizar o proj.db do OSGeo4W automaticamente."
                ))
        except Exception as _e_proj:
            feedback.pushWarning(self.tr(f"Aviso ao fixar PROJ_DATA: {_e_proj}"))

        # ── 1. Recuperar parâmetros ─────────────────────────────────────
        raster_layer  = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        vector_layer  = self.parameterAsVectorLayer(parameters, self.INPUT_VECTOR, context)
        gps_layer     = self.parameterAsVectorLayer(parameters, self.INPUT_GPS_TRACKS, context)
        limits_layer  = self.parameterAsVectorLayer(parameters, self.INPUT_LIMITS, context)

        buffer_dist_m       = self.parameterAsDouble(parameters, self.BUFFER_DIST, context)
        transect_interval_m = self.parameterAsDouble(parameters, self.TRANSECT_INTERVAL, context)
        output_crs_param    = self.parameterAsCrs(parameters, self.OUTPUT_CRS, context)

        raster_path_orig = raster_layer.source()
        vector_path_orig = vector_layer.source()
        tmp = context.temporaryFolder()

        # ── 2. Detectar CRS e reprojetar se necessário ──────────────────
        feedback.pushInfo(self.tr("A verificar SRC dos dados de entrada..."))

        qgs_raster_crs = raster_layer.crs()
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
            try:
                with rasterio.open(raster_path_orig) as src:
                    if src.crs is not None:
                        raster_crs_wkt = src.crs.to_wkt()
                        raster_is_geo  = _is_geographic(raster_crs_wkt)
                        feedback.pushInfo(self.tr(
                            f"SRC do raster lido via rasterio: "
                            f"{'Geográfico' if raster_is_geo else 'Projetado'}"
                        ))
            except Exception:
                pass
            if raster_crs_wkt is None:
                feedback.pushWarning(self.tr(
                    "AVISO: O raster não tem SRC definido nem no QGIS nem no ficheiro."
                ))

        qgs_vector_crs = vector_layer.crs()
        vector_epsg    = None
        vector_is_geo  = None

        if qgs_vector_crs.isValid():
            vector_crs_wkt = qgs_vector_crs.toWkt()
            vector_is_geo  = _is_geographic(vector_crs_wkt)
            try:
                auth_id = qgs_vector_crs.authid()
                if auth_id.upper().startswith("EPSG:"):
                    vector_epsg = int(auth_id.split(":")[1])
            except Exception:
                vector_epsg = None
            feedback.pushInfo(self.tr(
                f"SRC do vetor lido do QGIS: "
                f"{qgs_vector_crs.authid()} — "
                f"{'Geográfico (graus)' if vector_is_geo else 'Projetado (métrico)'}"
            ))
        else:
            try:
                gdf_crs_check = _read_vector(vector_path_orig)
                if gdf_crs_check.crs is not None:
                    vector_epsg   = gdf_crs_check.crs.to_epsg()
                    vector_is_geo = gdf_crs_check.crs.is_geographic
                    feedback.pushInfo(self.tr(f"SRC do vetor lido via geopandas: EPSG:{vector_epsg}"))
            except Exception:
                pass
            if vector_is_geo is None:
                feedback.pushWarning(self.tr("AVISO: O vetor não tem SRC definido."))

        if raster_crs_wkt is None and vector_is_geo is None:
            feedback.reportError(
                self.tr(
                    "ERRO: Nao foi possivel determinar o SRC do raster nem do vetor."
                ),
                fatalError=True,
            )
            return {}

        # Determinar EPSG de trabalho (métrico)
        work_epsg = None

        if raster_crs_wkt is not None and not raster_is_geo:
            work_epsg = None
            try:
                auth_id = qgs_raster_crs.authid()
                if auth_id.upper().startswith("EPSG:"):
                    work_epsg = int(auth_id.split(":")[1])
            except Exception:
                pass
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
            if work_epsg is None:
                try:
                    srid = qgs_raster_crs.postgisSrid()
                    if srid and srid > 0:
                        work_epsg = srid
                except Exception:
                    pass
            if work_epsg:
                feedback.pushInfo(self.tr(f"SRC de trabalho: EPSG:{work_epsg} (raster métrico)."))
            else:
                feedback.pushWarning(self.tr(
                    f"Nao foi possivel extrair EPSG numerico do raster ({qgs_raster_crs.authid()})."
                ))
            raster_path = raster_path_orig

        elif raster_crs_wkt is not None and raster_is_geo:
            lon, lat  = _get_raster_center_lonlat(raster_path_orig)
            work_epsg = _auto_utm_epsg(lon, lat)
            feedback.pushInfo(self.tr(
                f"SRC do raster é geográfico. Reprojetando para EPSG:{work_epsg} (UTM)..."
            ))
            raster_path = _reproject_raster_to_metric(raster_path_orig, work_epsg, tmp)

        elif vector_is_geo is not None and not vector_is_geo and vector_epsg:
            work_epsg = vector_epsg
            feedback.pushWarning(self.tr(
                f"Raster sem SRC. Usando SRC do vetor (EPSG:{work_epsg})."
            ))
            raster_path = raster_path_orig

        else:
            try:
                gdf_tmp = _read_vector(vector_path_orig)
                cx = gdf_tmp.geometry.unary_union.centroid.x
                cy = gdf_tmp.geometry.unary_union.centroid.y
                work_epsg = _auto_utm_epsg(cx, cy)
            except Exception:
                feedback.reportError(
                    self.tr("ERRO: Não foi possível calcular o UTM automático."),
                    fatalError=True,
                )
                return {}
            feedback.pushWarning(self.tr(
                f"Raster sem SRC e vetor geográfico. Usando EPSG:{work_epsg} (UTM auto)."
            ))
            raster_path = raster_path_orig

        # Reprojetar vetor se necessário
        needs_vector_reproject = False
        if vector_is_geo is True:
            needs_vector_reproject = True
        elif work_epsg and vector_epsg and vector_epsg != work_epsg:
            needs_vector_reproject = True

        if needs_vector_reproject:
            feedback.pushInfo(self.tr(
                f"SRC do vetor (EPSG:{vector_epsg}) difere do SRC de trabalho "
                f"(EPSG:{work_epsg}). A reprojetar vetor..."
            ))
            vector_path = _reproject_vector_to_metric(vector_path_orig, work_epsg, tmp)
        else:
            feedback.pushInfo(self.tr(
                f"SRC do vetor compativel com SRC de trabalho (EPSG:{work_epsg})."
            ))
            vector_path = vector_path_orig

        gdf_check = _read_vector(vector_path)

        feedback.pushInfo(self.tr(
            f"Parâmetros de distância — Buffer: {buffer_dist_m} m | "
            f"Transecto: {transect_interval_m} m"
        ))

        if work_epsg:
            work_crs_qgs = QgsCoordinateReferenceSystem(f"EPSG:{work_epsg}")
        elif qgs_raster_crs.isValid():
            work_crs_qgs = qgs_raster_crs
            try:
                srid = qgs_raster_crs.postgisSrid()
                if srid and srid > 0:
                    work_epsg = srid
                    work_crs_qgs = QgsCoordinateReferenceSystem(f"EPSG:{work_epsg}")
            except Exception:
                pass
        else:
            work_crs_qgs = context.project().crs()

        feedback.pushInfo(self.tr(f"SRC QGIS de trabalho definido: {work_crs_qgs.authid()}"))

        # ── A. ROTAS CENTRAIS ───────────────────────────────────────────
        # (NÃO ALTERADO — mantido exatamente como estava)
        # ---------------------------------------------------------------
        feedback.pushInfo(self.tr("A gerar rotas centrais — fase 1/3: a quantizar raster..."))

        temp_uint8_path = os.path.join(tmp, "temp_uint8.tif")
        temp_skel_path  = os.path.join(tmp, "temp_skeleton.tif")

        with rasterio.open(raster_path) as src:
            transform = src.transform
            crs       = src.crs
            height    = src.height
            width     = src.width

            bytes_per_row  = width
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
                    feedback.setProgress(int((i + 1) / n_blocks * 20))
                    if feedback.isCanceled():
                        return {}

        feedback.pushInfo(self.tr("Fase 2/3: a pre-processar e esqueletonizar..."))

        with rasterio.open(temp_uint8_path) as src:
            data_uint8   = src.read(1)
            pixel_size_m = abs(src.transform.a)
        feedback.setProgress(22)

        data_bool = data_uint8 > 0
        del data_uint8
        gc.collect()

        close_r       = max(1, int(round((buffer_dist_m * 0.10) / max(pixel_size_m, 1e-6))))
        open_r        = max(1, int(round((buffer_dist_m * 0.05) / max(pixel_size_m, 1e-6))))
        min_island_px = max(10, int(3.14159 * ((buffer_dist_m * 0.5) / max(pixel_size_m, 1e-6)) ** 2))

        feedback.pushInfo(self.tr(
            f"Pre-processamento morfologico: pixel={pixel_size_m:.2f}m | "
            f"close_r={close_r}px | open_r={open_r}px | min_island={min_island_px}px"
        ))

        feedback.pushInfo(self.tr("  Removendo objectos pequenos (ruido)..."))
        data_bool = remove_small_objects(data_bool, min_size=min_island_px, connectivity=2)
        gc.collect()
        feedback.setProgress(25)

        feedback.pushInfo(self.tr("  Fecho morfologico (preenchimento de gaps)..."))
        data_bool = binary_closing(data_bool, footprint=disk(close_r))
        gc.collect()
        feedback.setProgress(30)

        feedback.pushInfo(self.tr("  Abertura morfologica (suavizacao de bordas)..."))
        data_bool = binary_opening(data_bool, footprint=disk(open_r))
        gc.collect()
        feedback.setProgress(33)

        feedback.pushInfo(self.tr("  Esqueletonizando..."))
        skeleton_bool = skeletonize(data_bool)
        del data_bool
        gc.collect()
        feedback.setProgress(35)

        feedback.pushInfo(self.tr("  Afinamento (thin)..."))
        skeleton_bool = skimage_thin(skeleton_bool)
        gc.collect()
        feedback.setProgress(37)

        prune_px = max(3, int(round((buffer_dist_m * 0.5) / max(pixel_size_m, 1e-6))))
        feedback.pushInfo(self.tr(
            f"  Poda de ramos curtos: prune_px={prune_px}px ({prune_px * pixel_size_m:.1f}m)..."
        ))

        skel = skeleton_bool.copy()
        del skeleton_bool
        gc.collect()

        neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_kernel[1, 1] = 0

        for _iter in range(prune_px):
            if not skel.any():
                break
            neighbor_count = nd_convolve(skel.astype(np.uint8), neighbor_kernel, mode="constant", cval=0)
            terminals = skel & (neighbor_count == 1)
            if not terminals.any():
                break
            skel = skel & ~terminals

        feedback.setProgress(40)

        skeleton_uint8 = skel.astype(np.uint8)
        del skel
        gc.collect()

        feedback.pushInfo(self.tr("Fase 3/3: a gravar esqueleto em disco..."))

        profile_skel = profile_uint8.copy()
        with rasterio.open(temp_skel_path, "w", **profile_skel) as dst:
            for i, row_off in enumerate(range(0, height, rows_per_block)):
                actual_rows = min(rows_per_block, height - row_off)
                window = rasterio.windows.Window(0, row_off, width, actual_rows)
                dst.write(skeleton_uint8[row_off:row_off + actual_rows, :], 1, window=window)
                feedback.setProgress(40 + int((i + 1) / n_blocks * 10))
                if feedback.isCanceled():
                    return {}

        del skeleton_uint8
        gc.collect()

        try:
            os.remove(temp_uint8_path)
        except OSError:
            pass

        skel_dest = parameters.get(self.OUTPUT_SKELETON)
        if skel_dest:
            feedback.pushInfo(self.tr("A exportar raster do esqueleto para inspeção..."))
            import shutil
            try:
                skel_out_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_SKELETON, context)
                shutil.copy2(temp_skel_path, skel_out_path)
                feedback.pushInfo(self.tr(f"Raster do esqueleto gravado em: {skel_out_path}"))
            except Exception as e_skel:
                feedback.pushWarning(self.tr(f"Nao foi possivel exportar o raster do esqueleto: {e_skel}"))

        feedback.pushInfo(self.tr("Esqueleto gravado. A vectorizar via rastreio de grafo..."))

        from shapely.geometry import LineString as ShpLineString
        from osgeo            import ogr, osr
        from scipy.ndimage    import label as nd_label2

        feedback.pushInfo(self.tr("A carregar esqueleto para rastreio de grafo..."))

        with rasterio.open(temp_skel_path) as src:
            skel_arr       = src.read(1).astype(bool)
            skel_transform = src.transform
            pixel_size     = abs(skel_transform.a)

        feedback.setProgress(51)

        n_neighbors = nd_convolve(
            skel_arr.astype(np.uint8), neighbor_kernel, mode="constant", cval=0
        )

        terminals  = skel_arr & (n_neighbors == 1)
        internals  = skel_arr & (n_neighbors == 2)
        junctions  = skel_arr & (n_neighbors >= 3)
        skel_arr   = skel_arr & (n_neighbors >= 1)

        feedback.setProgress(53)

        def _rc_to_xy(r, c):
            x, y = rasterio.transform.xy(skel_transform, r, c, offset="center")
            return float(x), float(y)

        NEIGHBORS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        node_mask = terminals | junctions
        visited_edges = set()
        lines_rc = []
        node_rows, node_cols = np.where(node_mask)
        total_nodes = len(node_rows)

        def _all_neighbors(r, c):
            nb = []
            for dr, dc in NEIGHBORS_8:
                nr, nc_ = r+dr, c+dc
                if 0 <= nr < skel_arr.shape[0] and 0 <= nc_ < skel_arr.shape[1]:
                    if skel_arr[nr, nc_]:
                        nb.append((nr, nc_))
            return nb

        for ni, (nr0, nc0) in enumerate(zip(node_rows, node_cols)):
            for (vr, vc) in _all_neighbors(nr0, nc0):
                edge_key = (min((nr0,nc0),(vr,vc)), max((nr0,nc0),(vr,vc)))
                if edge_key in visited_edges:
                    continue
                chain = [(nr0, nc0), (vr, vc)]
                visited_edges.add(edge_key)
                cur_r, cur_c = vr, vc
                if node_mask[cur_r, cur_c]:
                    if len(chain) >= 2:
                        lines_rc.append(chain)
                    continue
                prev_r, prev_c = nr0, nc0
                while True:
                    nbs = [(r, c) for r, c in _all_neighbors(cur_r, cur_c)
                           if (r, c) != (prev_r, prev_c)]
                    if not nbs:
                        break
                    if len(nbs) == 1:
                        next_r, next_c = nbs[0]
                        e2 = (min((cur_r,cur_c),(next_r,next_c)),
                              max((cur_r,cur_c),(next_r,next_c)))
                        chain.append((next_r, next_c))
                        visited_edges.add(e2)
                        prev_r, prev_c = cur_r, cur_c
                        cur_r, cur_c = next_r, next_c
                        if node_mask[cur_r, cur_c]:
                            break
                    else:
                        break
                if len(chain) >= 2:
                    lines_rc.append(chain)

            if ni % 2000 == 0:
                feedback.setProgress(53 + int(ni / max(total_nodes, 1) * 5))
                if feedback.isCanceled():
                    return {}

        visited_flat = set()
        for chain in lines_rc:
            for px in chain:
                visited_flat.add(px)

        orphaned_mask = skel_arr.copy()
        for r, c in visited_flat:
            orphaned_mask[r, c] = False
        n_orphaned = int(orphaned_mask.sum())

        if n_orphaned > 0:
            feedback.pushInfo(self.tr(f"  Segunda passagem: {n_orphaned} pixels orphaned..."))
            struct_8 = np.ones((3, 3), dtype=bool)
            labeled_orphans, n_groups = nd_label(orphaned_mask, structure=struct_8)

            for grp in range(1, n_groups + 1):
                grp_mask = labeled_orphans == grp
                grp_rows, grp_cols = np.where(grp_mask)
                if len(grp_rows) < 2:
                    visited_flat.add((int(grp_rows[0]), int(grp_cols[0])))
                    continue
                sr2, sc2 = int(grp_rows[0]), int(grp_cols[0])
                best_start_n = 9
                for ri, ci in zip(grp_rows.tolist(), grp_cols.tolist()):
                    n_nb = sum(
                        1 for dr, dc in NEIGHBORS_8
                        if 0 <= ri+dr < grp_mask.shape[0]
                        and 0 <= ci+dc < grp_mask.shape[1]
                        and grp_mask[ri+dr, ci+dc]
                    )
                    if n_nb < best_start_n:
                        best_start_n = n_nb
                        sr2, sc2 = int(ri), int(ci)
                    if best_start_n == 1:
                        break
                chain2  = []
                v2      = np.zeros(skel_arr.shape, dtype=bool)
                cur_r2, cur_c2 = sr2, sc2
                while True:
                    v2[cur_r2, cur_c2] = True
                    visited_flat.add((cur_r2, cur_c2))
                    chain2.append((cur_r2, cur_c2))
                    next_r2, next_c2 = -1, -1
                    for dr, dc in NEIGHBORS_8:
                        nr2, nc2 = cur_r2+dr, cur_c2+dc
                        if 0 <= nr2 < skel_arr.shape[0] and 0 <= nc2 < skel_arr.shape[1]:
                            if grp_mask[nr2, nc2] and not v2[nr2, nc2]:
                                next_r2, next_c2 = nr2, nc2
                                break
                    if next_r2 == -1:
                        break
                    cur_r2, cur_c2 = next_r2, next_c2
                if len(chain2) >= 2:
                    lines_rc.append(chain2)

        feedback.pushInfo(self.tr(
            f"Rastreio concluido: {len(lines_rc)} cadeias | "
            f"{max(0, n_orphaned - len(visited_flat))} pixels ainda nao cobertos."
        ))

        del skel_arr, terminals, internals, junctions, n_neighbors, visited_flat, orphaned_mask
        gc.collect()
        feedback.setProgress(58)

        from shapely.ops import linemerge, unary_union as shp_unary_union
        from shapely.geometry import MultiLineString as ShpMultiLine

        simplify_tol  = pixel_size * 1.0
        shapely_lines = []

        feedback.pushInfo(self.tr("  Convertendo cadeias para LineStrings..."))
        smooth_window = max(2, int(round((buffer_dist_m * 0.25) / max(pixel_size_m, 1e-6))))
        feedback.pushInfo(self.tr(
            f"  Smooth de media movel: janela={smooth_window * 2 + 1} pontos "
            f"({(smooth_window * 2 + 1) * pixel_size_m:.1f}m)"
        ))

        def _moving_average_smooth(coords, window, pt_start, pt_end):
            n = len(coords)
            if n < 3:
                return coords
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            sx, sy = [], []
            for i in range(n):
                lo = max(0, i - window)
                hi = min(n - 1, i + window)
                count = hi - lo + 1
                sx.append(sum(xs[lo:hi+1]) / count)
                sy.append(sum(ys[lo:hi+1]) / count)
            sx[0],  sy[0]  = pt_start[0], pt_start[1]
            sx[-1], sy[-1] = pt_end[0],   pt_end[1]
            return list(zip(sx, sy))

        for chain in lines_rc:
            if len(chain) < 2:
                continue
            coords = [_rc_to_xy(r, c) for r, c in chain]
            if len(coords) < 2:
                continue
            pt_start = coords[0]
            pt_end   = coords[-1]
            smoothed = _moving_average_smooth(coords, smooth_window, pt_start, pt_end)
            if len(smoothed) < 2:
                continue
            line = ShpLineString(smoothed)
            if line.is_empty or line.length == 0:
                continue
            shapely_lines.append(line)

        del lines_rc
        gc.collect()
        feedback.pushInfo(self.tr(f"  {len(shapely_lines)} segmentos antes do merge."))

        feedback.pushInfo(self.tr("  Unindo segmentos contiguos (linemerge)..."))
        merged = linemerge(shapely_lines)
        del shapely_lines
        gc.collect()

        if merged.geom_type == "LineString":
            merged_lines = [merged]
        elif merged.geom_type == "MultiLineString":
            merged_lines = list(merged.geoms)
        else:
            merged_lines = [g for g in merged.geoms if g.geom_type == "LineString"]

        feedback.pushInfo(self.tr(f"  {len(merged_lines)} linhas apos linemerge."))

        from scipy.spatial import cKDTree as _cKDTree

        SNAP_TOL = pixel_size * 2.0

        all_endpoints = []
        ep_map = []
        for li, ml in enumerate(merged_lines):
            coords_ml = list(ml.coords)
            all_endpoints.append(coords_ml[0])
            ep_map.append((li, 0))
            all_endpoints.append(coords_ml[-1])
            ep_map.append((li, -1))

        if len(all_endpoints) >= 2:
            ep_arr  = np.array(all_endpoints)
            tree    = _cKDTree(ep_arr)
            pairs   = tree.query_pairs(SNAP_TOL)

            parent = list(range(len(all_endpoints)))
            def _find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def _union(a, b):
                a, b = _find(a), _find(b)
                if a != b: parent[b] = a

            for i, j in pairs:
                _union(i, j)

            from collections import defaultdict
            groups = defaultdict(list)
            for idx in range(len(all_endpoints)):
                groups[_find(idx)].append(idx)

            snap_to = {}
            for root, members in groups.items():
                if len(members) < 2:
                    continue
                xs = [all_endpoints[m][0] for m in members]
                ys = [all_endpoints[m][1] for m in members]
                cx_mean = sum(xs) / len(xs)
                cy_mean = sum(ys) / len(ys)
                for m in members:
                    snap_to[m] = (cx_mean, cy_mean)

            snapped_lines = []
            for li, ml in enumerate(merged_lines):
                coords_ml = list(ml.coords)
                start_idx = li * 2
                end_idx   = li * 2 + 1
                if start_idx in snap_to:
                    coords_ml[0]  = snap_to[start_idx]
                if end_idx in snap_to:
                    coords_ml[-1] = snap_to[end_idx]
                if len(coords_ml) >= 2:
                    snapped_lines.append(ShpLineString(coords_ml))

            del tree, ep_arr, groups, snap_to
            gc.collect()
        else:
            snapped_lines = merged_lines

        merged2 = linemerge(snapped_lines)
        del snapped_lines
        gc.collect()

        if merged2.geom_type == "LineString":
            merged_lines = [merged2]
        elif merged2.geom_type == "MultiLineString":
            merged_lines = list(merged2.geoms)
        else:
            merged_lines = [g for g in getattr(merged2, "geoms", [])
                           if g.geom_type == "LineString"]

        feedback.pushInfo(self.tr(f"  {len(merged_lines)} linhas apos snap topologico."))

        DEDUP_DIST = pixel_size * 3.0
        N_SAMPLES  = 10

        def _avg_dist(la, lb):
            total = 0.0
            for k in range(N_SAMPLES):
                t = k / max(N_SAMPLES - 1, 1)
                pa = la.interpolate(t, normalized=True)
                total += lb.distance(pa)
            return total / N_SAMPLES

        if len(merged_lines) > 1:
            feedback.pushInfo(self.tr(f"  Deduplicando {len(merged_lines)} linhas..."))
            keep = [True] * len(merged_lines)
            lengths = [l.length for l in merged_lines]
            for i in range(len(merged_lines)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(merged_lines)):
                    if not keep[j]:
                        continue
                    bi = merged_lines[i].bounds
                    bj = merged_lines[j].bounds
                    bb_dist = max(
                        max(bi[0], bj[0]) - min(bi[2], bj[2]),
                        max(bi[1], bj[1]) - min(bi[3], bj[3]),
                        0
                    )
                    if bb_dist > DEDUP_DIST * 2:
                        continue
                    if _avg_dist(merged_lines[i], merged_lines[j]) < DEDUP_DIST:
                        if lengths[i] >= lengths[j]:
                            keep[j] = False
                        else:
                            keep[i] = False
                            break
            merged_lines = [l for l, k in zip(merged_lines, keep) if k]
            feedback.pushInfo(self.tr(f"  {len(merged_lines)} linhas apos deduplicacao."))

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
        gc.collect()
        feedback.pushInfo(self.tr(f"Vectorizacao concluida: {feat_count} polylines gravadas."))
        feedback.setProgress(58)

        # União de extremidades próximas dentro do polígono de água
        feedback.pushInfo(self.tr("A unir extremidades proximas dentro da mascara de agua..."))

        from shapely.geometry import LineString as ShpLineString, Point as ShpPoint
        from shapely.ops      import linemerge as shp_linemerge

        water_gdf_join   = _read_vector(vector_path)[["geometry"]]
        water_union_join = water_gdf_join.geometry.unary_union
        del water_gdf_join
        gc.collect()

        JOIN_DIST = pixel_size * 3.0

        from osgeo import ogr as _ogr2
        ds_read  = _ogr2.Open(temp_central_path)
        lyr_read = ds_read.GetLayer(0)
        central_shapely = []
        for feat_r in lyr_read:
            wkt = feat_r.GetGeometryRef().ExportToWkt()
            try:
                from shapely import wkt as shp_wkt
                geom = shp_wkt.loads(wkt)
                if geom and not geom.is_empty:
                    central_shapely.append(geom)
            except Exception:
                pass
        ds_read = None
        gc.collect()

        endpoints = []
        for li, line in enumerate(central_shapely):
            coords = list(line.coords)
            if len(coords) >= 2:
                endpoints.append((ShpPoint(coords[0]),  li, "start"))
                endpoints.append((ShpPoint(coords[-1]), li, "end"))

        connector_lines = []
        used_pairs = set()
        for i, (pa, la, _) in enumerate(endpoints):
            for j, (pb, lb, _) in enumerate(endpoints):
                if la == lb:
                    continue
                pair_key = (min(la, lb), max(la, lb))
                if pair_key in used_pairs:
                    continue
                dist = pa.distance(pb)
                if dist < 1e-6 or dist > JOIN_DIST:
                    continue
                connector = ShpLineString([pa.coords[0], pb.coords[0]])
                if water_union_join.contains(connector):
                    connector_lines.append(connector)
                    used_pairs.add(pair_key)

        if connector_lines:
            feedback.pushInfo(self.tr(f"  {len(connector_lines)} ligacoes criadas."))
            ds_conn  = _ogr2.Open(temp_central_path, 1)
            lyr_conn = ds_conn.GetLayer(0)
            lyr_conn.StartTransaction()
            lyr_defn_conn = lyr_conn.GetLayerDefn()
            for conn_line in connector_lines:
                og = _ogr2.CreateGeometryFromWkt(conn_line.wkt)
                if og is None:
                    continue
                f = _ogr2.Feature(lyr_defn_conn)
                f.SetGeometry(og)
                f.SetField("source", "central_connector")
                lyr_conn.CreateFeature(f)
                f = None
            lyr_conn.CommitTransaction()
            ds_conn = None
        else:
            feedback.pushInfo(self.tr("  Nenhuma extremidade proxima encontrada."))

        del water_union_join, endpoints, connector_lines
        gc.collect()
        feedback.setProgress(60)

        # Entregar OUTPUT_CENTRAL (opcional)
        central_dest = parameters.get(self.OUTPUT_CENTRAL)
        if central_dest:
            feedback.pushInfo(self.tr("A exportar rotas centrais (polylines)..."))
            from qgis.core import QgsFields, QgsField, QgsWkbTypes
            central_fields = QgsFields()
            central_fields.append(QgsField("source", QVariant.String))
            (central_sink, central_dest_id) = self.parameterAsSink(
                parameters, self.OUTPUT_CENTRAL, context,
                central_fields, QgsWkbTypes.LineString, work_crs_qgs
            )
            if central_sink is not None:
                ds_out  = _ogr2.Open(temp_central_path)
                lyr_out = ds_out.GetLayer(0)
                for feat_out in lyr_out:
                    wkt_out = feat_out.GetGeometryRef().ExportToWkt()
                    try:
                        from shapely import wkt as shp_wkt2
                        g = shp_wkt2.loads(wkt_out)
                        parts_out = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
                        for part_out in parts_out:
                            if part_out.is_empty or part_out.length == 0:
                                continue
                            qfeat = QgsFeature(central_fields)
                            from qgis.core import QgsGeometry
                            qfeat.setGeometry(QgsGeometry.fromWkt(part_out.wkt))
                            qfeat.setAttribute("source", feat_out.GetField("source") or "central")
                            central_sink.addFeature(qfeat, QgsFeatureSink.FastInsert)
                    except Exception:
                        pass
                ds_out = None
                del central_sink

        central_routes = temp_central_path
        feedback.setProgress(62)

        # ── B. ROTAS MARGINAIS ──────────────────────────────────────────
        # (NÃO ALTERADO — mantido exatamente como estava)
        # ---------------------------------------------------------------
        feedback.pushInfo(self.tr("A gerar rotas marginais..."))

        nav_mediana = buffer_dist_m
        if gps_layer:
            feedback.pushInfo(self.tr(
                "Camada GPS encontrada. Usando buffer como fallback."
            ))

        water_gdf = _read_vector(vector_path)[["geometry"]].copy()
        water_gdf["geometry"] = (
            water_gdf.geometry
            .buffer(-abs(nav_mediana))
            .boundary
        )
        water_gdf = water_gdf[~water_gdf.geometry.is_empty].reset_index(drop=True)

        temp_marg_path = os.path.join(tmp, "temp_marginais.gpkg")
        water_gdf.to_file(temp_marg_path, driver="GPKG")

        del water_gdf
        gc.collect()
        feedback.setProgress(65)

        # ── C. ROTAS TRANSVERSAIS ───────────────────────────────────────
        #
        # NOVO PIPELINE INTERNO:
        #
        # As rotas transversais são posicionadas sobre centerlines geradas
        # INTERNAMENTE a partir do vetor de água, usando um pipeline de
        # buffer positivo → buffer negativo → rasterize → skeletonize.
        # Este pipeline é completamente temporário — os seus produtos
        # intermediários não são entregues como output.
        #
        # O clip final de cada transversal pelo polígono de água original
        # permanece igual ao comportamento anterior.
        # ---------------------------------------------------------------
        feedback.pushInfo(self.tr(
            "A gerar centerlines INTERNAS para posicionamento das transversais "
            "(buffer+ → buffer- → rasterize → skeletonize)..."
        ))

        internal_centerlines = _compute_internal_centerlines(
            vector_path      = vector_path,
            reference_raster_path = raster_path,
            buffer_dist_m    = buffer_dist_m,
            pixel_size_m     = pixel_size_m,
            tmp_folder       = tmp,
            feedback         = feedback,
        )

        if not internal_centerlines:
            feedback.pushWarning(self.tr(
                "AVISO: Nenhuma centerline interna gerada para as transversais. "
                "A usar rotas centrais do esqueleto como fallback."
            ))
            # Fallback: usar as rotas centrais geradas na etapa A
            from shapely import wkt as shp_wkt_fb
            internal_centerlines = []
            ds_fb  = ogr.Open(central_routes)
            lyr_fb = ds_fb.GetLayer(0)
            for feat_fb in lyr_fb:
                geom_fb = feat_fb.GetGeometryRef()
                if geom_fb is None:
                    continue
                try:
                    g_fb = shp_wkt_fb.loads(geom_fb.ExportToWkt())
                    if g_fb and not g_fb.is_empty:
                        if g_fb.geom_type == "MultiLineString":
                            internal_centerlines.extend(list(g_fb.geoms))
                        elif g_fb.geom_type == "LineString":
                            internal_centerlines.append(g_fb)
                except Exception:
                    pass
            ds_fb = None

        # Linemerge das centerlines internas
        from shapely.ops import linemerge as shp_linemerge2
        merged_internal = shp_linemerge2(internal_centerlines)
        del internal_centerlines
        gc.collect()

        if merged_internal.geom_type == "LineString":
            central_line_list = [merged_internal]
        elif merged_internal.geom_type == "MultiLineString":
            central_line_list = list(merged_internal.geoms)
        else:
            central_line_list = [g for g in getattr(merged_internal, "geoms", [])
                                 if g.geom_type == "LineString"]

        feedback.pushInfo(self.tr(
            f"  {len(central_line_list)} centerlines internas para transversais."
        ))

        # Carregar polígono de água para clip das transversais
        water_mask_gdf = _read_vector(vector_path)[["geometry"]]
        water_union    = water_mask_gdf.geometry.unary_union
        del water_mask_gdf
        gc.collect()

        # Criar datasource OGR para as transversais
        temp_transect_path = os.path.join(tmp, "temp_transects.gpkg")
        drv_t   = ogr.GetDriverByName("GPKG")
        ds_t    = drv_t.CreateDataSource(temp_transect_path)
        srs_t   = osr.SpatialReference()
        srs_t.ImportFromWkt(work_crs_qgs.toWkt())
        lyr_t   = ds_t.CreateLayer("transects", srs=srs_t, geom_type=ogr.wkbLineString)
        fdef_t  = ogr.FieldDefn("source", ogr.OFTString); fdef_t.SetWidth(32)
        lyr_t.CreateField(fdef_t)
        lyr_defn_t = lyr_t.GetLayerDefn()
        lyr_t.StartTransaction()
        t_count = 0

        def _make_perpendicular_v2(line_geom, dist_along, water_poly):
            total_len = line_geom.length
            delta = max(1.0, total_len * 0.0005)
            p0 = line_geom.interpolate(max(0.0, dist_along - delta))
            p1 = line_geom.interpolate(min(total_len, dist_along + delta))
            dx, dy = p1.x - p0.x, p1.y - p0.y
            norm = (dx**2 + dy**2) ** 0.5
            if norm < 1e-10:
                return None

            cp = line_geom.interpolate(dist_along)
            cx, cy = cp.x, cp.y

            from shapely.geometry import Point as ShpPoint
            center_pt = ShpPoint(cx, cy)
            boundary  = water_poly.boundary
            dist_to_boundary = center_pt.distance(boundary)
            half_len = max(50.0, dist_to_boundary * 1.5)

            px, py = -dy / norm, dx / norm

            return ShpLineString([
                (cx - px * half_len, cy - py * half_len),
                (cx + px * half_len, cy + py * half_len),
            ])

        total_lines_t = len(central_line_list)
        for gi, line in enumerate(central_line_list):
            if line is None or line.is_empty:
                continue
            length = line.length
            if length < transect_interval_m:
                continue

            dist = transect_interval_m / 2.0
            while dist < length:
                perp = _make_perpendicular_v2(line, dist, water_union)
                if perp is not None:
                    clipped = perp.intersection(water_union)
                    if not clipped.is_empty and clipped.length > 1.0:
                        candidates = []
                        if clipped.geom_type == "LineString":
                            candidates = [clipped]
                        elif clipped.geom_type in ("MultiLineString", "GeometryCollection"):
                            candidates = [g for g in clipped.geoms
                                         if hasattr(g, "coords") and g.length > 1.0]
                        elif hasattr(clipped, "coords"):
                            try:
                                candidates = [ShpLineString(list(clipped.coords))]
                            except Exception:
                                pass

                        for seg in candidates:
                            try:
                                ogr_g = ogr.CreateGeometryFromWkt(seg.wkt)
                            except Exception:
                                ogr_g = None
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

            feedback.setProgress(65 + int((gi + 1) / max(total_lines_t, 1) * 17))
            if feedback.isCanceled():
                lyr_t.CommitTransaction()
                ds_t = None
                return {}

        lyr_t.CommitTransaction()
        ds_t = None
        del central_line_list, water_union
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
        feedback.setProgress(88)

        # ── E. SPLIT LINES AT INTERSECTIONS ────────────────────────────
        #
        # Divide todas as linhas da rede nos seus pontos de interseção,
        # garantindo topologia correcta para análise de redes (routing,
        # connectivity). Executado APÓS o merge e fixgeometries, ANTES
        # da reprojeção final.
        #
        # Usa o algoritmo nativo do QGIS Processing "splitwithlines"
        # que divide as features de INPUT nos pontos onde se intersectam
        # com as features de LINES. Ao passar a própria rede como ambos
        # os argumentos, todas as interseções internas são resolvidas.
        # ---------------------------------------------------------------
        feedback.pushInfo(self.tr(
            "A dividir linhas nos pontos de interseção (split at intersections)..."
        ))

        split_network = processing.run(
            "native:splitwithlines",
            {
                "INPUT":  merged_network,
                "LINES":  merged_network,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context=context,
            feedback=feedback,
        )["OUTPUT"]

        del merged_network
        gc.collect()
        feedback.setProgress(92)

        feedback.pushInfo(self.tr("A reprojetar a rede final..."))

        # ── F. REPROJETAR PARA O SRC DE SAÍDA ───────────────────────────
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
                "INPUT":      split_network,
                "TARGET_CRS": target_crs,
                "OUTPUT":     "TEMPORARY_OUTPUT",
            },
            context=context,
            feedback=feedback,
        )["OUTPUT"]

        del split_network
        gc.collect()
        feedback.setProgress(95)

        # ── G. GRAVAR NA SAÍDA FINAL VIA FEATURESINK ───────────────────
        feedback.pushInfo(self.tr("A gravar a rede final..."))

        if isinstance(reprojected, str):
            lyr_final = QgsVectorLayer(reprojected, "rede_final", "ogr")
        else:
            lyr_final = reprojected

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT_NETWORK, context,
            lyr_final.fields(), lyr_final.wkbType(), lyr_final.crs()
        )

        if sink is None:
            feedback.reportError(
                self.tr("Nao foi possivel criar a camada de saida."), fatalError=True
            )
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
