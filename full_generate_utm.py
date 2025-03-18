import argparse
import copy
import math
import sys
import time
import tracemalloc
import logging
from tqdm import tqdm

import geojson
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from pyproj import Transformer
from scipy.spatial import Delaunay
from shapely.geometry import MultiPolygon, Point, Polygon, shape

import srtm
from stl import mesh

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

SCALE = 0.0003
WATER_SETBACK = 10 * 1000  # 10 км от воды
WGS84 = 'EPSG:4326'
UTM = 'EPSG:32646'
REDUCE = 0.10  # mm

elevation_data = srtm.get_data(local_cache_dir="data/tmp_cache")

class Timer:
    def __init__(self, name: str = None):
        self.name = name or "Operation"
        self._start = None
        self._lap = None

    def __enter__(self):
        self._start = time.perf_counter()
        self._lap = self._start
        # logger.info(f"Starting {self.name}...")
        return self

    def lap(self, message: str):
        current = time.perf_counter()
        elapsed = current - self._lap
        self._lap = current
        # Форматирование с фиксированной шириной
        logger.info(f"{message + ':':<50} {elapsed:>7.2f}s")

    def __exit__(self, *args):
        total = time.perf_counter() - self._start
        # Выравнивание итогового времени
        logger.info(f"{self.name + ' - Total':<50} {total:>7.2f}s")


def func(x_values):
    return np.log2(x_values * 0.005 + 1)

def scale_elevation(elevations):
    adjusted_elevations = [(x * SCALE, y * SCALE, func(h) / func(6000) * 25)  for (x, y, h) in elevations]
    # adjusted_elevations = [(x * SCALE, y * SCALE, h / 5600 * 20)  for (x, y, h) in elevations]
    return np.array(adjusted_elevations)

def frange(start, stop, step):
    while start < stop:
        yield start
        start += step

def GetHeight(points, step, water):
    find = [
        (0, -step),
        (0, step),
        (-step, 0),
        (step, 0),
        (-step, -step),
        (-step, step),
        (step, -step),
        (step, step),
    ]

    transformer = Transformer.from_crs(UTM, WGS84, always_xy=True)
    points_utm = points
    points_wgs84 = [transformer.transform(lon, lat) for (lon, lat) in points]

    new_points = []
    for i, (lon_utm, lat_utm) in enumerate(points_utm):
        point = Point(lon_utm, lat_utm)
        coef = 1
        if len(water) > 0:
            mins_dist = [point.distance(area) for area in water]
            min_dist = min(mins_dist)[0]
            coef = min(min_dist, WATER_SETBACK) / WATER_SETBACK

        lon_wgs84 = points_wgs84[i][0]
        lat_wgs84 = points_wgs84[i][1]
        if lat_wgs84 < 60:
            h = elevation_data.get_elevation(lat_wgs84, lon_wgs84)

            index = 0
            multy = 1
            while h is None:
                lon_utm = points_utm[i][0] + multy * find[index][0]
                lat_utm = points_utm[i][1] + multy * find[index][1]
                lon_tmp, lat_tmp = transformer.transform(lon_utm, lat_utm)
                h = elevation_data.get_elevation(lat_tmp, lon_tmp)
                index += 1
                if len(find) == index:
                    index = 0
                    multy += 1

            if h < 0:
                h = 0
            new_points.append((points_utm[i][0], points_utm[i][1], h * coef))
        else:
            new_points.append((points_utm[i][0], points_utm[i][1], 0.01 * coef))

    return new_points

def multy_to_polygon(geometry, src):
    sum_poligons = []
    for multi in geometry:
        if isinstance(multi, MultiPolygon):
            for polygon in multi.geoms:
                sum_poligons.append(polygon)
        else:
            sum_poligons.append(multi)

    geo = gpd.GeoDataFrame({'geometry': sum_poligons})
    geo.set_crs(src, inplace=True)
    geo = geo.to_crs(UTM)
    return geo

def bypass_polygon(geo, step):
    contour = geo.exterior
    length = contour.length
    points_on_contour = []

    distance = 0
    while distance < length:
        point = contour.interpolate(distance)
        points_on_contour.append(Point(point.x, point.y))
        distance += step

    return points_on_contour

def make_contour(geo_simply, step_m):
    with Timer("Contour creation"):
        contour_raw = [bypass_polygon(poly, step_m) for poly in geo_simply.geometry]
        contour = []
        for poly in tqdm(contour_raw, desc="Processing polygons"):
            if len(poly) > 50:
                p = Polygon(poly)
                minx, miny, maxx, maxy = p.bounds
                full_point_x = int((maxx - minx) / step_m)
                if full_point_x > 0:
                    contour.append(p)
    return contour

def make_mesh(contour, step_m):
    mesh = [[] for _ in range(len(contour))]
    with Timer("Mesh generation"):
        for i, polygon in enumerate(tqdm(contour, desc="Generating mesh")):
            minx, miny, maxx, maxy = polygon.bounds
            for x in frange(minx, maxx, step_m):
                for y in frange(miny, maxy, step_m):
                    point = Point(x, y)
                    mesh[i].append(point)
    return mesh

def filter_mesh(contour, mesh):
    mesh_inside = []
    with Timer("Mesh filtering"):
        for i, polygon in enumerate(tqdm(contour, desc="Filtering mesh")):
            gdf_points = gpd.GeoDataFrame({'geometry': mesh[i]})
            mesh_inside.append(gdf_points[gdf_points.within(polygon)])
    return mesh_inside

def area_stl(contour, my_mesh):
    with Timer("Area STL generation"):
        tri_area = Delaunay(my_mesh[:, :2])
        polygon = Polygon(contour)
        triangle = [Polygon(my_mesh[simplex]) for simplex in tri_area.simplices]

        gdf_points = gpd.GeoDataFrame({'geometry': triangle})
        filtered_triangles_area = gdf_points[gdf_points.within(polygon)]

        area_surface_mesh = mesh.Mesh(np.zeros(len(filtered_triangles_area), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(filtered_triangles_area.geometry):
            area_surface_mesh.vectors[i] = np.array(f.exterior.coords[:3])
    return area_surface_mesh

def wall_stl(contour, contour_zero):
    with Timer("Wall STL generation"):
        assert len(contour_zero) == len(contour)
        len_wall = len(contour_zero)

        wall_triangles = []
        for i in range(0, len_wall - 1):
            wall_triangles.append([i + 1, len_wall + i, len_wall + i + 1])
            wall_triangles.append([i, i + 1, len_wall + i])
        wall_triangles.append([0, 2 * len_wall - 1, len_wall])
        wall_triangles.append([len_wall - 1, 0, 2 * len_wall - 1])

        wall_points = np.concatenate((contour_zero, contour))

        wall_surface_mesh = mesh.Mesh(np.zeros(len(wall_triangles), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(wall_triangles):
            for j in range(3):
                wall_surface_mesh.vectors[i][j] = wall_points[f[j], :]
    return wall_surface_mesh

def bottom_stl(contour_zero):
    with Timer("Bottom STL generation"):
        tri_bottom = Delaunay(contour_zero[:, :2])
        polygon = Polygon(contour_zero).buffer(0)
        filtered_triangles_bottom = []
        for simplex in tri_bottom.simplices:
            triangle = Polygon(contour_zero[simplex])
            if polygon.contains(triangle):
                filtered_triangles_bottom.append(simplex)

        bottom_surface_mesh = mesh.Mesh(np.zeros(len(filtered_triangles_bottom), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(filtered_triangles_bottom):
            for j in range(3):
                bottom_surface_mesh.vectors[i][j] = contour_zero[f[j], :]
    return bottom_surface_mesh

def combined_stl(area, wall, bottom):
    with Timer("STL combination"):
        combined_vectors = np.concatenate([area.vectors, wall.vectors, bottom.vectors])
        combined_mesh = mesh.Mesh(np.zeros(combined_vectors.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(combined_vectors):
            combined_mesh.vectors[i] = f
        combined_mesh.update_normals()
    return combined_mesh

def make_stl_obl(utm_contour, utm_mesh, utm_contour_zero):
    list_stl = []
    count_stl = len(utm_contour)
    with Timer("Full STL generation"):
        for i in range(count_stl):
            with Timer(f"STL part {i+1}"):
                contour_scaled = scale_elevation(utm_contour[i])
                area_scaled = scale_elevation(utm_mesh[i])
                contour_zero_scaled = scale_elevation(utm_contour_zero[i])

                logger.info(f"Contour: {len(utm_contour[i])} points | "
                            f"Area: {len(utm_mesh[i])} points | "
                            f"Zero: {len(utm_contour_zero[i])} points")

                min_h = min(area_scaled, key=lambda x: x[2])[2]
                max_h = max(area_scaled, key=lambda x: x[2])[2]
                logger.info(f"Height range: {min_h:.2f}mm - {max_h:.2f}mm | "
                            f"Difference: {max_h - min_h:.2f}mm")

                area = area_stl(contour_scaled, area_scaled)
                wall = wall_stl(contour_scaled, contour_zero_scaled)
                bottom = bottom_stl(contour_zero_scaled)
                list_stl.append(combined_stl(area, wall, bottom))
    return list_stl

def generate(obl_name: str, path_save: str, step_m: int, oblast, water) -> None:
    total_timer = Timer(f"Processing {obl_name}")
    with total_timer:
        # fig, ax = plt.subplots(figsize=(5, 5))
        logger.info(f"{'Starting parameters':<50} Step: {step_m}m | Scale: {step_m * SCALE:.2f}mm")
        
        with Timer("Geometry preparation"):
            geo_simply = multy_to_polygon(oblast.geometry, WGS84)
            geo_simply = multy_to_polygon(geo_simply.buffer(-(REDUCE / SCALE)), UTM)
            logger.info(f"{'Reduction parameters':<50} {REDUCE:.2f}mm | {REDUCE / SCALE:.2f}m")

        contour = make_contour(geo_simply, step_m)
        area_mesh = filter_mesh(contour, make_mesh(contour, step_m))

        with Timer("Height processing"):
            min_h = 6000
            max_h = 0
            for i, polygon in enumerate(tqdm(contour, desc="Contour height")):
                contour[i] = GetHeight(polygon.exterior.coords, step_m, water)
                min_h = min(min_h, min(contour[i], key=lambda x: x[2])[2])
                max_h = max(max_h, max(contour[i], key=lambda x: x[2])[2])

            for i, points in enumerate(tqdm(area_mesh, desc="Area height")):
                area_mesh[i] = GetHeight([(point.x, point.y) for point in points.geometry], step_m, water)
                min_h = min(min_h, min(area_mesh[i], key=lambda x: x[2])[2])
                max_h = max(max_h, max(area_mesh[i], key=lambda x: x[2])[2])

            logger.info(f"{'Height range':<50} {min_h:.2f}m - {max_h:.2f}m")

        with Timer("Coordinate conversion"):
            utm_contour = [[(p[0], p[1], p[2]) for p in points] for points in contour]
            utm_mesh = [copy.deepcopy(utm_contour[i]) + [(p[0], p[1], p[2]) for p in points] 
                       for i, points in enumerate(area_mesh)]
            utm_contour_zero = [[(p[0], p[1], 0) for p in points] for points in utm_contour]

        list_stl = make_stl_obl(utm_contour, utm_mesh, utm_contour_zero)
        
        with Timer("Saving STL files"):
            for i, obl in enumerate(list_stl):
                filename = f'{path_save}{obl_name}_{step_m}_{i}.stl'
                obl.save(filename)
                logger.info(f"Saved {filename}")

def main():
    tracemalloc.start()
    path = "stl/"
    water = []
    paths_water = {}

    parser = argparse.ArgumentParser(
        description="Генератор геоданных",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-C', '--circle', 
                        action='store_true',
                        help="Режим генерации для круговой области")
    parser.add_argument('-s', '--step', 
                        type=int,
                        required=True,
                        help="Шаг генерации (метры)")
    parser.add_argument('-n', '--name', 
                        type=str,
                        help="Название области для генерации")
    
    args = parser.parse_args()

    with Timer("Water data loading"):
        for path_water in paths_water:
            with open(path_water, 'r') as f:
                sea_data = geojson.load(f)
            water.append([shape(feature['geometry']) for feature in sea_data['features']])

    if args.circle:
        SCALE = 0.001
        with Timer("Circle data loading"):
            gpkd = gpd.read_file("circle.geojson")
        generate("circle", path, args.step, gpkd, water)
    else:
        with Timer("Region data loading"):
            gpkd = gpd.read_file("data/russia_regions.geojson")
        
        if args.name:
            oblast = gpkd[gpkd["region"] == args.name]
            if oblast.empty:
                available = '\n'.join(gpkd["region"].unique())
                logger.error(f"Область '{args.name}' не найдена. Доступные области:\n{available}")
                exit(1)
            generate(args.name, path, args.step, oblast, water)
        else:
            for region in gpkd["region"]:
                oblast = gpkd[gpkd["region"] == region]
                generate(region, path, args.step, oblast, water)

    size, peak = tracemalloc.get_traced_memory()
    logger.info(f"{'Memory usage':<50} {size/1024:>7.1f} KB")
    logger.info(f"{'Peak memory usage':<50} {peak/1024:>7.1f} KB")

if __name__ == "__main__":
    main()
