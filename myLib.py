"""
Общая библиотека для генерации STL моделей из геоданных.
Содержит общие функции для работы с высотами, геометрией и STL.
"""

import copy
import logging
import math
import time

import geopandas as gpd
import numpy as np
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from pyproj import Transformer
from scipy.spatial import Delaunay
from shapely.geometry import MultiPolygon, Point, Polygon, shape

import srtm
import copernicus
from stl import mesh

# Константы
SCALE = 0.0003
WATER_SETBACK = 10 * 1000  # 10 км от воды
WGS84 = 'EPSG:4326'
UTM = 'EPSG:32646'  # UTM зона 46 (для Кавказа)
REDUCE = 0.10  # mm

# Выбор источника данных высот: 'srtm' или 'copernicus'
ELEVATION_SOURCE = 'copernicus'

if ELEVATION_SOURCE == 'copernicus':
    elevation_data = copernicus.get_data(local_cache_dir="data/tmp_cache_copernicus")
else:
    elevation_data = srtm.get_data(local_cache_dir="data/tmp_cache")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class Timer:
    """Класс для измерения времени выполнения операций"""
    def __init__(self, name: str = None):
        self.name = name or "Operation"
        self._start = None
        self._lap = None

    def __enter__(self):
        self._start = time.perf_counter()
        self._lap = self._start
        return self

    def __exit__(self, *args):
        total = time.perf_counter() - self._start
        logger.info(f"{self.name:<50} {total:>7.2f}s")


def func(x_values):
    """Логарифмическая функция для масштабирования высот"""
    return np.log2(x_values * 0.005 + 1)


def scale_elevation(elevations):
    """Масштабирование высот"""
    adjusted_elevations = [(x * SCALE, y * SCALE, func(h) / func(6000) * 25) for (x, y, h) in elevations]
    return np.array(adjusted_elevations)


def frange(start, stop, step):
    """Генератор диапазона с плавающей точкой"""
    while start < stop:
        yield start
        start += step


def GetHeight(points, step, water=None):
    """
    Получает высоту для точек с возможностью поиска ближайших значений
    и учетом отступа от воды
    """
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
        # Учет отступа от воды
        coef = 1
        if water is not None and len(water) > 0:
            point = Point(lon_utm, lat_utm)
            mins_dist = [point.distance(area) for area in water]
            min_dist = min(mins_dist)[0]
            coef = min(min_dist, WATER_SETBACK) / WATER_SETBACK

        lon_wgs84 = points_wgs84[i][0]
        lat_wgs84 = points_wgs84[i][1]
        
        h = elevation_data.get_elevation(lat_wgs84, lon_wgs84)

        # Поиск ближайшего значения высоты, если текущее None
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

    return new_points


def circle_points(cx, cy, radius, num_points):
    """Создает точки круга с заданным центром, радиусом и количеством точек"""
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    return points


def get_city_coordinates(obl_name):
    """Получает координаты городов в области через OSM API"""
    overpass = Overpass()
    nominatim = Nominatim()

    # Находим ID области при помощи Nominatim
    # - admin_level=2 - страна
    # - admin_level=4 - регион/область
    # - admin_level=6 - район
    # - admin_level=8 - поселения.
    area = nominatim.query(obl_name, featuretype='relation', adminLevel=4)
    areaId = area.areaId()

    # Создаем запрос для Overpass API
    query = overpassQueryBuilder(
        area=areaId,
        elementType='node',
        selector='place~"city|town"',
        includeGeometry=False
    )

    # Выполняем запрос и обрабатываем результат
    result = overpass.query(query)

    # Извлечение и печать всех найденных городов
    print(f"Количество городов {len(result.elements())}")
    return result.elements()


def make_city_stl(city_data, step):
    """Создает STL модели для городов"""
    wgs2utm = Transformer.from_crs(WGS84, UTM, always_xy=True)
    utm2wgs = Transformer.from_crs(UTM, WGS84, always_xy=True)

    stl_city = list()
    for city in city_data:
        print(city.tags().get('name'))  # Отображаем название

        centr = wgs2utm.transform(city.lon(), city.lat())
        points_utm = circle_points(centr[0], centr[1], 5000, 20)
        points_wgs = [utm2wgs.transform(x, y) for (x, y) in points_utm]
        circle = GetHeight(points_wgs, step)
        circle = [(x1, y1, h2) for (x1, y1), (x2, y2, h2) in zip(points_utm, circle)]

        max_z = max(circle, key=lambda x: x[2])[2]
        circle_bot = np.array(scale_elevation([(x, y, 0) for (x, y, _) in circle]))
        circle_top = np.array(scale_elevation([(x, y, max_z + 100) for (x, y, _) in circle]))

        tri = Delaunay(circle_bot[:, :2])

        bot_mesh = mesh.Mesh(np.zeros(tri.simplices.shape[0], dtype=mesh.Mesh.dtype))
        top_mesh = mesh.Mesh(np.zeros(tri.simplices.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(tri.simplices):
            for j in range(3):
                bot_mesh.vectors[i][j] = circle_bot[f[j], :]
                top_mesh.vectors[i][j] = circle_top[f[j], :]

        wall_mesh = wall_stl(circle_bot, circle_top)
        stl_city.append(combined_stl(top_mesh, wall_mesh, bot_mesh))

    return stl_city


def multy_to_polygon(geometry, src):
    """Преобразует геометрию в полигоны и преобразует в UTM"""
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
    """Обходит полигон с заданным шагом"""
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
    """Создает контур из геометрии"""
    with Timer("Contour creation"):
        contour_raw = [bypass_polygon(poly, step_m) for poly in geo_simply.geometry]
        contour = []
        for poly in contour_raw:
            if len(poly) > 50:
                p = Polygon(poly)
                minx, miny, maxx, maxy = p.bounds
                full_point_x = int((maxx - minx) / step_m)
                if full_point_x > 0:
                    contour.append(p)
    return contour


def make_mesh(contour, step_m):
    """Создает сетку точек внутри контура"""
    mesh_points = [[] for _ in range(len(contour))]
    with Timer("Mesh generation"):
        for i, polygon in enumerate(contour):
            minx, miny, maxx, maxy = polygon.bounds
            for x in frange(minx, maxx, step_m):
                for y in frange(miny, maxy, step_m):
                    point = Point(x, y)
                    mesh_points[i].append(point)
    return mesh_points


def filter_mesh(contour, mesh_points):
    """Фильтрует точки сетки, оставляя только те, что внутри контура"""
    mesh_inside = []
    with Timer("Mesh filtering"):
        for i, polygon in enumerate(contour):
            gdf_points = gpd.GeoDataFrame({'geometry': mesh_points[i]})
            mesh_inside.append(gdf_points[gdf_points.within(polygon)])
    return mesh_inside


def area_stl(contour, my_mesh):
    """Создает поверхность области"""
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
    """Создает стенки STL"""
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
    """Создает дно STL"""
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
    """Объединяет несколько STL мешей в один"""
    with Timer("STL combination"):
        combined_vectors = np.concatenate([area.vectors, wall.vectors, bottom.vectors])
        combined_mesh = mesh.Mesh(np.zeros(combined_vectors.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(combined_vectors):
            combined_mesh.vectors[i] = f
        combined_mesh.update_normals()
    return combined_mesh


def vertices_faces_to_mesh(vertices, faces):
    """Преобразует вершины и грани в STL меш"""
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        return None
    
    stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        stl_mesh.vectors[i] = vertices[face]
    
    stl_mesh.update_normals()
    return stl_mesh


def make_stl_obl(utm_contour, utm_mesh, utm_contour_zero):
    """Создает STL модели для области"""
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
