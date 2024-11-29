import copy
import math
import sys
import time
import tracemalloc

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

SCALE = 0.0003
WATER_SETBACK = 10 * 1000 # 10 км от воды
WGS84 = 'EPSG:4326'
UTM = 'EPSG:32646'
REDUCE = 0.10 # mm


_start_time = time.perf_counter()
_start_time_total = time.perf_counter()

tracemalloc.start()

def func(x_values):
    return np.log2(x_values*0.005 + 1)

def scale_elevation(elevations):
    adjusted_elevations = [(x * SCALE, y * SCALE, func(h) / func(6000) * 25)  for (x, y, h) in elevations]
    return np.array(adjusted_elevations)

def frange(start, stop, step):
    while start < stop:
        yield start
        start += step



def get_city_coordinates(obl_name):
    # Инициализация Overpass и Nominatim
    overpass = Overpass()
    nominatim = Nominatim()

    # Находим ID области при помощи Nominatim
    # - admin_level=2 - страна
    # - admin_level=4 - регион/область
    # - admin_level=6 - район
    # - admin_level=8 - поселения.
    area = nominatim.query(obl_name, featuretype='relation', adminLevel=4) # используйте название вашей области
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
    print(f"Количество районов {len(result.elements())}")
    # for element in result.elements():
    #     print(element.tags().get('name'))  # Отображаем название
    return result.elements()


def make_city_stl(city_data, step):
    wgs2utm = Transformer.from_crs(WGS84, UTM, always_xy=True)
    utm2wgs = Transformer.from_crs(UTM, WGS84, always_xy=True)

    stl_city = list()
    for city in city_data:
        print(city.tags().get('name'))  # Отображаем название

        centr = wgs2utm.transform(city.lon(), city.lat())
        points_utm = circle_points(centr[0], centr[1] , 5000, 20)
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


def circle_points(cx, cy, radius, num_points):
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    return points


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

    elevation_data = srtm.get_data(local_cache_dir="data/tmp_cache")
    transformer = Transformer.from_crs(UTM, WGS84, always_xy=True)

    points_utm = points
    points_wgs84 = [transformer.transform(lon, lat) for (lon, lat) in points]

    new_points = list()
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


def print_time():
    global _start_time
    time_job = time.perf_counter() - _start_time
    _start_time = time.perf_counter()
    return time_job


def multy_to_polygon(geometry, src):
    sum_poligons = list()
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
    points_on_contour = list()

    distance = 0
    while distance < length:
        point = contour.interpolate(distance)
        points_on_contour.append(Point(point.x, point.y))
        distance += step

    return points_on_contour


def make_contour(geo_simply, step_m):
    contour_raw = [bypass_polygon(poly, step_m) for poly in geo_simply.geometry]
    contour = list()
    for poly in contour_raw:
        if len(poly) > 50: # Считаем что слишком маленькая область
            p = Polygon(poly)
            minx, miny, maxx, maxy = p.bounds
            full_point_x = int((maxx - minx) / step_m)
            if full_point_x > 0:
                contour.append(p)

    return contour


def make_mesh(contour, step_m):
    mesh = [[] for _ in range(len(contour))]
    for i, polygon in enumerate(contour):
        minx, miny, maxx, maxy = polygon.bounds

        for x in frange(minx, maxx, step_m):
            for y in frange(miny, maxy, step_m):
                point = Point(x, y)
                mesh[i].append(point)
    
    return mesh


def filter_mesh(contour, mesh):
    mesh_inside = list()
    for i, polygon in enumerate(contour):
        gdf_points = gpd.GeoDataFrame({'geometry': mesh[i]})
        # mesh_inside.append(gdf_points[gdf_points.intersects(polygon)])
        mesh_inside.append(gdf_points[gdf_points.within(polygon)])

    return mesh_inside


def area_stl(contour, my_mesh):
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
    combined_vectors = np.concatenate([area.vectors, wall.vectors, bottom.vectors])

    combined_mesh = mesh.Mesh(np.zeros(combined_vectors.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(combined_vectors):
        combined_mesh.vectors[i] = f

    # Вычисление нормалей и сохранение в файл
    combined_mesh.update_normals()

    return combined_mesh


def make_stl_obl(utm_contour, utm_mesh, utm_contour_zero):
    count_stl = len(utm_contour)
    contour_scaled_points = list()
    area_scaled_points = list()
    contour_zero_scaled_points = list()
    for i in range(0, count_stl):
        contour_scaled_points.append(scale_elevation(utm_contour[i]))
        area_scaled_points.append(scale_elevation(utm_mesh[i]))
        contour_zero_scaled_points.append(scale_elevation(utm_contour_zero[i]))
        print(f'contour-{len(utm_contour[i])} | area-{len(utm_mesh[i])} | zero-{len(utm_contour_zero[i])}')

        min_h = min(area_scaled_points[i], key=lambda x: x[2])[2]
        max_h = max(area_scaled_points[i], key=lambda x: x[2])[2]
        print(f'Height stl {max_h:.2f} mm | ({min_h:.2f}:{max_h:.2f}) | diff {max_h - min_h:.2f} mm')

    list_stl = list()
    for i in range(0, count_stl):
        area = area_stl(contour_scaled_points[i], area_scaled_points[i])
        wall = wall_stl(contour_scaled_points[i], contour_zero_scaled_points[i])
        bottom = bottom_stl(contour_zero_scaled_points[i])
        list_stl.append(combined_stl(area, wall, bottom))
        print(f"Make STL №{i+1:02}: {print_time():.2f}s")

    return list_stl


def generate(obl_name: str, path_save: str, step_m: int, gpkd, water) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    # -----------------------------------------------------------------
    oblast = gpkd[gpkd["region"] == obl_name]
    print('-' * 100)
    print(f"Start {obl_name} | step_m {step_m}m | distance point {step_m * SCALE:.2f}mm")

    geo_simply = multy_to_polygon(oblast.geometry, WGS84)
    geo_simply = multy_to_polygon(geo_simply.buffer(-(REDUCE / SCALE)), UTM) # уменьшаем в глубь
    print(f'Reduce {REDUCE:.2f}mm | {REDUCE / SCALE:.2f}m')
    contour = make_contour(geo_simply, step_m)
    print(f'Contour done: {print_time():.2f}s')
    area_mesh = filter_mesh(contour, make_mesh(contour, step_m))
    print(f"Mesh done | count {len(area_mesh)}: {print_time():.2f}s")

    # -----------------------------------------------------------------
    #  Height
    # -----------------------------------------------------------------
    min_h = 6000
    max_h = 0
    for i, polygon in enumerate(contour):
        contour[i] = GetHeight(polygon.exterior.coords, step_m, water)
        min_h = min(min_h, min(contour[i], key=lambda x: x[2])[2])
        max_h = max(max_h, max(contour[i], key=lambda x: x[2])[2])
        
    for i, points in enumerate(area_mesh):
        area_mesh[i] = GetHeight([(point.x, point.y) for point in points.geometry], step_m, water)
        min_h = min(min_h, min(area_mesh[i], key=lambda x: x[2])[2])
        max_h = max(max_h, max(area_mesh[i], key=lambda x: x[2])[2])

    print(f'Height done ({min_h}:{max_h}): {print_time():.2f}s')

    # -----------------------------------------------------------------
    #  Convert to UTM
    # -----------------------------------------------------------------
    # list(x, y, h)
    utm_contour = list()
    for points in contour:
        utm_contour.append([(point[0], point[1], point[2]) for point in points])

    utm_mesh = copy.deepcopy(utm_contour)
    for i, points in enumerate(area_mesh):
        utm_mesh[i] += [(point[0], point[1], point[2]) for point in points]

    utm_contour_zero = list()
    for points in utm_contour:
        utm_contour_zero.append([(point[0], point[1], 0) for point in points])

    # -----------------------------------------------------------------
    #  STL
    # -----------------------------------------------------------------
    list_stl = make_stl_obl(utm_contour, utm_mesh, utm_contour_zero)
    # info_city = get_city_coordinates(name_oblast)
    # stl_city = make_city_stl(info_city, step_m)

    # for i, obl in enumerate(list_stl):
    #     if stl_city[i] is None:
    #         obl.update_normals()
    #         obl.save(f'{path_save}{obl_name}_{step_m}_{i}.stl')
    #     else:
    #         combined_mesh_data = np.concatenate([obl.data, stl_city[i].data])
    #         combined_mesh = mesh.Mesh(combined_mesh_data)
    #         combined_mesh.update_normals()
    #         combined_mesh.save(f'{path_save}{obl_name}_{step_m}_{i}.stl')
    for i, obl in enumerate(list_stl):
        obl.update_normals()
        obl.save(f'{path_save}{obl_name}_{step_m}_{i}.stl')


    # combined_mesh_data = np.concatenate([list_stl[0].data] + [city.data for city in stl_city])
    # combined_mesh = mesh.Mesh(combined_mesh_data)
    # combined_mesh.update_normals()
    # combined_mesh.save(f'{path_save}{obl_name}_{step_m}_{i}.stl')

    print(f"All done | time {((time.perf_counter() - _start_time_total)/60):.2f}m")
    return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error\nfull_generate.py step_m (obl)")
        geo_json = gpd.read_file("data/russia_regions.geojson")
        for i in range(0, len(geo_json["region"])):
            print(f'{i:02}\t{geo_json["region"][i]:<20}')
        exit(1)

    step = int(sys.argv[1])
    path = "stl/"
    gpkd = gpd.read_file("data/russia_regions.geojson")

    water = list()
    paths_water = {}
    
    for path_water in paths_water:
        with open(path_water, 'r') as f:
            sea_data = geojson.load(f)
        water.append([shape(feature['geometry']) for feature in sea_data['features']])

    if len(sys.argv) == 3:
        obl_name = sys.argv[2]
        generate(obl_name, path, step, gpkd, water)
    else:

        for i in range(0, len(gpkd["region"])):
            print(f'{i:02}\t{gpkd["region"][i]:<20}')
            generate(gpkd["region"][i], path, step, gpkd, water)

    print('-' * 100)
    size, peak = tracemalloc.get_traced_memory()
    size /= 1024
    peak /= 1024
    print(f"{size=}, {peak=}")
