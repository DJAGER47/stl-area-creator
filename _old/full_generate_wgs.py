import copy
import math
import sys
import time
import tracemalloc

import geojson
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from pyproj import Transformer
from scipy.spatial import Delaunay
from shapely.geometry import MultiPolygon, Point, Polygon, shape

import srtm
from stl import mesh

METERS_PER_DEGREE = 111320
WATER_SETBACK = 10000 / METERS_PER_DEGREE # 10 км от воды
MIN_AREA = 100
WGS84 = 'EPSG:4326'
UTM = 'EPSG:32646'

SCALE = 0.0003

_start_time = time.perf_counter()
_start_time_total = time.perf_counter()

tracemalloc.start()

def func(x):
    # x = x*math.tan(math.radians(1)) + 1
    # x = np.log2(x)
    # x = x*(math.tan(math.radians(100)) + 10)

    x = x*0.01 + 1
    x = np.log2(x) + 1
    x = np.log10(x) * 30
    return x

def scale_elevation(elevations):
    adjusted_elevations = [(x * SCALE, y * SCALE, func(h))  for (x, y, h) in elevations]
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


def GetHeight(wgs84_points, step, water):
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

    new_points = list()
    for i, (lon, lat) in enumerate(wgs84_points):

        point = Point(lon, lat)
        mins_dist = [point.distance(area) for area in water]
        min_dist = min(mins_dist)[0]
        coef = min(min_dist, WATER_SETBACK) / WATER_SETBACK

        if lat < 60:
            h = elevation_data.get_elevation(lat, lon)
            index = 0
            multy = 1
            while h is None:
                wgs84_lon = wgs84_points[i][0] + multy * find[index][0]
                wgs84_lat = wgs84_points[i][1] + multy * find[index][1]
                h = elevation_data.get_elevation(wgs84_lat, wgs84_lon)
                index += 1
                if len(find) == index:
                    index = 0
                    multy += 1

            if h < 0:
                print(f"h: {h}")
                h = 0
            new_points.append((wgs84_points[i][0], wgs84_points[i][1], h * coef))
        else:
            new_points.append((wgs84_points[i][0], wgs84_points[i][1], 0.01 * coef))

    return new_points


def print_time():
    global _start_time
    time_job = time.perf_counter() - _start_time
    _start_time = time.perf_counter()
    return time_job


def multy_to_polygon(geo, src):
    sum_poligons = list()
    for multi in geo.geometry:
        if isinstance(multi, MultiPolygon):
            for polygon in multi.geoms:
                sum_poligons.append(polygon)
        else:
            sum_poligons.append(multi)

    geo = gpd.GeoDataFrame({'geometry': sum_poligons})
    geo.set_crs(src, inplace=True)
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


def bypass_multy(geo, step):
    ret_points = list()
    for poly in geo:
        if poly.area > MIN_AREA:
            tmp = bypass_polygon(poly, step)
            if len(tmp) > 0:
                ret_points += tmp

    return ret_points


def make_contour(geo_simply, step_deg):
    contour_raw = [bypass_polygon(poly, step_deg) for poly in geo_simply.geometry]
    contour = list()
    for poly in contour_raw:
        if len(poly) > 2:
            p = Polygon(poly)
            minx, miny, maxx, maxy = p.bounds
            full_point_x = int((maxx - minx) / step_deg)
            if full_point_x > 0:
                contour.append(p)

    # coords = np.array(wgs84_contour_geometry[0].exterior.coords)
    # ax.plot(coords[:, 0], coords[:, 1], color='r')
    # plt.show()
    return contour


def make_mesh(contour, step_deg):
    mesh = [[] for _ in range(len(contour))]
    for i, polygon in enumerate(contour):
        minx, miny, maxx, maxy = polygon.bounds

        for x in frange(minx, maxx, step_deg):
            for y in frange(miny, maxy, step_deg):
                point = Point(x, y)
                mesh[i].append(point)
        print(f"\r{i:04}\ttime: {print_time():.2f}s")
    
    # x_coords = [point.x for point in wgs84_mesh[0]]
    # y_coords = [point.y for point in wgs84_mesh[0]]
    # plt.scatter(x_coords, y_coords, color='blue', marker='o', alpha=0.5)
    # plt.show()
    return mesh


def filter_mesh(contour, mesh):
    mesh_inside = list()
    for i, polygon in enumerate(contour):
        gdf_points = gpd.GeoDataFrame({'geometry': mesh[i]})
        # mesh_inside.append(gdf_points[gdf_points.intersects(polygon)])
        mesh_inside.append(gdf_points[gdf_points.within(polygon)])
    print(f"inside filtred\tpoint {len(mesh_inside[0])}\ttime: {print_time():.2f}s")

    # x_coords = [point.x for point in mesh_inside[0].geometry]
    # y_coords = [point.y for point in mesh_inside[0].geometry]
    # plt.scatter(x_coords, y_coords, color='red', marker='o', alpha=0.5)
    # plt.show()
    return mesh_inside


def area_stl(contour, my_mesh):
    tri_area = Delaunay(my_mesh[:, :2])
    print(f"\tDelaunay\ttime: {print_time():.2f}s")

    polygon = Polygon(contour)
    triangle = [Polygon(my_mesh[simplex]) for simplex in tri_area.simplices]

    gdf_points = gpd.GeoDataFrame({'geometry': triangle})
    filtered_triangles_area = gdf_points[gdf_points.within(polygon)]
    print(f"\tarea_triangles\ttime: {print_time():.2f}s")

    # fig, ax = plt.subplots(figsize=(5, 5))
    # coords = np.array(polygon.exterior.coords)
    # ax.plot(coords[:, 0], coords[:, 1], color='blue')

    # for triangle in filtered_triangles_area.geometry:
    #     vertices = np.array(triangle.exterior.coords)
    #     ax.plot(vertices[:, 0], vertices[:, 1], linestyle='--', alpha=0.5)
    # ax.set_aspect('equal', adjustable='datalim')
    # plt.show()

    area_surface_mesh = mesh.Mesh(np.zeros(len(filtered_triangles_area), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(filtered_triangles_area.geometry):
        area_surface_mesh.vectors[i] = np.array(f.exterior.coords[:3])
    print(f"\tarea_surface\ttime: {print_time():.2f}s")

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
    print(f"\twall_triangles\ttime: {print_time():.2f}s")

    wall_surface_mesh = mesh.Mesh(np.zeros(len(wall_triangles), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(wall_triangles):
        for j in range(3):
            wall_surface_mesh.vectors[i][j] = wall_points[f[j], :]
    print(f"\twall_surface\ttime: {print_time():.2f}s")

    return wall_surface_mesh


def bottom_stl(contour_zero):
    tri_bottom = Delaunay(contour_zero[:, :2])
    print(f"\tDelaunay\ttime: {print_time():.2f}s")

    polygon = Polygon(contour_zero).buffer(0)
    filtered_triangles_bottom = []
    for simplex in tri_bottom.simplices:
        triangle = Polygon(contour_zero[simplex])
        if polygon.contains(triangle):
            filtered_triangles_bottom.append(simplex)
    print(f"\tbot_triangles\ttime: {print_time():.2f}s")

    bottom_surface_mesh = mesh.Mesh(np.zeros(len(filtered_triangles_bottom), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(filtered_triangles_bottom):
        for j in range(3):
            bottom_surface_mesh.vectors[i][j] = contour_zero[f[j], :]
    print(f"\tbot_surface\ttime: {print_time():.2f}s")

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

    #  поверхность
    area_stl_list = list()
    for i in range(0, count_stl):
        area_stl_list.append(area_stl(contour_scaled_points[i], area_scaled_points[i]))
        print(f"{i:04} area\ttime: {print_time():.2f}s")

    #  стенка
    wall_stl_list = list()
    for i in range(0, count_stl):
        wall_stl_list.append(wall_stl(contour_scaled_points[i], contour_zero_scaled_points[i],))
        print(f"{i:04} wall\ttime: {print_time():.2f}s")

    #  Дно
    bottom_stl_list = list()
    for i in range(0, count_stl):
        bottom_stl_list.append(bottom_stl(contour_zero_scaled_points[i]))
        print(f"{i:04} bottom\ttime: {print_time():.2f}s")

    # Обьединение
    list_stl = list()
    for i in range(0, count_stl):
        list_stl.append(combined_stl(area_stl_list[i], wall_stl_list[i], bottom_stl_list[i]))
        print(f"{i:04} combined\ttime: {print_time():.2f}s")

    return list_stl


def generate(obl_name: str, path_save: str, step_m: int, gpkd, water) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    transformer = Transformer.from_crs(WGS84, UTM, always_xy=True)
    # -----------------------------------------------------------------
    step_deg = step_m / METERS_PER_DEGREE

    # print(f'utm step {step_m * SCALE * 100*1000}')
    # exit()

    oblast = gpkd[gpkd["region"] == obl_name]
    print(f"\nStart {obl_name} | step_m {step_m}m | step_deg {step_deg}")
    print(f'Contour done!   time: {print_time():.2f}s')
    print(f'wgs84: {oblast.crs} ## EPSG:4326')

    # oblast.plot(ax=ax, alpha=0.1, color='g')
    # oblast_json.plot(ax=ax, alpha=0.1, color='b')
    # plt.scatter(42.4375, 43.352778, color='red')
    # plt.show()
    # exit()

    geo_simply = multy_to_polygon(oblast, WGS84)
    contour = make_contour(geo_simply, step_deg)
    area_mesh = filter_mesh(contour, make_mesh(contour, step_deg))

    # -----------------------------------------------------------------
    #  Height
    # -----------------------------------------------------------------
    for i, polygon in enumerate(contour):
        contour[i] = GetHeight(polygon.exterior.coords, step_deg, water)
        
    for i, points in enumerate(area_mesh):
        area_mesh[i] = GetHeight([(point.x, point.y) for point in points.geometry], step_deg, water)

    print(f'Height done! time: {print_time():.2f}s')

    # oblast.plot(ax=ax, alpha=0.1, color='g')
    # for geom in water:
    #     gdf = gpd.GeoDataFrame({'geometry': [geom[0]]})
    #     gdf.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.1)

    # for (x,y,h,a) in contour[0]:
    #     plt.scatter(x, y, color='red', alpha=a)

    # for (x,y,h,a) in area_mesh[0]:
    #     if a != 0:
    #         plt.scatter(x, y, color='green', alpha=a)
    # plt.show()
    # exit()

    # -----------------------------------------------------------------
    #  Convert to UTM
    # -----------------------------------------------------------------
    # list(x, y, h)

    utm_contour = list()
    for points in contour:
        utm_contour.append([transformer.transform(point[0], point[1], point[2]) for point in points])

    utm_mesh = copy.deepcopy(utm_contour)
    for i, points in enumerate(area_mesh):
        utm_mesh[i] += [transformer.transform(point[0], point[1], point[2]) for point in points]

    utm_contour_zero = list()
    for points in utm_contour:
        utm_contour_zero.append([(point[0], point[1], 0) for point in points])

    print(f'Transform to utm! time: {print_time():.2f}s')
    # -----------------------------------------------------------------
    #  STL
    # -----------------------------------------------------------------
    list_stl = make_stl_obl(utm_contour, utm_mesh, utm_contour_zero)
    # info_city = get_city_coordinates(name_oblast)
    # stl_city = make_city_stl(info_city, step_deg)

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
    paths_water = {"data/sea/черное.geojson"}
    
    for path_water in paths_water:
        with open(path_water, 'r') as f:
            sea_data = geojson.load(f)
        water.append([shape(feature['geometry']) for feature in sea_data['features']])

    if len(sys.argv) == 3:
        obl_name = sys.argv[2]
        generate(obl_name, path, step, gpkd, water)
    else:

        for i in range(0, len(gpkd["region"])):
            print('-' * 30)
            print(f'{i:02}\t{gpkd["region"][i]:<20}')
            generate(gpkd["region"][i], path, step, gpkd, water)


    size, peak = tracemalloc.get_traced_memory()
    size /= 1024
    peak /= 1024
    print(f"{size=}, {peak=}")
