import copy
import math
import sys
import time
import tracemalloc

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests
from pyproj import Transformer
from scipy.spatial import Delaunay
from shapely.geometry import MultiPolygon, Point, Polygon

import pandas as pd

import srtm
from stl import mesh

MIN_AREA = 100
WGS84 = 'EPSG:4326'
UTM = 'EPSG:32646'

SCALE = 0.0003

_start_time = time.perf_counter()
_start_time_total = time.perf_counter()

tracemalloc.start()


def func(x):
    x = x*math.tan(math.radians(1)) + 1
    # x = x * 0.0001 + 1
    x = np.log2(x)
    x = x*(math.tan(math.radians(100)) + 10)
    return x

    # if x == 0:
    #     return x

    # x = x * 0.0001
    # x = (x*np.tan(np.radians(170)) + 1) * 10
    # return x

def log_scale_elevation(elevations):

    adjusted_elevations = [(x * SCALE, y * SCALE, func(h))  for (x, y, h) in elevations]

    return np.array(adjusted_elevations)


def convert_crs(x, y, to_crs=UTM, from_crs=WGS84):
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    new_x, new_y = transformer.transform(x, y)
    return Point(new_x, new_y)


def get_city_coordinates(city_name):
    # Заголовки запроса
    headers = {
        'User-Agent': 'MyGeocodingApp 1.0 (example@example.com)',  # Замените на свой email
        'Accept-Language': 'ru',  # Получить ответ на русском языке
    }

    url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json"
    time.sleep(1)
    response = requests.get(url, headers=headers)

    data = response.json()
    if data:
        lat = data[0]["lat"]
        lon = data[0]["lon"]
        return (lat, lon)
    else:
        return None


def circle_points(cx, cy, radius, num_points):
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append(Point(x, y))
    return points


def GetHeight_utm_srtm(points, step_m):
    find = [
        (0, -step_m),
        (0, step_m),
        (-step_m, 0),
        (step_m, 0),
        (-step_m, -step_m),
        (-step_m, step_m),
        (step_m, -step_m),
        (step_m, step_m),
    ]

    elevation_data = srtm.get_data(local_cache_dir="data/tmp_cache")

    utm_points = points.set_crs(UTM)
    wgs84_points = utm_points.to_crs(WGS84)

    utm_coordinates = [(point.x, point.y) for point in utm_points.geometry]
    wgs84_coordinates = [(point.x, point.y) for point in wgs84_points.geometry]

    new_points = list()
    for i, (lon, lat) in enumerate(wgs84_coordinates):
        h = elevation_data.get_elevation(lat, lon)

        index = 0
        multy = 1
        while h is None:
            lon_utm = utm_coordinates[i][0] + multy * find[index][0]
            lat_utm = utm_coordinates[i][1] + multy * find[index][1]
            transformer = Transformer.from_crs(UTM, WGS84, always_xy=True)
            wgs84_lon, wgs84_lat = transformer.transform(lon_utm, lat_utm)
            h = elevation_data.get_elevation(wgs84_lat, wgs84_lon)
            index += 1
            if len(find) == index:
                index = 0
                multy += 1

        new_points.append((utm_coordinates[i][0], utm_coordinates[i][1], h))

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


def utm_bypass_poligon(geo, step_m: int):
    contour = geo.exterior
    length = contour.length
    points_on_contour = list()

    distance = 0
    while distance < length:
        point = contour.interpolate(distance)
        points_on_contour.append(Point(point.x, point.y))
        distance += step_m

    return points_on_contour


def utm_bypass_multy(geo, step_m: int):
    ret_points = list()
    for poly in geo:
        if poly.area > MIN_AREA:
            tmp = utm_bypass_poligon(poly, step_m)
            if len(tmp) > 0:
                ret_points += tmp

    return ret_points


def area_stl(my_mesh, contour):
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


def wall_stl(contour_zero, contour):
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

    polygon = Polygon(contour_zero)
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


def generate(obl_name: str, path_save: str, step_m: int, gpkd) -> None:
    # fig, ax = plt.subplots(figsize=(5, 5))
    # -----------------------------------------------------------------
    #  Contour
    # -----------------------------------------------------------------
    print(f"\nStart {obl_name} step_m {step_m}m")
    oblast = gpkd[gpkd["NAME_1"] == obl_name]
    print(f'Contour done!   time: {print_time():.2f}s')
    # oblast.plot(ax=ax, alpha=0.1, color='r')

    # -----------------------------------------------------------------
    #  Mesh
    # -----------------------------------------------------------------
    wgs84_geo_original = oblast
    print(f'wgs84: {wgs84_geo_original.crs} ## EPSG:4326')
    utm_geo_original = oblast.to_crs(UTM)
    # utm_geo_original.plot(ax=ax, alpha=0.1, color='r')
    # -----------------------------------------------------------------
    utm_geo_simply = multy_to_polygon(utm_geo_original, UTM)

    utm_contour_raw = [utm_bypass_poligon(poly, step_m) for poly in utm_geo_simply.geometry]

    utm_contour = list()
    utm_contour_geometry = list()
    for poly in utm_contour_raw:
        if len(poly) > 2:
            p = Polygon(poly)
            minx, miny, maxx, maxy = p.bounds
            full_point_x = int((maxx - minx) / step_m)
            if full_point_x > 0:
                utm_contour.append(poly)
                utm_contour_geometry.append(Polygon(poly))

    utm_mesh = copy.deepcopy(utm_contour)
    for i, polygon in enumerate(utm_contour_geometry):
        minx, miny, maxx, maxy = polygon.bounds

        full_point_x = int((maxx - minx) / step_m)
        for j, x in enumerate(range(int(minx), int(maxx) + 1, step_m)):
            for y in range(int(miny), int(maxy) + 1, step_m):
                point = Point(x, y)
                utm_mesh[i].append(point)
            progress = j / full_point_x * 100
            print(f"\r{i:04} {progress:.2f}%", end='', flush=True)
        print(f"\ttime: {print_time():.2f}s")

    utm_mesh_inside = list()
    for i, polygon in enumerate(utm_contour_geometry):
        gdf_points = gpd.GeoDataFrame({'geometry': utm_mesh[i]})
        utm_mesh_inside.append(gdf_points[gdf_points.intersects(polygon)])
    print(f"inside filtred\tpoint {len(utm_mesh_inside[0])}\ttime: {print_time():.2f}s")

    # -----------------------------------------------------------------
    #  Height
    # -----------------------------------------------------------------
    for i, points in enumerate(utm_mesh_inside):
        utm_mesh_inside[i] = GetHeight_utm_srtm(points, step_m)

    # df_mesh = pd.DataFrame(utm_mesh_inside[0])
    # # Сохраняем в CSV файл
    # df_mesh.to_csv('mesh.csv', index=False)

    for i, points in enumerate(utm_contour):
        gdf_points = gpd.GeoDataFrame({'geometry': points})
        utm_contour[i] = GetHeight_utm_srtm(gdf_points, step_m)

    utm_contour_zero = list()
    for points in utm_contour:
        utm_contour_zero.append([(x, y, 0) for (x, y, z) in points])
    print(f'Height done! time: {print_time():.2f}s')

    # -----------------------------------------------------------------
    #  STL
    # -----------------------------------------------------------------
    #  поверхность
    count_stl = len(utm_contour)

    area_scaled_points = list()
    contour_zero_scaled_points = list()
    contour_scaled_points = list()
    for i in range(0, count_stl):
        area_scaled_points.append(log_scale_elevation(utm_mesh_inside[i]))
        contour_scaled_points.append(log_scale_elevation(utm_contour[i]))
        contour_zero_scaled_points.append(log_scale_elevation(utm_contour_zero[i]))

        print(f'contour-{len(utm_contour[i])} | area-{len(utm_mesh_inside[i])} | zero-{len(utm_contour_zero[i])}')

    area_stl_list = list()
    for i in range(0, count_stl):
        area_stl_list.append(area_stl(area_scaled_points[i], contour_scaled_points[i]))
        print(f"{i:04} area\ttime: {print_time():.2f}s")

    # for triangle in area_stl_list[0].vectors:
    #     vertices = triangle[:, :2]
    #     vertices = np.vstack([vertices, vertices[0]])
    #     ax.plot(vertices[:, 0], vertices[:, 1], linestyle='--', alpha=0.5)
    # ax.set_aspect('equal', adjustable='datalim')
    # plt.show()

    #  стенка
    wall_stl_list = list()
    for i in range(0, count_stl):
        wall_stl_list.append(wall_stl(contour_zero_scaled_points[i], contour_scaled_points[i]))
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

    # sity
    stl_sity = list()
    # coordinates_obl = ('54.7814057', '32.0461261')  # get_city_coordinates(obl_name)
    # coordinates_obl = convert_crs(float(coordinates_obl[1]), float(coordinates_obl[0]))
    # for i, polygon in enumerate(utm_contour_geometry):
    #     if polygon.contains(coordinates_obl):
    #         utm_points = circle_points(coordinates_obl.x, coordinates_obl.y, 5000, 20)
    #         gdf_points = gpd.GeoDataFrame({'geometry': utm_points})
    #         circle = GetHeight_utm_srtm(gdf_points, step_m)

    #         max_z = max(circle, key=lambda x: x[2])[2]
    #         circle_bot = np.array(log_scale_elevation([(x, y, 0) for (x, y, _) in circle]))
    #         circle_top = np.array(log_scale_elevation([(x, y, max_z + 100) for (x, y, _) in circle]))

    #         tri = Delaunay(circle_bot[:, :2])

    #         bot_mesh = mesh.Mesh(np.zeros(tri.simplices.shape[0], dtype=mesh.Mesh.dtype))
    #         top_mesh = mesh.Mesh(np.zeros(tri.simplices.shape[0], dtype=mesh.Mesh.dtype))
    #         for i, f in enumerate(tri.simplices):
    #             for j in range(3):
    #                 bot_mesh.vectors[i][j] = circle_bot[f[j], :]
    #                 top_mesh.vectors[i][j] = circle_top[f[j], :]

    #         wall_mesh = wall_stl(circle_bot, circle_top)
    #         stl_sity.append(combined_stl(top_mesh, wall_mesh, bot_mesh))
    #     else:
    #         stl_sity.append(None)

    # for i, obl in enumerate(list_stl):
    #     if stl_sity[i] is None:
    #         obl.update_normals()
    #         obl.save(f'{path_save}{obl_name}_{step_m}_{i}.stl')
    #     else:
    #         combined_mesh_data = np.concatenate([obl.data, stl_sity[i].data])
    #         combined_mesh = mesh.Mesh(combined_mesh_data)
    #         combined_mesh.update_normals()
    #         combined_mesh.save(f'{path_save}{obl_name}_{step_m}_{i}.stl')
    for i, obl in enumerate(list_stl):
        obl.update_normals()
        obl.save(f'{path_save}{obl_name}_{step_m}_{i}.stl')

    print(f"All done | time {((time.perf_counter() - _start_time_total)/60):.2f}m")

    # x = [x1, x2, 0]
    # y = [y1, y2, scale_h(0)]
    # plt.plot(x, y)
    # # ax.set_aspect('equal')
    # plt.show()
    return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error\nfull_generate.py step_m (obl)")
        gpkd = gpd.read_file("data/gadm41_RUS.gpkg", layer="ADM_ADM_1")
        for i in range(0, len(gpkd["NAME_1"])):
            print(f'{i:02}\t{gpkd["NAME_1"][i]:<20}\t{gpkd["NL_NAME_1"][i]}')
        exit(-1)

    step = int(sys.argv[1])
    path = "stl/"
    gpkd = gpd.read_file("data/gadm41_RUS.gpkg", layer="ADM_ADM_1")

    if len(sys.argv) == 3:
        obl_name = sys.argv[2]
        generate(obl_name, path, step, gpkd)
    else:

        for i in range(0, len(gpkd["NAME_1"])):
            print(f'{i:02}\t{gpkd["NAME_1"][i]:<20}\t{gpkd["NL_NAME_1"][i]}')
            generate(gpkd["NAME_1"][i], path, step, gpkd)

    size, peak = tracemalloc.get_traced_memory()
    size /= 1024
    peak /= 1024
    print(f"{size=}, {peak=}")
