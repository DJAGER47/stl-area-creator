import lib
import sys
import math
import numpy as np
from stl import mesh
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from pyproj import Proj
import geopandas as gpd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection


s = 0.0003
# scale = (s, s, 0.02)
scale = (s, s, 0.002)

obl_name = sys.argv[1]
step = int(sys.argv[2])
file_name = f"{obl_name}_{step}"


# Создаём функцию для пересчета списков точек
def convert_coords(points):

    new_points = points
    new_points = []
    utm_proj = Proj(proj='utm', zone=33, ellps='WGS84', preserve_units=False)
    for lat, lon, h in points:
        lon, lat = utm_proj(lon, lat)
        new_points.append((lat, lon, h))

    # min_lat = min(point[0] for point in new_points)
    # min_lon = min(point[1] for point in new_points)
    # new_points = [(lat - min_lat, lon-min_lon, h)
    #               for lat, lon, h in new_points]

    return new_points


def closest_point(points, target_x, target_y):
    min_distance = float('inf')
    closest_point_index = -1

    # Поиск точки с минимальным расстоянием до (target_x, target_y)
    for i, (x, y, z) in enumerate(points):
        distance = math.sqrt((target_x - x)**2 + (target_y - y)**2)
        if distance < min_distance:
            min_distance = distance
            closest_point_index = i

    if closest_point_index != -1:
        return points[closest_point_index][2]
    else:
        assert 0

# path = f'data/1_contour/{obl_name}.bin'
# contour = lib.read_binary_coordinates_from_file(path)
# contour_polygon = Polygon(contour)


path = f'data/2_mesh/{file_name}_simply_zero.bin'
wgs84_contour_zero = lib.read_binary_coordinates_from_file(path)
utm_contour_points = convert_coords(wgs84_contour_zero)
wgs84_contour_zero_polygon = Polygon(wgs84_contour_zero)

path = f'data/3_height/{file_name}_simply.bin'
wgs84_contour_height = lib.read_binary_coordinates_from_file(path)
wgs84_contour_height = [[lat, lon, h if h != -1 else
                        closest_point(wgs84_contour_height, lat, lon)]
                        for lat, lon, h in wgs84_contour_height]

path = f'data/3_height/{file_name}.bin'
wgs84_area_points = lib.read_binary_coordinates_from_file(path)
wgs84_area_points = [[lat, lon, h if h != -1 else
                     closest_point(wgs84_area_points, lat, lon)]
                     for lat, lon, h in wgs84_area_points]
utm_area_points = convert_coords(wgs84_area_points)


# -----------------------------------------------------------------
#  поверхность
# -----------------------------------------------------------------
utm_area_scaled_points = np.array([(x*scale[0], y*scale[1], z*scale[2])
                                   for x, y, z in utm_area_points])

wgs84_area_points = np.array(wgs84_area_points)
tri_area = Delaunay(wgs84_area_points[:, :2])

print("Delaunay done")

filtered_triangles_area = []
for simplex in tri_area.simplices:
    triangle_points = wgs84_area_points[simplex]
    triangle = Polygon(triangle_points)
    intersection = wgs84_contour_zero_polygon.intersection(triangle)

    if np.isclose(triangle.area, intersection.area):
        filtered_triangles_area.append(simplex)

print("filtered_triangles_area done")

area_surface_mesh = mesh.Mesh(
    np.zeros(len(filtered_triangles_area), dtype=mesh.Mesh.dtype))
for i, f in enumerate(filtered_triangles_area):
    for j in range(3):
        area_surface_mesh.vectors[i][j] = utm_area_scaled_points[f[j], :]

print("area_surface_mesh done")
# -----------------------------------------------------------------
#  стенка
# -----------------------------------------------------------------
assert len(wgs84_contour_zero) == len(wgs84_contour_height)
len_wall = len(wgs84_contour_zero)

wall_triangles = []
for i in range(0, len_wall-1):
    wall_triangles.append([i + 1, len_wall + i, len_wall + i + 1])
    wall_triangles.append([i, i+1, len_wall + i])

wgs84_wall_points = wgs84_contour_zero + wgs84_contour_height
utm_wall_points = convert_coords(wgs84_wall_points)
utm_wall_scaled_points = np.array([(x*scale[0], y*scale[1], z*scale[2])
                                   for x, y, z in utm_wall_points])

print("wall_triangles done")

wall_surface_mesh = mesh.Mesh(
    np.zeros(len(wall_triangles), dtype=mesh.Mesh.dtype))
for i, f in enumerate(wall_triangles):
    for j in range(3):
        wall_surface_mesh.vectors[i][j] = utm_wall_scaled_points[f[j], :]

print("wall_surface_mesh done")
# -----------------------------------------------------------------
#  Дно
# -----------------------------------------------------------------
wgs83_bottom_points = wgs84_contour_zero
utm_bottom_scaled_points = np.array([(x*scale[0], y*scale[1], z*scale[2])
                                     for x, y, z in utm_contour_points])

wgs84_bottom_points = np.array(wgs83_bottom_points)
tri_bottom = Delaunay(wgs84_bottom_points[:, :2])

print("Delaunay done")

filtered_triangles_bottom = []
for simplex in tri_bottom.simplices:
    triangle_points = wgs84_bottom_points[simplex]
    triangle = Polygon(triangle_points)
    intersection = wgs84_contour_zero_polygon.intersection(triangle)

    if np.isclose(triangle.area, intersection.area):
        filtered_triangles_bottom.append(simplex)

print("filtered_triangles_bottom done")

bottom_surface_mesh = mesh.Mesh(
    np.zeros(len(filtered_triangles_bottom), dtype=mesh.Mesh.dtype))
for i, f in enumerate(filtered_triangles_bottom):
    for j in range(3):
        bottom_surface_mesh.vectors[i][j] = utm_bottom_scaled_points[f[j], :]

print("bottom_surface_mesh done")
# -----------------------------------------------------------------
# Обьединение
# -----------------------------------------------------------------
combined_vectors = np.concatenate(
    [area_surface_mesh.vectors,
     wall_surface_mesh.vectors,
     bottom_surface_mesh.vectors])

combined_mesh = mesh.Mesh(
    np.zeros(combined_vectors.shape[0], dtype=mesh.Mesh.dtype))

for i, f in enumerate(combined_vectors):
    combined_mesh.vectors[i] = f

# Вычисление нормалей и сохранение в файл
combined_mesh.update_normals()
combined_mesh.save(f'stl/{file_name}.stl')
print("All done")

# -----------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.plot(points[:, 1], points[:, 0], 'bo', markersize=1)

# for simplex in tri.simplices:
#     ax.plot(points[simplex, 1], points[simplex, 0], alpha=0.1, color='g')

# for simplex in filtered_triangles:
#     ax.plot(points[simplex, 1], points[simplex, 0],
#             alpha=0.8, color='r', linestyle='--')

# for simplex in deleted_triangles:
#     ax.plot(points[simplex, 1], points[simplex, 0],
#             alpha=0.8, color='r', linestyle='--')

# draw_lat, draw_lon, draw_h = zip(*contour)
# ax.plot(draw_lon, draw_lat, alpha=0.5, color='g', linestyle=':')
# draw_lat, draw_lon, draw_h = zip(*contour_simply)
# ax.plot(draw_lon, draw_lat, alpha=0.5, color='b', linestyle='-')

# # plt.grid(True, linestyle='--')  # прерывистая линия для сетки
# ax.set_aspect('equal')
# plt.show()

# -----------------------------------------------------------------
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red')

# for simplex in filtered_triangles:
#     x = points[simplex, 0]
#     y = points[simplex, 1]
#     z = points[simplex, 2]
#     # Рисуем грани треугольников
#     vtx = [list(zip(x, y, z))]
#     ax.add_collection3d(Poly3DCollection(
#         vtx, facecolors='cyan', linewidths=1, edgecolors='b', alpha=0.5))

# # Отрисовываем треугольники
# for simplex in tri.simplices:
#     x = points[simplex, 0]
#     y = points[simplex, 1]
#     z = points[simplex, 2]
#     # Рисуем грани треугольников
#     vtx = [list(zip(x, y, z))]
#     ax.add_collection3d(Poly3DCollection(
#         vtx, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.1))

# -----------------------------------------------------------------
# x, y, h = zip(*contour)
# ax.scatter(x, y, [1]*len(x), color='green')
# plt.show()

# lats, lons, h = zip(*points)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(lons, lats, h,
#                 cmap='viridis', edgecolor='none')

# ax.set_box_aspect([1, 1, 0.15])  # равное соотношение сторон
# plt.show()
