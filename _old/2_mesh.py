import sys
import lib
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

obl_name = sys.argv[1]
step = int(sys.argv[2])
file_name = f"{obl_name}_{step}"


def bypass_poligon(geo):
    # Расчет интерполированных точек
    contour = geo.exterior
    length = contour.length
    points_on_contour = list()

    distance = 0
    while distance < length:
        point = contour.interpolate(distance)
        points_on_contour.append((point.x, point.y))
        distance += step

    if len(points_on_contour) > 2:
        mesh = gpd.GeoDataFrame(
            [Polygon(points_on_contour)], columns=['geometry'])
        mesh = mesh['geometry']
        mesh.set_crs(epsg=lib.UTM, inplace=True)
        mesh = mesh.to_crs(epsg=lib.WGS84)
        return zip(*mesh[0].exterior.coords)

    else:
        return list(), list()


def multy_to_polygon(geo):
    sum_poligons = list()
    for multi in geo:
        if isinstance(multi, MultiPolygon):
            for polygon in multi.geoms:
                sum_poligons.append(polygon)
        else:
            sum_poligons.append(multi)

    geo = gpd.GeoDataFrame(sum_poligons, columns=['geometry'])
    geo = geo['geometry']
    return geo


def bypass_multy(geo, h):
    ret_points = list()
    for poly in geo:
        if poly.area > 100:
            bypass_tmp_lat, bypass_tmp_lon = bypass_poligon(poly)
            tmp = list(zip(bypass_tmp_lat, bypass_tmp_lon,
                           [h]*len(bypass_tmp_lat)))
            ret_points += tmp

    return ret_points


path = f'data/1_contour/{obl_name}.bin'
points = lib.read_binary_coordinates_from_file(path)

lat_original, lon_original, h_original = zip(*points)

geo_original = gpd.GeoDataFrame(
    [Polygon(list(zip(lat_original, lon_original)))], columns=['geometry'])
geo_original = geo_original['geometry']
geo_original.set_crs(epsg=lib.WGS84, inplace=True)
geo_original = geo_original.to_crs(epsg=lib.UTM)

# -------------------------------------------------------------------------------
geo_original = multy_to_polygon(geo_original)

ret_points = bypass_multy(geo_original, 0)
path = f'data/2_mesh/{file_name}_simply_zero.bin'
points = lib.write_binary_coordinates_to_file(path, ret_points)

ret_points = [[lat, lon, 0] for lat, lon, _ in ret_points]
path = f'data/2_mesh/{file_name}_simply.bin'
points = lib.write_binary_coordinates_to_file(path, ret_points)

tmp_geo = geo_original
len_geo = sum(poly.exterior.length for poly in tmp_geo)
while len_geo > 1000:
    tmp_geo = tmp_geo.buffer(-step)
    tmp_geo = multy_to_polygon(tmp_geo)
    ret_points += bypass_multy(tmp_geo, 0)
    len_geo = sum(polygon.exterior.length for polygon in tmp_geo)
    print(f"\r{len_geo:010.01f} m", end='', flush=True)

path_out = f'data/2_mesh/{file_name}.bin'
lib.write_binary_coordinates_to_file(path_out, ret_points)

print(f'\rMesh done! {len(ret_points)} len')

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.plot(lon_original, lat_original, alpha=0.5, color='g')
# draw_lat, draw_lon, draw_h = zip(*ret_points)
# ax.plot(draw_lon, draw_lat, 'bo', markersize=1)
# ax.set_aspect('equal')
# plt.show()
