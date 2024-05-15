# for poly in utm_contour:
#     x, y = zip(*poly)
#     ax.plot(x, y, alpha=0.5, color='g', marker='o', markersize=1)
# -----------------------------------------------------------------
# utm_mesh = utm_contour
# len_geometry = len(utm_geo_simply.geometry)
# for i, polygon in enumerate(utm_geo_simply.geometry):
#     tmp_geo = gpd.GeoDataFrame({'geometry': [polygon]})
#     tmp_geo.set_crs(UTM, inplace=True)
#     len_geo_start = sum(poly.exterior.length for poly in tmp_geo.geometry)
#     len_geo = len_geo_start
#     while len_geo > step_m:
#         tmp_geo = tmp_geo.buffer(-step_m)
#         tmp_geo = multy_to_polygon(tmp_geo, UTM)
#         utm_mesh[i] += utm_bypass_multy(tmp_geo.geometry, step_m)
#         len_geo = sum(poly.exterior.length for poly in tmp_geo.geometry)
#         len_tmp = f"{len_geo:010.01f}:{len_geo_start:010.01f}m"
#         print(f"\r{i:04}:{len_geometry:04} {len_tmp}", end='', flush=True)
#     print(f"\ttime: {print_time():.2f}s")
# -----------------------------------------------------------------


def convert_crs(geo):
    centroid = geo.geometry.unary_union.centroid
    mean_lon = centroid.x
    mean_lat = centroid.y
    utm_zone = int((mean_lon + 180) / 6) + 1
    hemisphere = '+north' if mean_lat >= 0 else '+south'
    utm_crs_code = f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    utm_crs = CRS(utm_crs_code)
    return geo.to_crs(utm_crs)
    # return geo.to_crs(geo.estimate_utm_crs())


# def area_stl(my_mesh, contour):
#     tri_area = Delaunay(my_mesh[:, :2])
#     print(f"\tDelaunay\ttime: {print_time():.2f}s")

#     polygon = Polygon(contour)
#     filtered_triangles_area = []
#     for simplex in tri_area.simplices:
#         triangle = Polygon(my_mesh[simplex])
#         # intersection = polygon.intersection(triangle)
#         # if np.isclose(triangle.area, intersection.area):
#         #     filtered_triangles_area.append(simplex)

#         if polygon.contains(triangle):
#             filtered_triangles_area.append(simplex)
#     print(f"\tarea_triangles\ttime: {print_time():.2f}s")

#     area_surface_mesh = mesh.Mesh(np.zeros(len(filtered_triangles_area), dtype=mesh.Mesh.dtype))
#     for i, f in enumerate(filtered_triangles_area):
#         for j in range(3):
#             area_surface_mesh.vectors[i][j] = my_mesh[f[j], :]
#     print(f"\tarea_surface\ttime: {print_time():.2f}s")

#     return area_surface_mesh
