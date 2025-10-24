import lib
import sys
import geopandas as gpd
import matplotlib.pyplot as plt


obl_name = sys.argv[1]

origing = gpd.read_file("data/gadm41_RUS.gpkg",
                        layer="ADM_ADM_1")  # ADM_ADM_0-3
# print(gdf.columns)
# print(gdf.head())
# print(gdf['NAME_1'][63])
# print(gdf['geometry'][63].head())

oblast = origing[origing["NAME_1"] == obl_name]

points_str = ""
for p in oblast.boundary:
    points_str = str(p)


points_str = points_str[16:].strip()  # MULTILINESTRING
points_str = points_str.replace("(", "").replace(")", "")
points_double = points_str.split(",")
points = list()
for p in points_double:
    p = p.strip().split(" ")
    lat = float(p[1])
    lon = float(p[0])
    points.append((lat, lon, 0))

output_file_path = f"data/1_contour/{obl_name}"
# lib.write_coordinates_to_file(output_file_path + ".txt", points)
lib.write_binary_coordinates_to_file(output_file_path + ".bin", points)

print('Contour done!')

# lat, lon, h = zip(*points)
# fig, ax = plt.subplots(figsize=(5, 5))
# oblast.plot(ax=ax, color="#000000", alpha=0.1)
# oblast.boundary.plot(ax=ax, color="black", linewidth=0.5)
# # gdf['geometry'].boundary.plot(
# #     ax=ax, color='black', linewidth=0.5)
# # ax.plot(lon, lat, color='g')
# ax.set_axis_off()
# plt.tight_layout()
# plt.show()
