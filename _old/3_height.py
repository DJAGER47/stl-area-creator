import lib
import sys
import srtm
import matplotlib.pyplot as plt

obl_name = sys.argv[1]
step = int(sys.argv[2])
file_name = f"{obl_name}_{step}"


def GetHeight(points):
    elevation_data = srtm.get_data(local_cache_dir="data/tmp_cache")
    new_points = [(lat, lon, elevation_data.get_elevation(lat, lon)
                   if h != -1 else 0) for lat, lon, h in points]

    new_points = [(lat, lon, h
                   if h is not None else -1) for lat, lon, h in new_points]
    return new_points


path = [[f'data/2_mesh/{file_name}_simply.bin', f'data/3_height/{file_name}_simply.bin'],
        [f'data/2_mesh/{file_name}.bin', f'data/3_height/{file_name}.bin']]

for p in path:
    points = lib.read_binary_coordinates_from_file(p[0])
    new_points = GetHeight(points)
    lib.write_binary_coordinates_to_file(p[1], new_points)

print(f'Height done! {len(new_points)} len')

# lats, lons, h = zip(*new_points)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(lons, lats, h,
#                 cmap='viridis', edgecolor='none')

# ax.set_box_aspect([1, 1, 0.15])  # равное соотношение сторон
# plt.show()
