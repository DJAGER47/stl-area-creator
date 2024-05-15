import struct
import matplotlib.pyplot as plt

WGS84 = 4326
UTM = 32633

# coords = [(1.5, 2.3, 3.7), (4.0, 5.6, 6.8), (7.1, 8.2, 9.9)]


def write_coordinates_to_file(filename, coordinates):
    with open(filename, "w") as file:
        for coord in coordinates:
            x, y, z = coord
            file.write(f"{x} {y} {z}\n")


def read_coordinates_from_file(filename):
    coordinates = []
    with open(filename, "r") as file:
        for line in file:
            x, y, z = map(float, line.split())
            coordinates.append((x, y, z))
    return coordinates


def write_binary_coordinates_to_file(filename, coordinates):
    with open(filename, "wb") as file:
        for coord in coordinates:
            # 'fff' означает три float числа
            packed_data = struct.pack("fff", *coord)
            file.write(packed_data)


def read_binary_coordinates_from_file(filename):
    coordinates = []
    with open(filename, "rb") as file:
        while True:
            # Каждый float занимает 4 байта, поэтому читаем 4*3=12 байт
            packed_data = file.read(12)
            if not packed_data:
                break
            coord = struct.unpack("fff", packed_data)
            coordinates.append(coord)
    return coordinates
