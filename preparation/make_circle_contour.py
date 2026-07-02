#!/usr/bin/env python3
"""
Скрипт для генерации кругового контура.

Примеры:
  # С параметрами по умолчанию
  python make_circle_contour.py

  # С указанием центра и радиуса
  python make_circle_contour.py --lat 43.348397 --lon 42.454421 --radius 50000

  # С показом графика
  python make_circle_contour.py --plot
"""

import sys
import os
import argparse
import geojson
from shapely.geometry import Point, mapping
from shapely.geometry.polygon import Polygon
import math
from pyproj import Transformer

# Добавляем корневую директорию в путь для импорта myLib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import myLib


def circle_points(cx, cy, radius, num_points):
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    return points


def create_circle(radius, lat, lon):
    wgs2utm = Transformer.from_crs(myLib.WGS84, myLib.UTM, always_xy=True)
    utm2wgs = Transformer.from_crs(myLib.UTM, myLib.WGS84, always_xy=True)

    centr = wgs2utm.transform(lon, lat)
    points_utm = circle_points(centr[0], centr[1], radius, 1000)
    points_wgs = [utm2wgs.transform(x, y) for (x, y) in points_utm]

    return Polygon(points_wgs)


def save_geojson(geometry, filename="contour.geojson"):
    """Сохраняет геометрию в файл GeoJSON."""
    feature = geojson.Feature(geometry=mapping(geometry))
    feature_collection = geojson.FeatureCollection([feature])
    with open(filename, "w") as f:
        geojson.dump(feature_collection, f)
    print(f"Контур сохранён в {filename}.")


def plot_circle(geometry, lat, lon, radius):
    """Показывает график кругового контура."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    if hasattr(geometry, 'exterior'):
        xs, ys = geometry.exterior.xy
        ax.plot(xs, ys, 'b-', linewidth=2, label='Контур')
        ax.fill(xs, ys, alpha=0.1, color='blue')

    ax.scatter(lon, lat, color='purple', s=60, marker='+',
               linewidths=2, label='Центр', zorder=7)

    ax.set_xlabel('Долгота')
    ax.set_ylabel('Широта')
    ax.set_title(f'Круговой контур (радиус {radius} м)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Генерация кругового контура"
    )
    parser.add_argument(
        '--lat', type=float, default=43.348397,
        help='Широта центра (по умолчанию: 43.348397)'
    )
    parser.add_argument(
        '--lon', type=float, default=42.454421,
        help='Долгота центра (по умолчанию: 42.454421)'
    )
    parser.add_argument(
        '--radius', type=float, default=50000,
        help='Радиус в метрах (по умолчанию: 50000)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='contour.geojson',
        help='Выходной GeoJSON файл (по умолчанию: contour.geojson)'
    )
    parser.add_argument(
        '--plot', '-p', action='store_true',
        help='Показать график контура'
    )

    args = parser.parse_args()

    print(f"Центр: lat={args.lat}, lon={args.lon}")
    print(f"Радиус: {args.radius} м")

    circle = create_circle(args.radius, args.lat, args.lon)
    save_geojson(circle, args.output)

    if args.plot:
        plot_circle(circle, args.lon, args.lat, args.radius)

    print("\nГотово!")


if __name__ == "__main__":
    main()
