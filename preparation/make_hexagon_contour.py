#!/usr/bin/env python3
"""
Скрипт для генерации гексагона.

Режимы работы:
  1. --track <gpx_file>  — рассчитать гексагон по GPX треку (охватывает все точки)
  2. --center-radius <lat> <lon> <radius_meters> — гексагон по центру и радиусу

Общие параметры:
  --rotation <deg>  — поворот гексагона в градусах (по умолчанию 0)
  --scale <x>       — коэффициент масштабирования размера гексагона (по умолчанию 1.0)
  --output <file>   — путь к выходному GeoJSON файлу
  --plot            — показать график с треком и гексагоном
"""

import xml.etree.ElementTree as ET
import math
import json
import sys
import os
import argparse
from pathlib import Path
from pyproj import Transformer, CRS
from _find_gpx_center import find_gpx_center

# Добавляем корневую директорию в путь для импорта myLib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import myLib


def latlon_to_utm(lat, lon):
    """
    Конвертирует географические координаты (myLib.WGS84) в myLib.UTM.

    Args:
        lat: Широта в градусах
        lon: Долгота в градусах

    Returns:
        Кортеж (x, y) в метрах myLib.UTM
    """
    transformer = Transformer.from_crs(myLib.WGS84, myLib.UTM, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def utm_to_latlon(x, y):
    """
    Конвертирует myLib.UTM координаты в географические (myLib.WGS84).

    Args:
        x: Координата X в метрах myLib.UTM
        y: Координата Y в метрах myLib.UTM

    Returns:
        Кортеж (lon, lat) в градусах
    """
    transformer = Transformer.from_crs(myLib.UTM, myLib.WGS84, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat


def get_gpx_points(gpx_file):
    """
    Извлекает все точки из GPX файла.

    Args:
        gpx_file: Путь к GPX файлу

    Returns:
        Список кортежей [(lat, lon), ...] или None, если точек нет
    """
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}

    tree = ET.parse(gpx_file)
    root = tree.getroot()

    track_points = root.findall('.//gpx:trkpt', ns)

    if not track_points:
        return None

    points = []
    for point in track_points:
        lat = float(point.get('lat'))
        lon = float(point.get('lon'))
        points.append((lat, lon))

    return points


def get_gpx_bounds(gpx_file):
    """
    Находит границы (min/max) всех точек GPX файла.

    Args:
        gpx_file: Путь к GPX файлу

    Returns:
        Словарь с границами: min_lat, max_lat, min_lon, max_lon
    """
    points = get_gpx_points(gpx_file)
    if not points:
        print("Ошибка: В GPX файле не найдены точки трека (trkpt)")
        return None

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    return {
        'min_lat': min(lats),
        'max_lat': max(lats),
        'min_lon': min(lons),
        'max_lon': max(lons)
    }


def calculate_hexagon_radius(center_lat, center_lon, bounds):
    """
    Вычисляет радиус гексагона в метрах, чтобы все точки GPX были внутри.

    Args:
        center_lat: Широта центра
        center_lon: Долгота центра
        bounds: Словарь с границами точек

    Returns:
        Радиус гексагона в метрах
    """
    # Конвертируем центр в myLib.UTM
    center_x, center_y = latlon_to_utm(center_lat, center_lon)

    # Конвертируем границы в метры относительно центра
    corners = [
        (bounds['min_lat'], bounds['min_lon']),
        (bounds['min_lat'], bounds['max_lon']),
        (bounds['max_lat'], bounds['min_lon']),
        (bounds['max_lat'], bounds['max_lon'])
    ]

    max_distance = 0
    for lat, lon in corners:
        x, y = latlon_to_utm(lat, lon)
        distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = max(max_distance, distance)

    # Для гексагона нужно учесть, что расстояние до вершин больше, чем до граней
    # Умножаем на коэффициент для гарантии, что все точки будут внутри
    # Коэффициент 1.2 обеспечивает запас
    radius = max_distance * 1.2

    return radius


def generate_hexagon_vertices(center_lat, center_lon, radius_meters, rotation_deg=0):
    """
    Генерирует вершины гексагона.

    Args:
        center_lat: Широта центра
        center_lon: Долгота центра
        radius_meters: Радиус гексагона в метрах
        rotation_deg: Угол поворота гексагона в градусах (по умолчанию 0)

    Returns:
        Список координат вершин [(lon, lat), ...]
    """
    vertices = []

    # Конвертируем центр в myLib.UTM
    center_x, center_y = latlon_to_utm(center_lat, center_lon)

    # Начальный угол с учётом поворота
    # Гексагон имеет 6 вершин, угол между ними 60 градусов (π/3 радиан)
    for i in range(6):
        angle_deg = 60 * i + rotation_deg
        angle_rad = math.radians(angle_deg)

        # Вычисляем координаты вершины в метрах myLib.UTM
        x = center_x + radius_meters * math.cos(angle_rad)
        y = center_y + radius_meters * math.sin(angle_rad)

        # Конвертируем обратно в географические координаты
        lon, lat = utm_to_latlon(x, y)

        vertices.append([lon, lat])

    # Замыкаем полигон (первая точка в конце)
    vertices.append(vertices[0])

    return vertices


def create_geojson_polygon(vertices):
    """
    Создает GeoJSON объект с полигоном.

    Args:
        vertices: Список координат вершин

    Returns:
        GeoJSON словарь
    """
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Hexagon Contour",
                    "description": "Hexagon covering all GPX track points"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [vertices]
                }
            }
        ]
    }

    return geojson


def save_geojson(geojson, output_file):
    """
    Сохраняет GeoJSON в файл.

    Args:
        geojson: GeoJSON словарь
        output_file: Путь к выходному файлу
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)

    print(f"Гексагон сохранен в файл: {output_file}")


def plot_hexagon_with_track(vertices, gpx_file=None, center_lat=None, center_lon=None):
    """
    Строит график с треком и гексагоном для визуальной оценки.

    Args:
        vertices: Список вершин гексагона [(lon, lat), ...]
        gpx_file: Путь к GPX файлу (опционально, для отрисовки трека)
        center_lat: Широта центра (опционально)
        center_lon: Долгота центра (опционально)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Рисуем гексагон
    hex_lons = [v[0] for v in vertices]
    hex_lats = [v[1] for v in vertices]
    ax.plot(hex_lons, hex_lats, 'b-', linewidth=2, label='Гексагон')
    ax.fill(hex_lons, hex_lats, alpha=0.1, color='blue')

    # Рисуем вершины гексагона
    ax.scatter(hex_lons[:-1], hex_lats[:-1], color='blue', s=30, zorder=5)
    for i, (lon, lat) in enumerate(zip(hex_lons[:-1], hex_lats[:-1])):
        ax.annotate(str(i + 1), (lon, lat), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color='blue')

    # Рисуем трек из GPX
    if gpx_file:
        points = get_gpx_points(gpx_file)
        if points:
            track_lats = [p[0] for p in points]
            track_lons = [p[1] for p in points]
            ax.plot(track_lons, track_lats, 'r-', linewidth=1.5, alpha=0.8, label='Трек')
            ax.scatter(track_lons[0], track_lats[0], color='green', s=40, marker='o',
                       label='Старт', zorder=6)
            ax.scatter(track_lons[-1], track_lats[-1], color='red', s=40, marker='x',
                       label='Финиш', zorder=6)

    # Рисуем центр
    if center_lat is not None and center_lon is not None:
        ax.scatter(center_lon, center_lat, color='purple', s=60, marker='+',
                   linewidths=2, label='Центр', zorder=7)

    ax.set_xlabel('Долгота')
    ax.set_ylabel('Широта')
    ax.set_title('Гексагон и трек')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.show()


def run_track_mode(args):
    """
    Режим 1: расчёт гексагона по GPX треку.
    """
    gpx_file = args.track
    output_file = args.output or "contour.geojson"
    rotation_deg = args.rotation
    scale = args.scale

    # Проверяем существование GPX файла
    if not os.path.exists(gpx_file):
        print(f"Ошибка: Файл '{gpx_file}' не найден")
        sys.exit(1)

    print(f"Режим: track")
    print(f"Обработка файла: {gpx_file}")
    print(f"Поворот: {rotation_deg}°")
    print(f"Масштаб: {scale}")

    # Находим центр всех точек
    center = find_gpx_center(gpx_file)
    if not center:
        print("Ошибка: Не удалось найти центр GPX трека")
        sys.exit(1)

    center_lat, center_lon = center

    # Получаем границы всех точек
    bounds = get_gpx_bounds(gpx_file)
    if not bounds:
        print("Ошибка: Не удалось получить границы GPX трека")
        sys.exit(1)

    print(f"\nГраницы трека:")
    print(f"  Широта: {bounds['min_lat']:.6f} - {bounds['max_lat']:.6f}")
    print(f"  Долгота: {bounds['min_lon']:.6f} - {bounds['max_lon']:.6f}")

    # Вычисляем радиус гексагона в метрах
    radius_meters = calculate_hexagon_radius(center_lat, center_lon, bounds)
    radius_meters = radius_meters * scale
    print(f"Радиус гексагона: {radius_meters:.2f} метров (с учётом масштаба {scale})")

    # Генерируем вершины гексагона с поворотом
    vertices = generate_hexagon_vertices(center_lat, center_lon, radius_meters, rotation_deg)

    print(f"\nВершины гексагона:")
    for i, (lon, lat) in enumerate(vertices[:-1]):  # Без последней (дублирующей) точки
        print(f"  Вершина {i+1}: lat={lat:.6f}, lon={lon:.6f}")

    # Создаем GeoJSON
    geojson = create_geojson_polygon(vertices)

    # Сохраняем в файл
    save_geojson(geojson, output_file)

    # Показываем график, если запрошено
    if args.plot:
        plot_hexagon_with_track(
            vertices=vertices,
            gpx_file=gpx_file,
            center_lat=center_lat,
            center_lon=center_lon
        )

    print("\nГотово!")


def run_center_radius_mode(args):
    """
    Режим 2: гексагон по центру (lat, lon) и радиусу в метрах.
    """
    center_lat = args.lat
    center_lon = args.lon
    radius_meters = args.radius * args.scale
    output_file = args.output or "contour.geojson"
    rotation_deg = args.rotation
    scale = args.scale

    print(f"Режим: center-radius")
    print(f"Центр: lat={center_lat:.6f}, lon={center_lon:.6f}")
    print(f"Радиус: {radius_meters:.2f} метров (с учётом масштаба {scale})")
    print(f"Поворот: {rotation_deg}°")
    print(f"Масштаб: {scale}")

    # Генерируем вершины гексагона с поворотом
    vertices = generate_hexagon_vertices(center_lat, center_lon, radius_meters, rotation_deg)

    print(f"\nВершины гексагона:")
    for i, (lon, lat) in enumerate(vertices[:-1]):  # Без последней (дублирующей) точки
        print(f"  Вершина {i+1}: lat={lat:.6f}, lon={lon:.6f}")

    # Создаем GeoJSON
    geojson = create_geojson_polygon(vertices)

    # Сохраняем в файл
    save_geojson(geojson, output_file)

    # Показываем график, если запрошено
    if args.plot:
        plot_hexagon_with_track(
            vertices=vertices,
            center_lat=center_lat,
            center_lon=center_lon
        )

    print("\nГотово!")


def main():
    parser = argparse.ArgumentParser(
        description="Генерация гексагонального контура"
    )

    # Создаём взаимоисключающие группы для режимов
    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--track",
        type=str,
        metavar="GPX_FILE",
        help="Режим 1: путь к GPX файлу для расчёта гексагона по треку"
    )
    mode_group.add_argument(
        "--center-radius",
        type=float,
        nargs=3,
        metavar=("LAT", "LON", "RADIUS_M"),
        help="Режим 2: центр (lat, lon) и радиус в метрах"
    )

    # Общие параметры
    parser.add_argument(
        "--rotation",
        type=float,
        default=0,
        help="Поворот гексагона в градусах (по умолчанию: 0)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Коэффициент масштабирования размера гексагона (по умолчанию: 1.0). "
             "Например, 1.1 увеличит размер на 10%%, 0.9 — уменьшит на 10%%"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Выходной GeoJSON файл (по умолчанию: contour.geojson)"
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Показать график с треком и гексагоном для визуальной оценки"
    )

    args = parser.parse_args()

    # Определяем режим
    if args.track is not None:
        run_track_mode(args)
    elif args.center_radius is not None:
        args.lat, args.lon, args.radius = args.center_radius
        run_center_radius_mode(args)
    else:
        # Если режим не указан — показываем help
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
