#!/usr/bin/env python3
"""
Скрипт для генерации гексагона, который охватывает все точки GPX файла.
Гексагон центрируется в центре всех точек GPX и имеет размер, достаточный
для включения всех точек.
"""

import xml.etree.ElementTree as ET
import math
import json
import sys
import os
from pathlib import Path
from pyproj import Transformer, CRS
from find_center import find_gpx_center

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


def get_gpx_bounds(gpx_file):
    """
    Находит границы (min/max) всех точек GPX файла.
    
    Args:
        gpx_file: Путь к GPX файлу
        
    Returns:
        Словарь с границами: min_lat, max_lat, min_lon, max_lon
    """
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    
    tree = ET.parse(gpx_file)
    root = tree.getroot()
    
    track_points = root.findall('.//gpx:trkpt', ns)
    
    if not track_points:
        print("Ошибка: В GPX файле не найдены точки трека (trkpt)")
        return None
    
    lats = []
    lons = []
    
    for point in track_points:
        lat = float(point.get('lat'))
        lon = float(point.get('lon'))
        lats.append(lat)
        lons.append(lon)
    
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


def generate_hexagon_vertices(center_lat, center_lon, radius_meters):
    """
    Генерирует вершины гексагона.
    
    Args:
        center_lat: Широта центра
        center_lon: Долгота центра
        radius_meters: Радиус гексагона в метрах
        
    Returns:
        Список координат вершин [(lon, lat), ...]
    """
    vertices = []
    
    # Конвертируем центр в myLib.UTM
    center_x, center_y = latlon_to_utm(center_lat, center_lon)
    
    # Гексагон имеет 6 вершин, угол между ними 60 градусов (π/3 радиан)
    for i in range(6):
        angle_deg = 60 * i
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


def main():
    # Проверяем аргументы командной строки
    if len(sys.argv) < 2:
        print("Использование: python generate_hexagon.py <путь_к_gpx_файлу> [выходной_файл]")
        print("Пример: python generate_hexagon.py track.gpx contour.geojson")
        sys.exit(1)
    
    gpx_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "contour.geojson"
    
    # Проверяем существование GPX файла
    try:
        with open(gpx_file, 'r', encoding='utf-8') as f:
            pass
    except FileNotFoundError:
        print(f"Ошибка: Файл '{gpx_file}' не найден")
        sys.exit(1)
    
    print(f"Обработка файла: {gpx_file}")
    
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
    print(f"Радиус гексагона: {radius_meters:.2f} метров")
    
    # Генерируем вершины гексагона
    vertices = generate_hexagon_vertices(center_lat, center_lon, radius_meters)
    
    print(f"\nВершины гексагона:")
    for i, (lon, lat) in enumerate(vertices[:-1]):  # Без последней (дублирующей) точки
        print(f"  Вершина {i+1}: lat={lat:.6f}, lon={lon:.6f}")
    
    # Создаем GeoJSON
    geojson = create_geojson_polygon(vertices)
    
    # Сохраняем в файл
    save_geojson(geojson, output_file)
    
    print("\nГотово!")


if __name__ == "__main__":
    main()
