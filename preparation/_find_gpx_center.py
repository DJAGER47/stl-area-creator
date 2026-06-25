#!/usr/bin/env python3
"""
Скрипт для нахождения центра всех точек из GPX трека.
Вычисляет среднее арифметическое всех координат широты и долготы.
"""

import xml.etree.ElementTree as ET
import sys


def find_gpx_center(gpx_file):
    """
    Находит центр всех точек из GPX файла.
    
    Args:
        gpx_file: Путь к GPX файлу
        
    Returns:
        Кортеж (lat, lon) с координатами центра
    """
    # Пространства имен GPX
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    
    # Парсим XML файл
    tree = ET.parse(gpx_file)
    root = tree.getroot()
    
    # Собираем все точки трека (trkpt)
    track_points = root.findall('.//gpx:trkpt', ns)
    
    if not track_points:
        print("Ошибка: В GPX файле не найдены точки трека (trkpt)")
        return None
    
    # Суммируем все координаты
    lat_sum = 0.0
    lon_sum = 0.0
    count = 0
    
    for point in track_points:
        lat = float(point.get('lat'))
        lon = float(point.get('lon'))
        lat_sum += lat
        lon_sum += lon
        count += 1
    
    # Вычисляем среднее значение (центр)
    center_lat = lat_sum / count
    center_lon = lon_sum / count
    
    print(f"Всего точек: {count}")
    print(f"Центр трека:")
    print(f"  Широта (lat): {center_lat:.6f}")
    print(f"  Долгота (lon): {center_lon:.6f}")
    
    return center_lat, center_lon


def main():
    # Проверяем, передан ли путь к GPX файлу как аргумент
    if len(sys.argv) < 2:
        print("Использование: python find_center.py <путь_к_gpx_файлу>")
        sys.exit(1)
    
    gpx_file = sys.argv[1]
    
    # Проверяем существование файла
    try:
        with open(gpx_file, 'r', encoding='utf-8') as f:
            pass
    except FileNotFoundError:
        print(f"Ошибка: Файл '{gpx_file}' не найден")
        sys.exit(1)
    
    # Находим центр
    center = find_gpx_center(gpx_file)
    
    if center:
        print(f"\nКоординаты центра: {center[0]:.6f}, {center[1]:.6f}")


if __name__ == "__main__":
    main()
