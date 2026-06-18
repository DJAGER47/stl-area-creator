#!/usr/bin/env python3
"""
Скрипт для генерации STL модели маршрута из GPX файла.
- wpt точки отображаются как круглые столбики
- линии маршрута имеют толщину 100 метров
"""

import argparse
import math
import sys
import tracemalloc
import xml.etree.ElementTree as ET

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point

import myLib
from stl import mesh

logger = myLib.logger

ADD_H = 150


def parse_gpx(gpx_file):
    """Парсит GPX файл и извлекает wpt точки и трек"""
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    
    tree = ET.parse(gpx_file)
    root = tree.getroot()
    
    # Извлекаем wpt точки
    wpt_points = []
    for wpt in root.findall('.//gpx:wpt', ns):
        lat = float(wpt.get('lat'))
        lon = float(wpt.get('lon'))
        name = wpt.find('gpx:name', ns)
        name_text = name.text if name is not None else ""
        wpt_points.append({'lat': lat, 'lon': lon, 'name': name_text})
    
    # Извлекаем точки трека
    track_points = []
    for trkpt in root.findall('.//gpx:trkpt', ns):
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        track_points.append((lat, lon))
    
    return wpt_points, track_points


def create_cylinder(center, radius, segments=32):
    """Создает цилиндр (круглый столбик) с высотой по рельефу для каждой точки"""
    vertices = []
    faces = []
    
    # Получаем координаты центра в WGS84 для запроса высоты
    transformer_to_wgs84 = Transformer.from_crs(myLib.UTM, myLib.WGS84, always_xy=True)
    
    # Нижний круг (на высоте 0)
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append([x, y, 0])
    
    # Верхний круг (с высотой по рельефу для каждой точки)
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        
        # Получаем высоту рельефа для этой точки
        lon, lat = transformer_to_wgs84.transform(x, y)
        h = myLib.elevation_data.get_elevation(lat, lon) + ADD_H
        if h is None:
            print(f"Не удалось получить высоту для точки ({lat:.6f}, {lon:.6f})")
            h = 0
        
        vertices.append([x, y, h])
    
    # Боковые грани
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([i, next_i, i + segments])
        faces.append([next_i, next_i + segments, i + segments])
    
    # Нижняя крышка
    for i in range(1, segments - 1):
        faces.append([0, i, i + 1])
    
    # Верхняя крышка (триангуляция для неровного верха)
    for i in range(1, segments - 1):
        faces.append([segments, segments + i, segments + i + 1])
    
    return np.array(vertices), np.array(faces)


def create_thick_line(line_points, thickness, segments_per_circle=16):
    """Создает стену вдоль маршрута: нижний слой на высоте 0, верхний на высоте рельефа"""
    if len(line_points) < 2:
        return None, None
    
    vertices = []
    faces = []
    
    half_thickness = thickness / 2
    
    # Создаем сегменты стены между каждой парой точек
    for i in range(len(line_points) - 1):
        p1 = np.array(line_points[i])
        p2 = np.array(line_points[i + 1])
        
        # Используем только X и Y координаты для направления
        p1_xy = np.array([p1[0], p1[1]])
        p2_xy = np.array([p2[0], p2[1]])
        
        # Вектор направления (только по XY)
        direction = p2_xy - p1_xy
        length = np.linalg.norm(direction)
        if length == 0:
            continue
        direction = direction / length
        
        # Перпендикулярный вектор (для смещения влево-вправо)
        perpendicular = np.array([-direction[1], direction[0]])
        
        # Создаем 4 точки для каждого сегмента (прямоугольник)
        # p1_left_bottom, p1_right_bottom, p2_left_bottom, p2_right_bottom
        # p1_left_top, p1_right_top, p2_left_top, p2_right_top
        
        # Нижний слой (высота 0)
        p1_left_bottom = np.array([p1[0] + perpendicular[0] * half_thickness,
                                    p1[1] + perpendicular[1] * half_thickness, 0])
        p1_right_bottom = np.array([p1[0] - perpendicular[0] * half_thickness,
                                     p1[1] - perpendicular[1] * half_thickness, 0])
        p2_left_bottom = np.array([p2[0] + perpendicular[0] * half_thickness,
                                    p2[1] + perpendicular[1] * half_thickness, 0])
        p2_right_bottom = np.array([p2[0] - perpendicular[0] * half_thickness,
                                     p2[1] - perpendicular[1] * half_thickness, 0])
        
        # Верхний слой (высота рельефа)
        p1_left_top = np.array([p1[0] + perpendicular[0] * half_thickness,
                                 p1[1] + perpendicular[1] * half_thickness, p1[2]])
        p1_right_top = np.array([p1[0] - perpendicular[0] * half_thickness,
                                  p1[1] - perpendicular[1] * half_thickness, p1[2]])
        p2_left_top = np.array([p2[0] + perpendicular[0] * half_thickness,
                                 p2[1] + perpendicular[1] * half_thickness, p2[2]])
        p2_right_top = np.array([p2[0] - perpendicular[0] * half_thickness,
                                  p2[1] - perpendicular[1] * half_thickness, p2[2]])
        
        # Добавляем вершины
        base_offset = len(vertices)
        vertices.extend([p1_left_bottom, p1_right_bottom, p2_left_bottom, p2_right_bottom,
                        p1_left_top, p1_right_top, p2_left_top, p2_right_top])
        
        # Создаем грани для сегмента
        # Нижняя грань
        faces.append([base_offset, base_offset + 1, base_offset + 2])
        faces.append([base_offset + 1, base_offset + 3, base_offset + 2])
        
        # Верхняя грань
        faces.append([base_offset + 4, base_offset + 6, base_offset + 5])
        faces.append([base_offset + 5, base_offset + 6, base_offset + 7])
        
        # Боковые грани
        faces.append([base_offset, base_offset + 2, base_offset + 4])
        faces.append([base_offset + 2, base_offset + 6, base_offset + 4])
        
        faces.append([base_offset + 1, base_offset + 5, base_offset + 3])
        faces.append([base_offset + 3, base_offset + 5, base_offset + 7])
        
        faces.append([base_offset, base_offset + 4, base_offset + 1])
        faces.append([base_offset + 1, base_offset + 4, base_offset + 5])
        
        faces.append([base_offset + 2, base_offset + 3, base_offset + 6])
        faces.append([base_offset + 3, base_offset + 7, base_offset + 6])
        
        # Соединяем с предыдущим сегментом
        if i > 0:
            # Предыдущие вершины
            prev_p1_left_bottom = base_offset - 8
            prev_p1_right_bottom = base_offset - 7
            prev_p2_left_bottom = base_offset - 6
            prev_p2_right_bottom = base_offset - 5
            prev_p1_left_top = base_offset - 4
            prev_p1_right_top = base_offset - 3
            prev_p2_left_top = base_offset - 2
            prev_p2_right_top = base_offset - 1
            
            # Соединяем грани между сегментами
            # Нижние соединения
            faces.append([prev_p2_left_bottom, base_offset, prev_p2_right_bottom])
            faces.append([prev_p2_right_bottom, base_offset, base_offset + 1])
            
            # Верхние соединения
            faces.append([prev_p2_left_top, base_offset + 4, prev_p2_right_top])
            faces.append([prev_p2_right_top, base_offset + 4, base_offset + 5])
            
            # Боковые соединения
            faces.append([prev_p2_left_bottom, prev_p2_left_top, base_offset])
            faces.append([base_offset, prev_p2_left_top, base_offset + 4])
            
            faces.append([prev_p2_right_bottom, base_offset + 1, prev_p2_right_top])
            faces.append([base_offset + 1, base_offset + 5, prev_p2_right_top])
    
    return np.array(vertices), np.array(faces)


def create_wpt_cylinders(wpt_points, radius, segments=32):
    """Создает STL меши для всех wpt точек с высотой по рельефу"""
    transformer = Transformer.from_crs(myLib.WGS84, myLib.UTM, always_xy=True)
    
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    for wpt in wpt_points:
        # Преобразуем в UTM
        x, y = transformer.transform(wpt['lon'], wpt['lat'])
        center = (x, y)
        
        # Создаем цилиндр с высотой рельефа
        vertices, faces = create_cylinder(center, radius, segments)
        
        # Добавляем вершины и грани
        all_vertices.extend(vertices)
        all_faces.extend(faces + vertex_offset)
        vertex_offset += len(vertices)
    
    return np.array(all_vertices), np.array(all_faces)


def create_track_mesh(track_points, thickness, segments_per_circle=16):
    """Создает STL меш для трека с высотой по рельефу"""
    transformer = Transformer.from_crs(myLib.WGS84, myLib.UTM, always_xy=True)
    
    # Преобразуем точки в UTM и получаем высоты
    utm_points = []
    for lat, lon in track_points:
        x, y = transformer.transform(lon, lat)
        h = myLib.elevation_data.get_elevation(lat, lon) + ADD_H
        if h is None or h < 0:
            h = 0
        utm_points.append((x, y, h))
    
    # Создаем толстую линию (высота определяется рельефом)
    vertices, faces = create_thick_line(utm_points, thickness, segments_per_circle)
    
    return vertices, faces




def generate_route_stl(gpx_file, output_file, wpt_radius, track_thickness):
    """Генерирует STL модель маршрута из GPX файла"""
    with myLib.Timer("GPX parsing"):
        wpt_points, track_points = parse_gpx(gpx_file)
        logger.info(f"Found {len(wpt_points)} waypoints and {len(track_points)} track points")
    
    with myLib.Timer("WPT cylinders creation"):
        wpt_vertices, wpt_faces = create_wpt_cylinders(wpt_points, wpt_radius)
        logger.info(f"WPT vertices: {len(wpt_vertices)}, faces: {len(wpt_faces)}")
    
    with myLib.Timer("Track mesh creation"):
        track_vertices, track_faces = create_track_mesh(track_points, track_thickness)
        if track_vertices is not None:
            logger.info(f"Track vertices: {len(track_vertices)}, faces: {len(track_faces)}")
        else:
            logger.info("Track mesh is empty")
    
    with myLib.Timer("STL mesh creation"):
        # Масштабируем X,Y координаты для соответствия с поверхностью
        if wpt_vertices is not None and len(wpt_vertices) > 0:
            wpt_vertices = myLib.scale_elevation(wpt_vertices)
        
        if track_vertices is not None and len(track_vertices) > 0:
            track_vertices = myLib.scale_elevation(track_vertices)
        
        wpt_mesh = myLib.vertices_faces_to_mesh(wpt_vertices, wpt_faces)
        track_mesh = myLib.vertices_faces_to_mesh(track_vertices, track_faces)
        
        # Объединяем меши
        if wpt_mesh is not None and track_mesh is not None:
            combined_vectors = np.concatenate([wpt_mesh.vectors, track_mesh.vectors])
            combined_mesh = mesh.Mesh(np.zeros(combined_vectors.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(combined_vectors):
                combined_mesh.vectors[i] = f
            combined_mesh.update_normals()
        elif wpt_mesh is not None:
            combined_mesh = wpt_mesh
        elif track_mesh is not None:
            combined_mesh = track_mesh
        else:
            logger.error("No mesh data to save")
            return
    
    with myLib.Timer("STL saving"):
        combined_mesh.save(output_file)
        logger.info(f"Saved STL to {output_file}")


def main():
    tracemalloc.start()
    
    parser = argparse.ArgumentParser(
        description="Генератор STL модели маршрута из GPX файла",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input',
                        type=str,
                        default="Планируемый путь.gpx",
                        help="Путь к GPX файлу")
    parser.add_argument('-o', '--output',
                        type=str,
                        default="stl/route.stl",
                        help="Путь для сохранения STL файла")
    parser.add_argument('--wpt-radius',
                        type=float,
                        default=400,
                        help="Радиус столбиков для wpt точек (метры)")
    parser.add_argument('--track-thickness',
                        type=float,
                        default=300,
                        help="Толщина линии маршрута (метры)")
    args = parser.parse_args()
    
    with myLib.Timer("Total generation"):
        generate_route_stl(
            args.input,
            args.output,
            args.wpt_radius,
            args.track_thickness
        )
    
    size, peak = tracemalloc.get_traced_memory()
    logger.info(f"{'Memory usage':<50} {size/1024:>7.1f} KB")
    logger.info(f"{'Peak memory usage':<50} {peak/1024:>7.1f} KB")


if __name__ == "__main__":
    main()
