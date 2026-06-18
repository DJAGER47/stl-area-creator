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
        h = myLib.elevation_data.get_elevation(lat, lon)
        if h is None or h < 0:
            print(f"Не удалось получить высоту для точки ({lat:.6f}, {lon:.6f})")
            h = 0
        h += ADD_H
        
        vertices.append([x, y, h])
    
    # Боковые грани
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([i, next_i, i + segments])
        faces.append([next_i, next_i + segments, i + segments])
    
    # Нижняя крышка (нормаль смотрит вниз: порядок CW при взгляде сверху)
    for i in range(1, segments - 1):
        faces.append([0, i + 1, i])
    
    # Верхняя крышка (триангуляция для неровного верха)
    for i in range(1, segments - 1):
        faces.append([segments, segments + i, segments + i + 1])
    
    return np.array(vertices), np.array(faces)


def create_thick_line(line_points, thickness, segments_per_circle=16):
    """Создает manifold-меш стены вдоль маршрута с miter-стыками.

    Для каждой точки трека создаются 4 вершины (индекс i*4):
      +0: left_bottom, +1: right_bottom, +2: left_top, +3: right_top.
    Нижний слой z=0, верхний — высота рельефа из line_points[i][2].
    Стыки между сегментами закрываются через miter-смещения (общие вершины),
    что гарантирует manifold-меш без дублирующихся граней.
    """
    n = len(line_points)
    if n < 2:
        return None, None

    half = thickness / 2
    pts = [np.array(p, dtype=float) for p in line_points]

    # Нормированные направления каждого сегмента
    dirs = []
    for i in range(n - 1):
        d = pts[i + 1][:2] - pts[i][:2]
        length = np.linalg.norm(d)
        if length < 1e-9:
            dirs.append(dirs[-1] if dirs else np.array([1.0, 0.0]))
        else:
            dirs.append(d / length)

    # Перпендикуляры сегментов (повёрнуты влево от направления)
    perps = [np.array([-d[1], d[0]]) for d in dirs]

    # Miter-смещения для каждой точки трека (XY-вектор длиной ~half)
    miter_offsets = []
    for i in range(n):
        if i == 0:
            miter_offsets.append(perps[0] * half)
        elif i == n - 1:
            miter_offsets.append(perps[-1] * half)
        else:
            avg = perps[i - 1] + perps[i]
            norm_avg = np.linalg.norm(avg)
            if norm_avg < 1e-9:
                # Разворот на 180° — используем текущий перпендикуляр
                miter_offsets.append(perps[i] * half)
            else:
                avg = avg / norm_avg
                cos_a = np.dot(avg, perps[i - 1])
                # Ограничиваем длину miter (не более 4×half на острых углах)
                scale = half / max(abs(cos_a), 0.25)
                miter_offsets.append(avg * scale)

    # Строим вершины: 4 вершины на точку трека
    vertices = []
    for i, p in enumerate(pts):
        off = miter_offsets[i]
        z = p[2]
        vertices.append([p[0] + off[0], p[1] + off[1], 0.0])  # left_bottom  (+0)
        vertices.append([p[0] - off[0], p[1] - off[1], 0.0])  # right_bottom (+1)
        vertices.append([p[0] + off[0], p[1] + off[1], z])    # left_top     (+2)
        vertices.append([p[0] - off[0], p[1] - off[1], z])    # right_top    (+3)

    # Строим грани с корректными нормалями (правило правой руки):
    #   A=left_bottom[i],   B=right_bottom[i],  C=left_top[i],   D=right_top[i]
    #   E=left_bottom[i+1], F=right_bottom[i+1],G=left_top[i+1], H=right_top[i+1]
    faces = []

    for i in range(n - 1):
        A = i * 4;        B = i * 4 + 1;    C = i * 4 + 2;    D = i * 4 + 3
        E = (i+1) * 4;   F = (i+1) * 4 + 1; G = (i+1) * 4 + 2; H = (i+1) * 4 + 3

        # Нижняя грань (нормаль -Z)
        faces.append([A, E, F])
        faces.append([A, F, B])

        # Верхняя грань (нормаль +Z)
        faces.append([C, H, G])
        faces.append([C, D, H])

        # Левая боковая грань (нормаль наружу влево)
        faces.append([A, C, G])
        faces.append([A, G, E])

        # Правая боковая грань (нормаль наружу вправо)
        faces.append([B, H, D])
        faces.append([B, F, H])

    # Передний торец (нормаль против направления движения)
    A, B, C, D = 0, 1, 2, 3
    faces.append([A, B, D])
    faces.append([A, D, C])

    # Задний торец (нормаль по направлению движения)
    last = (n - 1) * 4
    E, F, G, H = last, last + 1, last + 2, last + 3
    faces.append([E, G, H])
    faces.append([E, H, F])

    return np.array(vertices, dtype=float), np.array(faces, dtype=int)


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
        h = myLib.elevation_data.get_elevation(lat, lon)
        if h is None or h < 0:
            h = 0
        h += ADD_H

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
