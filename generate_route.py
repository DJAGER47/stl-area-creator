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
from shapely.geometry import LineString, MultiPolygon, Point
from shapely.ops import unary_union
from scipy.spatial import Delaunay

import myLib
from stl import mesh

logger = myLib.logger

ADD_H = 0


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


def create_wpt_cylinders(wpt_points, radius, buffer_resolution=32):
    """Создает STL меши для wpt-точек через Shapely Point.buffer() + Delaunay.

    Аналогично create_track_mesh(): каждый столбик — это круговой полигон,
    триангулированный Delaunay с нижним (z=0) и верхним (z=рельеф) слоями.
    """
    transformer     = Transformer.from_crs(myLib.WGS84, myLib.UTM, always_xy=True)
    transformer_inv = Transformer.from_crs(myLib.UTM, myLib.WGS84, always_xy=True)

    all_vertices = []
    all_faces    = []
    vertex_offset = 0

    for wpt in wpt_points:
        # 1) Центр в UTM
        cx, cy = transformer.transform(wpt['lon'], wpt['lat'])

        # 2) Круговой полигон через Shapely
        footprint = Point(cx, cy).buffer(radius, resolution=buffer_resolution)

        # 3) Точки контура (CCW для exterior)
        boundary_coords = np.array(footprint.exterior.coords[:-1])
        boundary_n      = len(boundary_coords)

        # 4) Внутренняя сетка
        step         = radius / 3
        interior_pts = []
        for gx in np.arange(cx - radius + step / 2, cx + radius, step):
            for gy in np.arange(cy - radius + step / 2, cy + radius, step):
                if footprint.contains(Point(gx, gy)):
                    interior_pts.append([gx, gy])

        if interior_pts:
            all_xy = np.vstack([boundary_coords, np.array(interior_pts)])
        else:
            all_xy = boundary_coords.copy()

        total_n = len(all_xy)

        # 5) Delaunay; отбрасываем треугольники вне footprint
        tri             = Delaunay(all_xy)
        valid_simplices = []
        for simplex in tri.simplices:
            mx, my = all_xy[simplex].mean(axis=0)
            if footprint.contains(Point(mx, my)):
                valid_simplices.append(simplex)

        # 6) Высоты
        elevations = np.zeros(total_n)
        for idx, (vx, vy) in enumerate(all_xy):
            lon_v, lat_v = transformer_inv.transform(vx, vy)
            h = myLib.elevation_data.get_elevation(lat_v, lon_v)
            if h is None:
                h = 0
            h += ADD_H
            if h < 0:
                h = 0
            elevations[idx] = h

        # 7) Вершины: нижний (z=0) + верхний (z=elevation)
        verts_bot = np.column_stack([all_xy, np.zeros(total_n)])
        verts_top = np.column_stack([all_xy, elevations])
        verts     = np.vstack([verts_bot, verts_top])

        faces = []

        # Нижняя поверхность (нормаль -Z)
        for s in valid_simplices:
            faces.append([s[0], s[2], s[1]])

        # Верхняя поверхность (нормаль +Z)
        for s in valid_simplices:
            faces.append([total_n + s[0], total_n + s[1], total_n + s[2]])

        # Боковые стены по контуру (CCW → нормаль наружу)
        for i in range(boundary_n):
            j = (i + 1) % boundary_n
            A = i;           B = j
            C = total_n + i; D = total_n + j
            faces.append([A, B, C])
            faces.append([B, D, C])

        # Накапливаем со смещением индексов
        all_vertices.extend(verts)
        all_faces.extend((np.array(faces, dtype=int) + vertex_offset).tolist())
        vertex_offset += len(verts)

    if not all_vertices:
        return None, None

    return np.array(all_vertices, dtype=float), np.array(all_faces, dtype=int)


def create_track_mesh(track_points, thickness, buffer_resolution=16):
    """Создает STL меш для трека используя Shapely buffer + Delaunay.

    Shapely LineString.buffer() автоматически объединяет перекрывающиеся
    участки (когда трек идёт дважды по одному месту), гарантируя
    единый корректный полигон без дублирующихся граней.
    """
    transformer     = Transformer.from_crs(myLib.WGS84, myLib.UTM, always_xy=True)
    transformer_inv = Transformer.from_crs(myLib.UTM, myLib.WGS84, always_xy=True)

    # 1) Преобразуем трек в UTM (только XY)
    xy_points = []
    for lat, lon in track_points:
        x, y = transformer.transform(lon, lat)
        xy_points.append((x, y))

    if len(xy_points) < 2:
        return None, None

    # 2) Строим 2D буфер — перекрывающиеся сегменты автоматически сливаются
    line      = LineString(xy_points)
    footprint = line.buffer(thickness / 2, resolution=buffer_resolution)
    if isinstance(footprint, MultiPolygon):
        footprint = unary_union(footprint)

    # 3) Точки контура (Shapely даёт CCW для exterior)
    boundary_coords = np.array(footprint.exterior.coords[:-1])  # без повторной точки
    boundary_n      = len(boundary_coords)

    # 4) Внутренняя сетка для более качественной триангуляции верхней поверхности
    minx, miny, maxx, maxy = footprint.bounds
    step         = thickness / 3
    interior_pts = []
    for gx in np.arange(minx + step / 2, maxx, step):
        for gy in np.arange(miny + step / 2, maxy, step):
            if footprint.contains(Point(gx, gy)):
                interior_pts.append([gx, gy])

    if interior_pts:
        all_xy = np.vstack([boundary_coords, np.array(interior_pts)])
    else:
        all_xy = boundary_coords.copy()

    total_n = len(all_xy)

    # 5) Delaunay-триангуляция; оставляем только треугольники внутри footprint
    tri             = Delaunay(all_xy)
    valid_simplices = []
    for simplex in tri.simplices:
        cx, cy = all_xy[simplex].mean(axis=0)
        if footprint.contains(Point(cx, cy)):
            valid_simplices.append(simplex)

    # 6) Высоты для каждой вершины
    elevations = np.zeros(total_n)
    for idx, (vx, vy) in enumerate(all_xy):
        lon_v, lat_v = transformer_inv.transform(vx, vy)
        h = myLib.elevation_data.get_elevation(lat_v, lon_v)
        if h is None:
            h = 0
        h += ADD_H
        if h < 0:
            h = 0
        elevations[idx] = h

    # 7) Вершины: нижний слой (z=0) + верхний (z=elevation)
    vertices_bot = np.column_stack([all_xy, np.zeros(total_n)])
    vertices_top = np.column_stack([all_xy, elevations])
    all_vertices = np.vstack([vertices_bot, vertices_top])

    faces = []

    # Нижняя поверхность (нормаль -Z: CW → порядок [s0, s2, s1])
    for s in valid_simplices:
        faces.append([s[0], s[2], s[1]])

    # Верхняя поверхность (нормаль +Z: CCW → порядок [s0, s1, s2])
    for s in valid_simplices:
        faces.append([total_n + s[0], total_n + s[1], total_n + s[2]])

    # Боковые стены по контуру (CCW контур → нормаль наружу)
    for i in range(boundary_n):
        j = (i + 1) % boundary_n
        A = i;           B = j
        C = total_n + i; D = total_n + j
        faces.append([A, B, C])
        faces.append([B, D, C])

    return all_vertices, np.array(faces, dtype=int)




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
