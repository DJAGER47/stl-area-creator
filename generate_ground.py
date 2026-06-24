import argparse
import copy
import os
import sys
import tracemalloc

import geojson
import geopandas as gpd
import numpy as np
from shapely.geometry import shape

import myLib
from stl import mesh

logger = myLib.logger

def generate(obl_name: str, path_save: str, step_m: int, oblast, water) -> None:
    total_timer = myLib.Timer(f"Processing {obl_name}")
    with total_timer:
        logger.info(f"{'Starting parameters':<50} Step: {step_m}m | Scale: {step_m * myLib.SCALE:.2f}mm")
        
        with myLib.Timer("Geometry preparation"):
            geo_simply = myLib.multy_to_polygon(oblast.geometry, myLib.WGS84)
            geo_simply = myLib.multy_to_polygon(geo_simply.buffer(-(myLib.REDUCE / myLib.SCALE)), myLib.UTM)
            logger.info(f"{'Reduction parameters':<50} {myLib.REDUCE:.2f}mm | {myLib.REDUCE / myLib.SCALE:.2f}m")

        contour = myLib.make_contour(geo_simply, step_m)
        area_mesh = myLib.filter_mesh(contour, myLib.make_mesh(contour, step_m))

        with myLib.Timer("Height processing"):
            min_h = 6000
            max_h = 0
            for i, polygon in enumerate(contour):
                contour[i] = myLib.GetHeight(polygon.exterior.coords, step_m, water)
                min_h = min(min_h, min(contour[i], key=lambda x: x[2])[2])
                max_h = max(max_h, max(contour[i], key=lambda x: x[2])[2])

            for i, points in enumerate(area_mesh):
                area_mesh[i] = myLib.GetHeight([(point.x, point.y) for point in points.geometry], step_m, water)
                min_h = min(min_h, min(area_mesh[i], key=lambda x: x[2])[2])
                max_h = max(max_h, max(area_mesh[i], key=lambda x: x[2])[2])

            logger.info(f"{'Height range':<50} {min_h:.2f}m - {max_h:.2f}m")

        with myLib.Timer("Height filtering"):
            total_filtered = 0
            total_points = 0
            
            # Фильтруем только внутренние точки сетки (area_mesh)
            for i, points in enumerate(area_mesh):
                area_mesh[i], stats = myLib.filter_height_points(points, step_m, threshold=0.3)
                total_filtered += stats['filtered_count']
                total_points += stats['total_count']
            
            filter_percent = (total_filtered / total_points * 100) if total_points > 0 else 0
            logger.info(f"{'Filtering results':<50} {total_filtered}/{total_points} ({filter_percent:.2f}%)")

        with myLib.Timer("Coordinate conversion"):
            utm_contour = [[(p[0], p[1], p[2]) for p in points] for points in contour]
            utm_mesh = [copy.deepcopy(utm_contour[i]) + [(p[0], p[1], p[2]) for p in points]
                       for i, points in enumerate(area_mesh)]
            utm_contour_zero = [[(p[0], p[1], 0) for p in points] for points in utm_contour]

        list_stl = myLib.make_stl_obl(utm_contour, utm_mesh, utm_contour_zero)
        
        with myLib.Timer("Saving STL files"):
            for i, obl in enumerate(list_stl):
                filename = f'{path_save}{obl_name}_{step_m}_{i}.stl'
                obl.save(filename)
                logger.info(f"Saved {filename}")

def main():
    tracemalloc.start()
    path = "stl/"
    water = []
    paths_water = {}

    parser = argparse.ArgumentParser(
        description="Генератор геоданных",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-C', '--contour',
                        type=str,
                        help="Путь к файлу контура (GeoJSON)")
    parser.add_argument('-s', '--step', 
                        type=int,
                        required=True,
                        help="Шаг генерации (метры)")
    parser.add_argument('-n', '--name', 
                        type=str,
                        help="Название области для генерации")
    
    args = parser.parse_args()

    with myLib.Timer("Water data loading"):
        for path_water in paths_water:
            with open(path_water, 'r') as f:
                sea_data = geojson.load(f)
            water.append([shape(feature['geometry']) for feature in sea_data['features']])

    if args.contour:
        with myLib.Timer("Contour data loading"):
            gpkd = gpd.read_file(args.contour)
        generate(os.path.basename(args.contour), path, args.step, gpkd, water)
    else:
        with myLib.Timer("Region data loading"):
            gpkd = gpd.read_file("data/russia_regions.geojson")
        
        if args.name:
            oblast = gpkd[gpkd["region"] == args.name]
            if oblast.empty:
                available = '\n'.join(gpkd["region"].unique())
                logger.error(f"Область '{args.name}' не найдена. Доступные области:\n{available}")
                exit(1)
            generate(args.name, path, args.step, oblast, water)
        else:
            for region in gpkd["region"]:
                oblast = gpkd[gpkd["region"] == region]
                generate(region, path, args.step, oblast, water)

    size, peak = tracemalloc.get_traced_memory()
    logger.info(f"{'Memory usage':<50} {size/1024:>7.1f} KB")
    logger.info(f"{'Peak memory usage':<50} {peak/1024:>7.1f} KB")

if __name__ == "__main__":
    main()
