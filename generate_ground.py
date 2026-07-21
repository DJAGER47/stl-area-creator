import argparse
import copy
import os
import sys
import tracemalloc

import argcomplete
import geojson
import geopandas as gpd
import numpy as np
from shapely.geometry import shape

import myLib
from stl import mesh

logger = myLib.logger

def find_geojson_files(prefix, parsed_args):
    """Находит файлы .geojson для автодополнения"""
    import glob
    files = glob.glob(f"{prefix}*.geojson")
    # Также ищем в текущей директории
    if not files:
        files = glob.glob(f"*.geojson")
    return files

def generate(obl_name: str, path_save: str, step_m: int, oblast, water, overlay_distance=None, split=False, add_height=0.0, shift_min=None) -> None:
    """Генерирует STL модель области
    
    Args:
        obl_name: Название области
        path_save: Путь для сохранения файлов
        step_m: Шаг генерации в метрах
        oblast: Геоданные области
        water: Данные о водных объектах
        overlay_distance: Если задано, нижняя подложка повторяет форму верхнего контура и опускается на это расстояние (метры)
        split: Если True, сохранять каждую область по отдельности. Если False, объединить все объекты в один STL файл
        add_height: Константное добавление высоты (метры) ко всем точкам. 0.0 — выключено
        shift_min: Если задано, после построения карты высот находит минимальную точку, отнимает от неё это значение
            и смещает нижнюю подложку вниз на получившуюся величину (верхняя поверхность остаётся на исходных высотах)
    """
    total_timer = myLib.Timer(f"Processing {obl_name}")
    with total_timer:
        logger.info(f"{'Starting parameters':<50} Step: {step_m}m | Scale: {step_m * myLib.SCALE:.2f}mm")
        if overlay_distance is not None:
            logger.info(f"{'Overlay mode':<50} Enabled (distance: {overlay_distance}m)")
        if shift_min is not None:
            logger.info(f"{'Shift-min mode':<50} Enabled (target min: {shift_min}m)")
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
                contour[i] = [(p[0], p[1], p[2] + add_height) for p in contour[i]]
                min_h = min(min_h, min(contour[i], key=lambda x: x[2])[2])
                max_h = max(max_h, max(contour[i], key=lambda x: x[2])[2])

            for i, points in enumerate(area_mesh):
                area_mesh[i] = myLib.GetHeight([(point.x, point.y) for point in points.geometry], step_m, water)
                area_mesh[i] = [(p[0], p[1], p[2] + add_height) for p in area_mesh[i]]
                min_h = min(min_h, min(area_mesh[i], key=lambda x: x[2])[2])
                max_h = max(max_h, max(area_mesh[i], key=lambda x: x[2])[2])

            logger.info(f"{'Height range':<50} {min_h:.2f}m - {max_h:.2f}m")
            if add_height:
                logger.info(f"{'Constant height addition':<50} +{add_height}m")

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

        shift_value = 0
        if shift_min is not None:
            with myLib.Timer("Min-height shift"):
                all_z = [p[2] for points in contour for p in points] + \
                        [p[2] for points in area_mesh for p in points]
                if not all_z:
                    logger.info(f"{'Shift skipped':<50} no height points")
                else:
                    min_h = min(all_z)
                    max_h = max(all_z)
                    shift_value = min_h - shift_min
                    logger.info(f"{'Min height (after filtering)':<50} {min_h:.2f}m")
                    logger.info(f"{'Shift value (min - target)':<50} {shift_value:.2f}m")
                    if shift_value > 0:
                        logger.info(f"{'Bottom shift':<50} base plate moved to z={shift_value:.2f}m")
                    else:
                        shift_value = 0
                        logger.info(f"{'Shift skipped':<50} shift_value <= 0")

        with myLib.Timer("Coordinate conversion"):
            utm_contour = [[(p[0], p[1], p[2]) for p in points] for points in contour]
            utm_mesh = [copy.deepcopy(utm_contour[i]) + [(p[0], p[1], p[2]) for p in points]
                       for i, points in enumerate(area_mesh)]
            
            if overlay_distance is None:
                utm_contour_zero = [[(p[0], p[1], shift_value) for p in points] for points in utm_contour]
            else:
                utm_contour_zero = None

        list_stl = myLib.make_stl_obl(utm_contour, utm_mesh, utm_contour_zero, overlay_distance)
        
        with myLib.Timer("Saving STL files"):
            if split:
                for i, obl in enumerate(list_stl):
                    filename = f'{path_save}{obl_name}_{step_m}_{i}.stl'
                    obl.save(filename)
            else:
                combined_vectors = np.concatenate([obl.vectors for obl in list_stl])
                combined_mesh = mesh.Mesh(np.zeros(combined_vectors.shape[0], dtype=mesh.Mesh.dtype))
                for i, f in enumerate(combined_vectors):
                    combined_mesh.vectors[i] = f
                combined_mesh.update_normals()
                filename = f'{path_save}{obl_name}_{step_m}.stl'
                combined_mesh.save(filename)

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
    parser.add_argument('-c', '--contour',
                        type=str,
                        help="Путь к файлу контура (GeoJSON)").completer = find_geojson_files
    parser.add_argument('-s', '--step',
                        type=int,
                        required=True,
                        help="Шаг генерации (метры)")
    parser.add_argument('-n', '--name',
                        type=str,
                        help="Название области для генерации")
    parser.add_argument('-o', '--overlay',
                        type=float,
                        default=None,
                        help="Режим наложения: нижняя подложка повторяет форму верхнего контура и опускается на указанное расстояние (метры)")
    parser.add_argument('--split',
                        action='store_true',
                        default=False,
                        help="Сохранять каждую область по отдельности. Если не указан, все объекты объединяются в один STL файл")
    parser.add_argument('--add-height',
                        type=float,
                        default=0.0,
                        help="Константное добавление высоты (метры) ко всем точкам модели. По умолчанию выключено (0.0)")
    parser.add_argument('--shift-min',
                        type=float,
                        default=None,
                        help="Сдвиг по минимальной высоте: после построения карты высот находит минимальную точку, "
                             "отнимает от неё это значение и смещает нижнюю подложку вниз на получившуюся величину "
                             "(верхняя поверхность остаётся на исходных высотах). По умолчанию выключено")
    
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    with myLib.Timer("Water data loading"):
        for path_water in paths_water:
            with open(path_water, 'r') as f:
                sea_data = geojson.load(f)
            water.append([shape(feature['geometry']) for feature in sea_data['features']])

    if args.contour:
        with myLib.Timer("Contour data loading"):
            gpkd = gpd.read_file(args.contour)
        generate(os.path.basename(args.contour), path, args.step, gpkd, water, args.overlay, args.split, args.add_height, args.shift_min)
    else:
        with myLib.Timer("Region data loading"):
            gpkd = gpd.read_file("data/russia_regions.geojson")
        
        if args.name:
            oblast = gpkd[gpkd["region"] == args.name]
            if oblast.empty:
                available = '\n'.join(gpkd["region"].unique())
                logger.error(f"Область '{args.name}' не найдена. Доступные области:\n{available}")
                exit(1)
            generate(args.name, path, args.step, oblast, water, args.overlay, args.split, args.add_height, args.shift_min)
        else:
            for region in gpkd["region"]:
                oblast = gpkd[gpkd["region"] == region]
                generate(region, path, args.step, oblast, water, args.overlay, args.split, args.add_height, args.shift_min)

    size, peak = tracemalloc.get_traced_memory()
    logger.info(f"{'Memory usage':<50} {size/1024:>7.1f} KB")
    logger.info(f"{'Peak memory usage':<50} {peak/1024:>7.1f} KB")

if __name__ == "__main__":
    main()
