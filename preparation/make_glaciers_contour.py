# -*- coding: utf-8 -*-
"""
Скрипт для загрузки ледников и фильтрации по контуру из contour.geojson
"""

import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
from _download_glacier_data import GlacierDataDownloader, download_glaciers_for_region
import logging
import sys
import os

# Добавляем корневую директорию в путь для импорта myLib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import myLib

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_contour_bounds(contour_file: str = "contour.geojson"):
    """
    Получает границы из файла контура.
    
    Args:
        contour_file: Путь к файлу контура
        
    Returns:
        GeoDataFrame с контуром и его границы
    """
    logger.info(f"Загрузка контура из {contour_file}")
    
    # Читаем контур
    contour_gdf = gpd.read_file(contour_file)
    
    # Убеждаемся, что контур в WGS84
    if contour_gdf.crs != myLib.WGS84:
        logger.info(f"Преобразование контура из {contour_gdf.crs} в {myLib.WGS84}")
        contour_gdf = contour_gdf.to_crs(myLib.WGS84)
    
    # Получаем границы
    bounds = contour_gdf.total_bounds
    logger.info(f"Границы контура: min_lon={bounds[0]:.4f}, min_lat={bounds[1]:.4f}, "
                f"max_lon={bounds[2]:.4f}, max_lat={bounds[3]:.4f}")
    
    return contour_gdf, bounds


def filter_glaciers_by_contour(glaciers_gdf: gpd.GeoDataFrame,
                               contour_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Фильтрует ледники, оставляя только те, что находятся внутри контура,
    и обрезает их по границам контура.
    
    Args:
        glaciers_gdf: GeoDataFrame с ледниками
        contour_gdf: GeoDataFrame с контуром
        
    Returns:
        GeoDataFrame с отфильтрованными и обрезанными ледниками
    """
    logger.info("Фильтрация ледников по контуру")
    
    # Получаем геометрию контура (предполагаем, что это один полигон)
    contour_geometry = contour_gdf.geometry.iloc[0]
    
    # Фильтруем ледники, которые пересекают контур
    intersecting_glaciers = glaciers_gdf[glaciers_gdf.intersects(contour_geometry)]
    
    logger.info(f"Найдено {len(intersecting_glaciers)} ледников, пересекающих контур")
    
    # Обрезаем ледники по границам контура
    filtered_glaciers = intersecting_glaciers.copy()
    filtered_glaciers['geometry'] = intersecting_glaciers['geometry'].intersection(contour_geometry)
    
    # Удаляем пустые геометрии (если ледник полностью вне контура)
    filtered_glaciers = filtered_glaciers[~filtered_glaciers.geometry.is_empty]
    
    logger.info(f"После обрезки: {len(filtered_glaciers)} ледников внутри контура")
    
    return filtered_glaciers


def visualize_results(contour_gdf: gpd.GeoDataFrame, 
                     glaciers_gdf: gpd.GeoDataFrame,
                     filtered_glaciers: gpd.GeoDataFrame,
                     output_file: str = "glaciers_in_contour.png"):
    """
    Визуализирует контур и ледники.
    
    Args:
        contour_gdf: GeoDataFrame с контуром
        glaciers_gdf: GeoDataFrame со всеми ледниками
        filtered_glaciers: GeoDataFrame с отфильтрованными ледниками
        output_file: Имя файла для сохранения визуализации
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Отображаем все ледники (полупрозрачными)
    if not glaciers_gdf.empty:
        glaciers_gdf.plot(ax=ax, color='lightblue', edgecolor='blue', 
                         alpha=0.3, label='Все ледники')
    
    # Отображаем контур
    contour_gdf.plot(ax=ax, facecolor='none', edgecolor='red', 
                    linewidth=2, linestyle='--', label='Контур')
    
    # Отображаем ледники внутри контура
    if not filtered_glaciers.empty:
        filtered_glaciers.plot(ax=ax, color='darkblue', edgecolor='navy',
                              alpha=0.7, linewidth=2, label='Ледники в контуре')
    
    # Устанавливаем границы по контуру
    bounds = contour_gdf.total_bounds
    ax.set_xlim(bounds[0] - 0.1, bounds[2] + 0.1)
    ax.set_ylim(bounds[1] - 0.1, bounds[3] + 0.1)
    
    ax.set_xlabel('Долгота')
    ax.set_ylabel('Широта')
    ax.set_title('Ледники внутри контура')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Основная функция для загрузки и фильтрации ледников по контуру.
    """
    parser = argparse.ArgumentParser(
        description="Загрузка ледников по контуру"
    )
    parser.add_argument(
        '--plot', '-p', action='store_true',
        help='Показать график с контуром и ледниками'
    )
    args = parser.parse_args()

    print("=== Поиск ледников внутри контура ===\n")
    
    # 1. Загружаем контур
    contour_gdf, bounds = get_contour_bounds("contour.geojson")
    
    # 2. Определяем источник данных на основе границ контура
    # Контур находится на Кавказе (примерно 42°E, 43°N)
    # Используем RGI регион 10 (Caucasus/Middle East)
    
    logger.info("Загрузка ледников для региона Кавказ (RGI регион 10)")
    
    # 3. Загружаем ледники
    downloader = GlacierDataDownloader()
    
    # Сначала пробуем Natural Earth (быстро)
    logger.info("Попытка загрузки из Natural Earth...")
    glaciers_gdf = downloader.download_natural_earth_glaciers("temp_glaciers.geojson")
    
    if glaciers_gdf.empty:
        logger.warning("Natural Earth не дал результатов, пробуем RGI...")
        # Если Natural Earth не сработал, пробуем RGI
        glaciers_gdf = downloader.download_rgi_data(10, "temp_glaciers.geojson")
    
    if glaciers_gdf.empty:
        logger.warning("RGI не дал результатов, пробуем GLIMS...")
        # Если RGI не сработал, пробуем GLIMS
        glaciers_gdf = downloader.download_glims_data(
            "Кавказ", bounds[0], bounds[1], bounds[2], bounds[3],
            "temp_glaciers.geojson"
        )
    
    if glaciers_gdf.empty:
        logger.error("Не удалось загрузить данные о ледниках ни из одного источника")
        return
    
    # Убеждаемся, что ледники в WGS84
    if glaciers_gdf.crs != myLib.WGS84:
        logger.info(f"Преобразование ледников из {glaciers_gdf.crs} в {myLib.WGS84}")
        glaciers_gdf = glaciers_gdf.to_crs(myLib.WGS84)
    
    # 4. Фильтруем ледники по контуру
    filtered_glaciers = filter_glaciers_by_contour(glaciers_gdf, contour_gdf)
    
    # 5. Визуализируем результаты (только если запрошено)
    if args.plot:
        visualize_results(contour_gdf, glaciers_gdf, filtered_glaciers)
    
    # 6. Сохраняем результаты
    if not filtered_glaciers.empty:
        output_file = "glaciers_contour.geojson"
        # Убеждаемся, что сохраняем в WGS84
        if filtered_glaciers.crs != myLib.WGS84:
            filtered_glaciers = filtered_glaciers.to_crs(myLib.WGS84)
        filtered_glaciers.to_file(output_file, driver='GeoJSON')
        logger.info(f"Ледники внутри контура сохранены в {output_file} (CRS: {myLib.WGS84})")
        
    else:
        print("\nЛедники внутри указанного контура не найдены.")
        print("Возможно, в данном регионе нет ледников или они слишком малы для текущего источника данных.")


if __name__ == "__main__":
    main()
