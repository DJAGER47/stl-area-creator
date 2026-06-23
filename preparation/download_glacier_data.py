# -*- coding: utf-8 -*-
"""
Скрипт для загрузки контуров ледников из открытых источников данных.
Поддерживает данные из GLIMS (Global Land Ice Measurements from Space) и других источников.
"""

import requests
import geopandas as gpd
import json
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import zipfile
import io

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GlacierDataDownloader:
    """
    Класс для загрузки данных о ледниках из различных источников.
    """
    
    def __init__(self, cache_dir: str = "data/glacier_cache"):
        """
        Инициализация загрузчика данных ледников.
        
        Args:
            cache_dir: Директория для кэширования загруженных данных
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_glims_data(self, region_name: str, 
                           min_lon: float, min_lat: float,
                           max_lon: float, max_lat: float,
                           output_file: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Загружает данные о ледниках из базы GLIMS.
        
        GLIMS (Global Land Ice Measurements from Space) - глобальная база данных ледников.
        
        Args:
            region_name: Название региона для идентификации
            min_lon: Минимальная долгота
            min_lat: Минимальная широта
            max_lon: Максимальная долгота
            max_lat: Максимальная широта
            output_file: Имя файла для сохранения (опционально)
            
        Returns:
            GeoDataFrame с контурами ледников
        """
        logger.info(f"Загрузка данных GLIMS для региона: {region_name}")
        
        # GLIMS API endpoint
        glims_url = "https://www.glims.org/maps/glims"
        
        # Формируем параметры запроса
        params = {
            'format': 'json',
            'bbox': f"{min_lon},{min_lat},{max_lon},{max_lat}",
            'per_page': 1000
        }
        
        try:
            response = requests.get(glims_url, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'features' not in data:
                logger.warning(f"Данные GLIMS не найдены для региона {region_name}")
                return gpd.GeoDataFrame()
            
            # Создаем GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(data['features'], crs='EPSG:4326')
            
            logger.info(f"Загружено {len(gdf)} ледников из GLIMS")
            
            # Сохраняем в файл если указано
            if output_file:
                self._save_geodataframe(gdf, output_file)
            
            return gdf
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка загрузки данных GLIMS: {e}")
            return gpd.GeoDataFrame()
    
    def download_rgi_data(self, region_id: int, 
                         output_file: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Загружает данные из Randolph Glacier Inventory (RGI).
        
        RGI - глобальный инвентарный список ледников, созданный в рамках GLIMS.
        
        Args:
            region_id: ID региона RGI (1-19 для разных регионов мира)
            output_file: Имя файла для сохранения (опционально)
            
        Returns:
            GeoDataFrame с контурами ледников
        """
        logger.info(f"Загрузка данных RGI для региона {region_id}")
        
        # RGI 6.0 download URLs (более стабильная версия)
        rgi_base_url = "https://www.glims.org/RGI/rgi60_files/"
        
        # Список регионов RGI 6.0
        rgi_regions = {
            1: "01_rgi60_Arctic_Canada_North.zip",
            2: "02_rgi60_Arctic_Canada_South.zip",
            3: "03_rgi60_Greenland_Periphery.zip",
            4: "04_rgi60_Iceland.zip",
            5: "05_rgi60_Svalbard_Jan_Mayen.zip",
            6: "06_rgi60_Scandinavia.zip",
            7: "07_rgi60_Russian_Arctic.zip",
            8: "08_rgi60_North_Asia.zip",
            9: "09_rgi60_Central_Europe.zip",
            10: "10_rgi60_Caucasus_Middle_East.zip",
            11: "11_rgi60_Central_Asia.zip",
            12: "12_rgi60_South_Asia_West.zip",
            13: "13_rgi60_South_Asia_East.zip",
            14: "14_rgi60_Low_Latitudes.zip",
            15: "15_rgi60_Southern_Andes.zip",
            16: "16_rgi60_New_Zealand.zip",
            17: "17_rgi60_Antarctic_Subantarctic.zip",
            18: "18_rgi60_Antarctica_Mainland.zip",
            19: "19_rgi60_Antarctic_Islands.zip"
        }
        
        if region_id not in rgi_regions:
            logger.error(f"Неверный ID региона RGI. Доступные регионы: {list(rgi_regions.keys())}")
            return gpd.GeoDataFrame()
        
        region_name = rgi_regions[region_id]
        url = f"{rgi_base_url}{region_name}"
        
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Сохраняем ZIP во временный файл
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                tmp_zip.write(response.content)
                tmp_zip_path = tmp_zip.name
            
            # Читаем shapefile из ZIP
            try:
                gdf = gpd.read_file(f"zip://{tmp_zip_path}")
            finally:
                # Удаляем временный файл
                import os
                os.unlink(tmp_zip_path)
            
            logger.info(f"Загружено {len(gdf)} ледников из RGI региона {region_id}")
            
            # Сохраняем в файл если указано
            if output_file:
                self._save_geodataframe(gdf, output_file)
            
            return gdf
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка загрузки данных RGI: {e}")
            return gpd.GeoDataFrame()
    
    def download_natural_earth_glaciers(self, output_file: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Загружает данные о ледниках из Natural Earth.
        
        Natural Earth предоставляет общие данные о ледниках и ледяных шапках.
        
        Args:
            output_file: Имя файла для сохранения (опционально)
            
        Returns:
            GeoDataFrame с контурами ледников
        """
        logger.info("Загрузка данных ледников из Natural Earth")
        
        # Natural Earth glaciers URL
        ne_url = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_glaciated_areas.zip"
        
        try:
            response = requests.get(ne_url, timeout=120)
            response.raise_for_status()
            
            # Сохраняем ZIP во временный файл
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                tmp_zip.write(response.content)
                tmp_zip_path = tmp_zip.name
            
            # Читаем shapefile из ZIP
            try:
                gdf = gpd.read_file(f"zip://{tmp_zip_path}")
            finally:
                # Удаляем временный файл
                import os
                os.unlink(tmp_zip_path)
            
            logger.info(f"Загружено {len(gdf)} ледниковых областей из Natural Earth")
            
            # Сохраняем в файл если указано
            if output_file:
                self._save_geodataframe(gdf, output_file)
            
            return gdf
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка загрузки данных Natural Earth: {e}")
            return gpd.GeoDataFrame()
    
    def filter_glaciers_by_bounds(self, gdf: gpd.GeoDataFrame,
                                  min_lon: float, min_lat: float,
                                  max_lon: float, max_lat: float) -> gpd.GeoDataFrame:
        """
        Фильтрует ледники по заданным границам.
        
        Args:
            gdf: GeoDataFrame с ледниками
            min_lon: Минимальная долгота
            min_lat: Минимальная широта
            max_lon: Максимальная долгота
            max_lat: Максимальная широта
            
        Returns:
            Отфильтрованный GeoDataFrame
        """
        from shapely.geometry import box
        
        # Создаем прямоугольник для фильтрации
        bounds_box = box(min_lon, min_lat, max_lon, max_lat)
        
        # Фильтруем ледники, пересекающие прямоугольник
        filtered_gdf = gdf[gdf.intersects(bounds_box)]
        
        logger.info(f"Отфильтровано {len(filtered_gdf)} ледников в заданных границах")
        
        return filtered_gdf
    
    def _save_geodataframe(self, gdf: gpd.GeoDataFrame, filename: str):
        """
        Сохраняет GeoDataFrame в файл.
        
        Args:
            gdf: GeoDataFrame для сохранения
            filename: Имя файла
        """
        filepath = self.cache_dir / filename
        
        try:
            gdf.to_file(filepath, driver='GeoJSON')
            logger.info(f"Данные сохранены в {filepath}")
        except Exception as e:
            logger.error(f"Ошибка сохранения файла: {e}")
    
    def visualize_glaciers(self, gdf: gpd.GeoDataFrame, 
                          bounds: Optional[Tuple[float, float, float, float]] = None):
        """
        Визуализирует ледники на карте.
        
        Args:
            gdf: GeoDataFrame с ледниками
            bounds: Границы для отображения (min_lon, min_lat, max_lon, max_lat)
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Отображаем ледники
        gdf.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.6, 
                label='Ледники')
        
        # Устанавливаем границы если указаны
        if bounds:
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
        elif not gdf.empty:
            # Используем границы данных
            bounds = gdf.total_bounds
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
        
        ax.set_xlabel('Долгота')
        ax.set_ylabel('Широта')
        ax.set_title('Контуры ледников')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.cache_dir / 'glaciers_visualization.png', dpi=300)
        logger.info(f"Визуализация сохранена в {self.cache_dir / 'glaciers_visualization.png'}")
        plt.show()


def download_glaciers_for_region(region_name: str, 
                                 min_lon: float, min_lat: float,
                                 max_lon: float, max_lat: float,
                                 source: str = "natural_earth",
                                 rgi_region_id: Optional[int] = None) -> gpd.GeoDataFrame:
    """
    Загружает ледники для указанного региона.
    
    Args:
        region_name: Название региона
        min_lon: Минимальная долгота
        min_lat: Минимальная широта
        max_lon: Максимальная долгота
        max_lat: Максимальная широта
        source: Источник данных ("rgi", "glims", "natural_earth")
        rgi_region_id: ID региона RGI (если source="rgi")
        
    Returns:
        GeoDataFrame с контурами ледников
    """
    downloader = GlacierDataDownloader()
    
    output_filename = f"{region_name.replace(' ', '_')}_glaciers.geojson"
    
    if source == "rgi":
        if rgi_region_id is None:
            logger.error("Для RGI необходимо указать rgi_region_id")
            return gpd.GeoDataFrame()
        
        gdf = downloader.download_rgi_data(rgi_region_id, output_filename)
        
    elif source == "glims":
        gdf = downloader.download_glims_data(region_name, min_lon, min_lat, 
                                             max_lon, max_lat, output_filename)
        
    elif source == "natural_earth":
        gdf = downloader.download_natural_earth_glaciers(output_filename)
        
    else:
        logger.error(f"Неизвестный источник данных: {source}")
        return gpd.GeoDataFrame()
    
    if not gdf.empty:
        # Фильтруем по границам
        gdf = downloader.filter_glaciers_by_bounds(gdf, min_lon, min_lat, 
                                                  max_lon, max_lat)
        
        # Визуализируем
        downloader.visualize_glaciers(gdf, (min_lon, min_lat, max_lon, max_lat))
    
    return gdf


if __name__ == "__main__":
    print("=== Загрузка контуров ледников ===\n")
    
    # Пример 1: Загрузка ледников из Natural Earth (рекомендуется для начала)
    print("Пример 1: Ледники из Natural Earth (быстро и надежно)")
    world_glaciers = download_glaciers_for_region(
        region_name="World",
        min_lon=-180.0, min_lat=-90.0,
        max_lon=180.0, max_lat=90.0,
        source="natural_earth"
    )
    
    # Пример 2: Загрузка ледников Кавказа через GLIMS
    print("\nПример 2: Ледники Кавказа через GLIMS")
    caucasus_glaciers = download_glaciers_for_region(
        region_name="Кавказ",
        min_lon=40.0, min_lat=42.0,
        max_lon=46.0, max_lat=44.0,
        source="glims"
    )
    
    # Пример 3: Загрузка ледников Аляски через RGI
    print("\nПример 3: Ледники Аляски через RGI")
    alaska_glaciers = download_glaciers_for_region(
        region_name="Аляска",
        min_lon=-152.0, min_lat=58.0,
        max_lon=-140.0, max_lat=62.0,
        source="rgi",
        rgi_region_id=1
    )
    
    print("\n=== Доступные источники данных ===")
    print("1. Natural Earth - быстрый и надежный источник (рекомендуется для начала)")
    print("2. GLIMS (Global Land Ice Measurements from Space) - API доступ")
    print("3. RGI (Randolph Glacier Inventory) - наиболее полный источник")
    
    print("\n=== Регионы RGI ===")
    print("1: Arctic Canada North, 2: Arctic Canada South, 3: Greenland Periphery")
    print("4: Iceland, 5: Svalbard/Jan Mayen, 6: Scandinavia, 7: Russian Arctic")
    print("8: North Asia, 9: Central Europe, 10: Caucasus/Middle East")
    print("11: Central Asia, 12: South Asia West, 13: South Asia East")
    print("14: Low Latitudes, 15: Southern Andes, 16: New Zealand")
    print("17: Antarctic/Subantarctic, 18: Antarctica Mainland, 19: Antarctic Islands")
