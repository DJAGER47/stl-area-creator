# -*- coding: utf-8 -*-
"""
Скрипт для загрузки рек и водоёмов и фильтрации по контуру из contour.geojson.

Источником данных служит OpenStreetMap (Overpass API). Загружаются:
  - реки (линейные объекты: waterway=river/stream/canal)
  - водоёмы (полигональные объекты: natural=water, water=*, waterway=riverbank/dock)

Результат сохраняется в water_contour.geojson в проекции WGS84.
"""

import argparse
import geopandas as gpd
from shapely.geometry import shape, mapping
from OSMPythonTools.overpass import Overpass
import matplotlib.pyplot as plt
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

    contour_gdf = gpd.read_file(contour_file)

    if contour_gdf.crs != myLib.WGS84:
        logger.info(f"Преобразование контура из {contour_gdf.crs} в {myLib.WGS84}")
        contour_gdf = contour_gdf.to_crs(myLib.WGS84)

    bounds = contour_gdf.total_bounds
    logger.info(f"Границы контура: min_lon={bounds[0]:.4f}, min_lat={bounds[1]:.4f}, "
                f"max_lon={bounds[2]:.4f}, max_lat={bounds[3]:.4f}")

    return contour_gdf, bounds


def _bbox_cache_path(prefix: str, min_lon: float, min_lat: float,
                     max_lon: float, max_lat: float) -> str:
    """Формирует путь к кеш-файлу на основе bounding box."""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    bbox_str = f"{min_lon:.4f}_{min_lat:.4f}_{max_lon:.4f}_{max_lat:.4f}"
    return os.path.join(cache_dir, f"{prefix}_{bbox_str}.geojson")


def _load_from_cache(cache_path: str) -> gpd.GeoDataFrame | None:
    """Пытается загрузить данные из кеш-файла."""
    if os.path.exists(cache_path):
        logger.info(f"Загрузка из кеша: {cache_path}")
        try:
            gdf = gpd.read_file(cache_path)
            if gdf.crs is None:
                gdf.set_crs(myLib.WGS84, inplace=True)
            logger.info(f"Загружено {len(gdf)} объектов из кеша")
            return gdf
        except Exception as e:
            logger.warning(f"Не удалось прочитать кеш {cache_path}: {e}")
    return None


def _save_to_cache(gdf: gpd.GeoDataFrame, cache_path: str):
    """Сохраняет данные в кеш-файл."""
    logger.info(f"Сохранение в кеш: {cache_path}")
    try:
        if gdf.crs is None:
            gdf.set_crs(myLib.WGS84, inplace=True)
        gdf.to_file(cache_path, driver='GeoJSON')
    except Exception as e:
        logger.warning(f"Не удалось сохранить кеш {cache_path}: {e}")


def download_water_from_osm(min_lon: float, min_lat: float,
                            max_lon: float, max_lat: float,
                            use_cache: bool = True) -> gpd.GeoDataFrame:
    """
    Загружает реки и водоёмы из OpenStreetMap через Overpass API по bounding box.

    Args:
        min_lon: Минимальная долгота
        min_lat: Минимальная широта
        max_lon: Максимальная долгота
        max_lat: Максимальная широта
        use_cache: Использовать кеш (по умолчанию True)

    Returns:
        GeoDataFrame с водными объектами (WGS84)
    """
    cache_path = _bbox_cache_path("water_osm", min_lon, min_lat, max_lon, max_lat)

    # Пробуем загрузить из кеша
    if use_cache:
        cached = _load_from_cache(cache_path)
        if cached is not None:
            return cached

    logger.info("Загрузка рек и водоёмов из OpenStreetMap (Overpass API)")

    overpass = Overpass()

    # Запрос: реки (линии) и водоёмы (полигоны) внутри bounding box.
    # Заголовок [out:json][timeout:...] добавляет сам OSMPythonTools.
    query = f"""
    (
      way["waterway"~"river|stream|canal"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["waterway"~"river|stream|canal"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["water"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["water"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom;
    """

    logger.info("Выполнение запроса к Overpass API...")
    result = overpass.query(query)
    elements = result.elements()
    logger.info(f"Получено {len(elements)} элементов из OSM")

    features = []
    for element in elements:
        try:
            geometry = element.geometry()
            if geometry is None:
                continue
            geom = shape(geometry)
            if geom.is_empty:
                continue

            tags = element.tags() or {}
            props = {
                'osm_id': element.id(),
                'osm_type': element.type(),
                'name': tags.get('name', ''),
                'waterway': tags.get('waterway', ''),
                'natural': tags.get('natural', ''),
                'water': tags.get('water', ''),
            }
            features.append({'type': 'Feature', 'properties': props,
                             'geometry': mapping(geom)})
        except Exception as e:
            logger.warning(f"Не удалось обработать элемент {element.id()}: {e}")
            continue

    if not features:
        logger.warning("Не найдено водных объектов в OSM для заданного bbox")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame.from_features(features, crs=myLib.WGS84)
    logger.info(f"Сформировано {len(gdf)} водных объектов")

    # Сохраняем в кеш
    if use_cache:
        _save_to_cache(gdf, cache_path)

    return gdf


def filter_water_by_contour(water_gdf: gpd.GeoDataFrame,
                            contour_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Фильтрует водные объекты, оставляя только те, что пересекают контур,
    и обрезает их по границам контура.

    Args:
        water_gdf: GeoDataFrame с водными объектами
        contour_gdf: GeoDataFrame с контуром

    Returns:
        GeoDataFrame с отфильтрованными и обрезанными водными объектами
    """
    logger.info("Фильтрация водных объектов по контуру")

    contour_geometry = contour_gdf.geometry.iloc[0]

    intersecting = water_gdf[water_gdf.intersects(contour_geometry)]
    logger.info(f"Найдено {len(intersecting)} объектов, пересекающих контур")

    filtered = intersecting.copy()
    filtered['geometry'] = intersecting['geometry'].intersection(contour_geometry)

    # Удаляем пустые геометрии
    filtered = filtered[~filtered.geometry.is_empty]
    filtered = filtered[~filtered.geometry.isna()]

    logger.info(f"После обрезки: {len(filtered)} объектов внутри контура")
    return filtered


def generalize_geometries(gdf: gpd.GeoDataFrame,
                         buffer_size: float = 0.0,
                         min_area: float = 0.0,
                         simplify_tolerance: float = 0.0) -> gpd.GeoDataFrame:
    """
    Обобщает геометрии: объединяет близкие/пересекающиеся полигоны в один,
    сглаживает рваные границы, удаляет мелкие фрагменты и упрощает контур.

    Алгоритм:
      1. Объединение всех геометрий в одну (dissolve) — сливает пересекающиеся
         и касающиеся полигоны в единый Multipolygon.
      2. Положительный буфер (buffer_size) — заполняет щели между близкими
         кусками и сглаживает рваные края.
      3. Отрицательный буфер того же размера — возвращает исходную площадь,
         сохраняя сглаженную форму (приём buffer-in/buffer-out).
      4. Удаление полигонов меньше min_area — отсекает мелкий мусор.
      5. Упрощение (simplify_tolerance) — алгоритм Дугласа-Пекера для
         уменьшения числа точек.

    Args:
        gdf: GeoDataFrame с объектами (WGS84)
        buffer_size: Размер буфера в градусах для объединения/сглаживания
        min_area: Минимальная площадь полигона в квадратных градусах;
                  меньшие удаляются
        simplify_tolerance: Допуск упрощения в градусах (0 — не упрощать)

    Returns:
        GeoDataFrame с обобщёнными геометриями (одна строка на связный кластер)
    """
    logger.info(f"Обобщение геометрий: buffer={buffer_size}, "
                f"min_area={min_area}, simplify={simplify_tolerance}")

    if gdf.empty:
        return gdf

    # 1. Объединяем все геометрии в одну
    from shapely.ops import unary_union
    merged = unary_union(list(gdf.geometry))
    logger.info(f"После объединения: {len(merged.geoms) if hasattr(merged, 'geoms') else 1} компонент(ов)")

    # 2-3. Сглаживание буфером: расширить → сжать (buffer-in/buffer-out)
    if buffer_size > 0:
        merged = merged.buffer(buffer_size).buffer(-buffer_size)
        merged = merged.simplify(0)  # убираем возможные артефакты

    # 4. Удаляем мелкие полигоны
    if min_area > 0:
        if hasattr(merged, 'geoms'):
            kept = [g for g in merged.geoms if g.area >= min_area]
            merged = type(merged)(kept) if kept else merged.__class__()
        elif merged.area < min_area:
            merged = merged.__class__()

    # 5. Упрощение контура
    if simplify_tolerance > 0:
        merged = merged.simplify(simplify_tolerance, preserve_topology=True)

    # Разбиваем обратно на отдельные полигоны для удобства
    if hasattr(merged, 'geoms'):
        geometries = [g for g in merged.geoms if not g.is_empty]
    else:
        geometries = [merged] if not merged.is_empty else []

    logger.info(f"После обобщения: {len(geometries)} связных кластер(ов)")
    return gpd.GeoDataFrame({'geometry': geometries}, crs=gdf.crs)


def visualize_results(contour_gdf: gpd.GeoDataFrame,
                      water_gdf: gpd.GeoDataFrame,
                      filtered_water: gpd.GeoDataFrame,
                      simplified_water: gpd.GeoDataFrame = None,
                      output_file: str = "water_in_contour.png"):
    """
    Визуализирует контур и водные объекты.

    Неупрощённые объекты выводятся с alpha=0.1, упрощённые — поверх них.

    Args:
        contour_gdf: GeoDataFrame с контуром
        water_gdf: GeoDataFrame со всеми водными объектами
        filtered_water: GeoDataFrame с отфильтрованными (неупрощёнными) объектами
        simplified_water: GeoDataFrame с упрощёнными объектами (опционально)
        output_file: Имя файла для сохранения визуализации
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Неупрощённые объекты — полупрозрачные (alpha=0.1)
    if not water_gdf.empty:
        water_gdf.plot(ax=ax, color='lightblue', edgecolor='blue',
                       alpha=0.1, label='Все объекты (неупрощённые)')

    contour_gdf.plot(ax=ax, facecolor='none', edgecolor='red',
                     linewidth=2, linestyle='--', label='Контур')

    # Неупрощённые отфильтрованные объекты — полупрозрачные (alpha=0.1)
    if not filtered_water.empty:
        filtered_water.plot(ax=ax, color='darkblue', edgecolor='navy',
                           alpha=0.1, linewidth=1, label='В контуре (неупрощённые)')

    # Упрощённые объекты — чётко поверх
    if simplified_water is not None and not simplified_water.empty:
        simplified_water.plot(ax=ax, color='darkblue', edgecolor='navy',
                              alpha=0.7, linewidth=2, label='В контуре (упрощённые)')

    bounds = contour_gdf.total_bounds
    ax.set_xlim(bounds[0] - 0.1, bounds[2] + 0.1)
    ax.set_ylim(bounds[1] - 0.1, bounds[3] + 0.1)

    ax.set_xlabel('Долгота')
    ax.set_ylabel('Широта')
    ax.set_title('Реки и водоёмы внутри контура')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    """
    Основная функция для загрузки и фильтрации рек и водоёмов по контуру.
    """
    parser = argparse.ArgumentParser(
        description="Загрузка рек и водоёмов по контуру"
    )
    parser.add_argument(
        '--buffer', type=float, default=0.0,
        help='Размер буфера в градусах для объединения близких полигонов '
             'и сглаживания рваных краёв (buffer-in/buffer-out). По умолчанию 0.'
    )
    parser.add_argument(
        '--min-area', type=float, default=0.0,
        help='Минимальная площадь полигона в кв. градусах; меньшие удаляются. '
             'По умолчанию 0 (не фильтровать).'
    )
    parser.add_argument(
        '--simplify', type=float, default=0.0,
        help='Допуск упрощения контура в градусах (алгоритм Дугласа-Пекера). '
             'По умолчанию 0 (не упрощать).'
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Не использовать кеш OSM данных, выполнить новый запрос к Overpass API'
    )
    parser.add_argument(
        '--plot', '-p', action='store_true',
        help='Показать график с контуром и водными объектами'
    )
    args = parser.parse_args()

    print("=== Поиск рек и водоёмов внутри контура ===\n")

    # 1. Загружаем контур
    contour_gdf, bounds = get_contour_bounds("contour.geojson")

    # 2. Загружаем водные объекты из OSM по bounding box контура
    water_gdf = download_water_from_osm(
        bounds[0], bounds[1], bounds[2], bounds[3],
        use_cache=not args.no_cache
    )

    if water_gdf.empty:
        logger.error("Не удалось загрузить данные о реках и водоёмах из OpenStreetMap")
        return

    # Убеждаемся, что данные в WGS84
    if water_gdf.crs != myLib.WGS84:
        logger.info(f"Преобразование водных объектов из {water_gdf.crs} в {myLib.WGS84}")
        water_gdf = water_gdf.to_crs(myLib.WGS84)

    # 3. Фильтруем по контуру
    filtered_water = filter_water_by_contour(water_gdf, contour_gdf)

    # 3.1 Обобщаем геометрии, если задан хотя бы один параметр
    use_generalize = (args.buffer > 0 or args.min_area > 0 or args.simplify > 0)
    simplified_water = None
    if use_generalize:
        simplified_water = generalize_geometries(
            filtered_water,
            buffer_size=args.buffer,
            min_area=args.min_area,
            simplify_tolerance=args.simplify
        )

    # 4. Визуализируем результаты (только если запрошено)
    if args.plot:
        visualize_results(contour_gdf, water_gdf, filtered_water, simplified_water)

    # 5. Сохраняем результаты
    # Если включено упрощение — сохраняем упрощённые, иначе исходные
    result_water = simplified_water if simplified_water is not None else filtered_water

    if not result_water.empty:
        output_file = "water_contour.geojson"
        if result_water.crs != myLib.WGS84:
            result_water = result_water.to_crs(myLib.WGS84)
        result_water.to_file(output_file, driver='GeoJSON')
        logger.info(f"Реки и водоёмы внутри контура сохранены в {output_file} (CRS: {myLib.WGS84})")
    else:
        print("\nРеки и водоёмы внутри указанного контура не найдены.")
        print("Возможно, в данном регионе нет водных объектов или данные OSM неполны.")


if __name__ == "__main__":
    main()
