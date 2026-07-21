# -*- coding: utf-8 -*-
"""
Скрипт для загрузки ледников и фильтрации по контуру из contour.geojson.

Источником данных служит OpenStreetMap (Overpass API). Загружаются:
  - ледники: natural=glacier

Результат сохраняется в glaciers_contour.geojson в проекции WGS84.
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
    """
    Формирует путь к кеш-файлу на основе bounding box.

    Args:
        prefix: Префикс имени файла (напр. 'glaciers_osm')
        min_lon, min_lat, max_lon, max_lat: Границы bbox

    Returns:
        Путь к файлу кеша
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    # Округляем до 4 знаков для читаемости
    bbox_str = f"{min_lon:.4f}_{min_lat:.4f}_{max_lon:.4f}_{max_lat:.4f}"
    return os.path.join(cache_dir, f"{prefix}_{bbox_str}.geojson")


def _load_from_cache(cache_path: str) -> gpd.GeoDataFrame | None:
    """
    Пытается загрузить данные из кеш-файла.

    Args:
        cache_path: Путь к файлу кеша

    Returns:
        GeoDataFrame или None, если кеш не найден
    """
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
    """
    Сохраняет данные в кеш-файл.

    Args:
        gdf: GeoDataFrame для сохранения
        cache_path: Путь к файлу кеша
    """
    logger.info(f"Сохранение в кеш: {cache_path}")
    try:
        if gdf.crs is None:
            gdf.set_crs(myLib.WGS84, inplace=True)
        gdf.to_file(cache_path, driver='GeoJSON')
    except Exception as e:
        logger.warning(f"Не удалось сохранить кеш {cache_path}: {e}")


def download_glaciers_from_osm(min_lon: float, min_lat: float,
                               max_lon: float, max_lat: float,
                               use_cache: bool = True) -> gpd.GeoDataFrame:
    """
    Загружает ледники из OpenStreetMap через Overpass API по bounding box.

    Args:
        min_lon: Минимальная долгота
        min_lat: Минимальная широта
        max_lon: Максимальная долгота
        max_lat: Максимальная широта
        use_cache: Использовать кеш (по умолчанию True)

    Returns:
        GeoDataFrame с ледниками (WGS84)
    """
    cache_path = _bbox_cache_path("glaciers_osm", min_lon, min_lat, max_lon, max_lat)

    # Пробуем загрузить из кеша
    if use_cache:
        cached = _load_from_cache(cache_path)
        if cached is not None:
            return cached

    logger.info("Загрузка ледников из OpenStreetMap (Overpass API)")

    overpass = Overpass()

    # Запрос: ледники внутри bounding box.
    # Заголовок [out:json][timeout:...] добавляет сам OSMPythonTools.
    query = f"""
    (
      way["natural"="glacier"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["natural"="glacier"]({min_lat},{min_lon},{max_lat},{max_lon});
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
                'natural': tags.get('natural', ''),
                'glacier_type': tags.get('glacier:type', ''),
            }
            features.append({'type': 'Feature', 'properties': props,
                             'geometry': mapping(geom)})
        except Exception as e:
            logger.warning(f"Не удалось обработать элемент {element.id()}: {e}")
            continue

    if not features:
        logger.warning("Не найдено ледников в OSM для заданного bbox")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame.from_features(features, crs=myLib.WGS84)
    logger.info(f"Сформировано {len(gdf)} ледников")

    # Сохраняем в кеш
    if use_cache:
        _save_to_cache(gdf, cache_path)

    return gdf


def filter_glaciers_by_contour(glaciers_gdf: gpd.GeoDataFrame,
                               contour_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Фильтрует ледники, оставляя только те, что пересекают контур,
    и обрезает их по границам контура.

    Args:
        glaciers_gdf: GeoDataFrame с ледниками
        contour_gdf: GeoDataFrame с контуром

    Returns:
        GeoDataFrame с отфильтрованными и обрезанными ледниками
    """
    logger.info("Фильтрация ледников по контуру")

    contour_geometry = contour_gdf.geometry.iloc[0]

    intersecting = glaciers_gdf[glaciers_gdf.intersects(contour_geometry)]
    logger.info(f"Найдено {len(intersecting)} ледников, пересекающих контур")

    filtered = intersecting.copy()
    filtered['geometry'] = intersecting['geometry'].intersection(contour_geometry)

    # Удаляем пустые геометрии
    filtered = filtered[~filtered.geometry.is_empty]
    filtered = filtered[~filtered.geometry.isna()]

    logger.info(f"После обрезки: {len(filtered)} ледников внутри контура")
    return filtered


def generalize_geometries(gdf: gpd.GeoDataFrame,
                         buffer_size: float = 0.0,
                         min_area: float = 0.0,
                         simplify_tolerance: float = 0.0,
                         smooth_iterations: int = 0) -> gpd.GeoDataFrame:
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
      6. Сглаживание Chaikin (smooth_iterations) — дополнительные итерации
         для получения более гладких кривых.

    Args:
        gdf: GeoDataFrame с объектами (WGS84)
        buffer_size: Размер буфера в градусах для объединения/сглаживания
        min_area: Минимальная площадь полигона в квадратных градусах;
                  меньшие удаляются
        simplify_tolerance: Допуск упрощения в градусах (0 — не упрощать)
        smooth_iterations: Количество итераций сглаживания Chaikin (0 — не сглаживать)

    Returns:
        GeoDataFrame с обобщёнными геометриями (одна строка на связный кластер)
    """
    logger.info(f"Обобщение геометрий: buffer={buffer_size}, "
                f"min_area={min_area}, simplify={simplify_tolerance}, "
                f"smooth={smooth_iterations}")

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

    # 6. Сглаживание Chaikin (несколько итераций)
    if smooth_iterations > 0:
        merged = smooth_geometry_chaikin(merged, iterations=smooth_iterations)

    # Разбиваем обратно на отдельные полигоны для удобства
    if hasattr(merged, 'geoms'):
        geometries = [g for g in merged.geoms if not g.is_empty]
    else:
        geometries = [merged] if not merged.is_empty else []

    logger.info(f"После обобщения: {len(geometries)} связных кластер(ов)")
    return gpd.GeoDataFrame({'geometry': geometries}, crs=gdf.crs)


def smooth_geometry_chaikin(geometry, iterations=3):
    """
    Сглаживает геометрию алгоритмом Chaikin.
    На каждой итерации каждая вершина заменяется двумя точками,
    лежащими на 1/4 и 3/4 отрезка между соседними вершинами.
    Это даёт гладкую кривую, сохраняющую общую форму.

    Args:
        geometry: Shapely геометрия (Polygon или MultiPolygon)
        iterations: Количество итераций сглаживания

    Returns:
        Сглаженная Shapely геометрия
    """
    from shapely.geometry import Polygon, MultiPolygon

    def _smooth_ring(ring_coords):
        """Сглаживает одно кольцо (список координат)."""
        coords = list(ring_coords)
        if len(coords) < 4:
            return coords

        for _ in range(iterations):
            new_coords = []
            n = len(coords) - 1  # последняя точка = первой (замкнутое кольцо)
            for i in range(n):
                p0 = coords[i]
                p1 = coords[(i + 1) % n]
                # Точка на 1/4 отрезка
                qx = 0.75 * p0[0] + 0.25 * p1[0]
                qy = 0.75 * p0[1] + 0.25 * p1[1]
                # Точка на 3/4 отрезка
                rx = 0.25 * p0[0] + 0.75 * p1[0]
                ry = 0.25 * p0[1] + 0.75 * p1[1]
                new_coords.append((qx, qy))
                new_coords.append((rx, ry))
            # Замыкаем кольцо
            new_coords.append(new_coords[0])
            coords = new_coords

        return coords

    def _smooth_polygon(poly):
        """Сглаживает один полигон."""
        exterior = _smooth_ring(poly.exterior.coords)
        interiors = [_smooth_ring(ring.coords) for ring in poly.interiors]
        return Polygon(exterior, interiors)

    if hasattr(geometry, 'geoms'):
        # MultiPolygon
        smoothed = [_smooth_polygon(g) for g in geometry.geoms if not g.is_empty]
        return MultiPolygon(smoothed) if smoothed else geometry
    else:
        # Polygon
        return _smooth_polygon(geometry)


def visualize_results(contour_gdf: gpd.GeoDataFrame,
                      glaciers_gdf: gpd.GeoDataFrame,
                      filtered_glaciers: gpd.GeoDataFrame,
                      simplified_glaciers: gpd.GeoDataFrame = None,
                      output_file: str = "glaciers_in_contour.png"):
    """
    Визуализирует контур и ледники.

    Неупрощённые объекты выводятся с alpha=0.1, упрощённые — поверх них.

    Args:
        contour_gdf: GeoDataFrame с контуром
        glaciers_gdf: GeoDataFrame со всеми ледниками
        filtered_glaciers: GeoDataFrame с отфильтрованными (неупрощёнными) ледниками
        simplified_glaciers: GeoDataFrame с упрощёнными ледниками (опционально)
        output_file: Имя файла для сохранения визуализации
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Неупрощённые объекты — полупрозрачные (alpha=0.1)
    if not glaciers_gdf.empty:
        glaciers_gdf.plot(ax=ax, color='lightblue', edgecolor='blue',
                         alpha=0.1, label='Все ледники (неупрощённые)')

    contour_gdf.plot(ax=ax, facecolor='none', edgecolor='red',
                     linewidth=2, linestyle='--', label='Контур')

    # Неупрощённые отфильтрованные объекты — полупрозрачные (alpha=0.1)
    if not filtered_glaciers.empty:
        filtered_glaciers.plot(ax=ax, color='darkblue', edgecolor='navy',
                              alpha=0.1, linewidth=1, label='В контуре (неупрощённые)')

    # Упрощённые объекты — чётко поверх
    if simplified_glaciers is not None and not simplified_glaciers.empty:
        simplified_glaciers.plot(ax=ax, color='darkblue', edgecolor='navy',
                                alpha=0.7, linewidth=2, label='В контуре (упрощённые)')

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
        description="Загрузка ледников по контуру из OpenStreetMap"
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
        '--smooth', type=int, default=0,
        help='Количество итераций сглаживания Chaikin для получения плавных '
             'кривых. По умолчанию 0 (не сглаживать). Рекомендуется 2-5.'
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Не использовать кеш OSM данных, выполнить новый запрос к Overpass API'
    )
    parser.add_argument(
        '--plot', '-p', action='store_true',
        help='Показать график с контуром и ледниками'
    )
    args = parser.parse_args()

    print("=== Поиск ледников внутри контура (OpenStreetMap) ===\n")

    # 1. Загружаем контур
    contour_gdf, bounds = get_contour_bounds("contour.geojson")

    # 2. Загружаем ледники из OSM по bounding box контура
    glaciers_gdf = download_glaciers_from_osm(
        bounds[0], bounds[1], bounds[2], bounds[3],
        use_cache=not args.no_cache
    )

    if glaciers_gdf.empty:
        logger.error("Не удалось загрузить данные о ледниках из OpenStreetMap")
        return

    # Убеждаемся, что данные в WGS84
    if glaciers_gdf.crs != myLib.WGS84:
        logger.info(f"Преобразование ледников из {glaciers_gdf.crs} в {myLib.WGS84}")
        glaciers_gdf = glaciers_gdf.to_crs(myLib.WGS84)

    # 3. Фильтруем по контуру
    filtered_glaciers = filter_glaciers_by_contour(glaciers_gdf, contour_gdf)

    # 3.1 Обобщаем геометрии, если задан хотя бы один параметр
    use_generalize = (args.buffer > 0 or args.min_area > 0 or args.simplify > 0 or args.smooth > 0)
    simplified_glaciers = None
    if use_generalize:
        simplified_glaciers = generalize_geometries(
            filtered_glaciers,
            buffer_size=args.buffer,
            min_area=args.min_area,
            simplify_tolerance=args.simplify,
            smooth_iterations=args.smooth
        )

    # 4. Визуализируем результаты (только если запрошено)
    if args.plot:
        visualize_results(contour_gdf, glaciers_gdf, filtered_glaciers, simplified_glaciers)

    # 5. Сохраняем результаты
    # Если включено упрощение — сохраняем упрощённые, иначе исходные
    result_glaciers = simplified_glaciers if simplified_glaciers is not None else filtered_glaciers

    if not result_glaciers.empty:
        output_file = "glaciers_contour.geojson"
        if result_glaciers.crs != myLib.WGS84:
            result_glaciers = result_glaciers.to_crs(myLib.WGS84)
        result_glaciers.to_file(output_file, driver='GeoJSON')
        logger.info(f"Ледники внутри контура сохранены в {output_file} (CRS: {myLib.WGS84})")
    else:
        print("\nЛедники внутри указанного контура не найдены.")
        print("Возможно, в данном регионе нет ледников или данные OSM неполны.")


if __name__ == "__main__":
    main()
