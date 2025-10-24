import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import shapely

from matplotlib.widgets import CheckButtons

# from shapely.geometry.multipolygon import MultiPolygon
# from shapely.geometry.polygon import Polygon
# from shapely.ops import snap, unary_union


def get_color_gradient(n, start_rgb, end_rgb):
    """Возвращает список цветов от начального RGB к конечному RGB."""
    return [np.array(start_rgb) + (np.array(end_rgb) - np.array(start_rgb)) * i / (n - 1) for i in range(n)]


def simplify_geometry(edges, tolerance=1.0):
    """Упрощает геометрию линий в GeoDataFrame с заданным допуском."""
    edges['geometry'] = edges['geometry'].simplify(tolerance)
    return edges


def plot_highways(region_name):
    # Загрузка графа дорог для области
    G = ox.graph_from_place(region_name, network_type='drive', simplify=False)

    # Преобразование графа в GeoDataFrame
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    print(f"Точек {shapely.get_num_coordinates(edges.geometry)}")
    # Упрощение геометрии
    edges = simplify_geometry(edges, 500)
    print(f'Точек {shapely.get_num_coordinates(edges.geometry)}')

    # Настройка отображения графика
    fig, ax = plt.subplots(figsize=(12, 12))

    highway_info = {
        'motorway': 2.5,
        'trunk': 2.2,
        'primary': 1.9,
        'secondary': 1.6,
        'tertiary': 1.3,
        'residential': 1.0,
        'unclassified': 0.7,
        'service': 0.4,
    }

    types = list(highway_info.keys())
    colors = get_color_gradient(len(types), (0, 1, 0), (1, 0, 0))

    # Отображение дорог с учетом толщины и цвета
    lines = {}
    for highway_type, color in zip(highway_info.keys(), colors):
        width = highway_info[highway_type]
        mask = edges['highway'].map(lambda x: highway_type in x if isinstance(x, list) else highway_type == x)
        subset = edges[mask]

        if not subset.empty:
            collection = subset.plot(ax=ax, linewidth=width, edgecolor=color, label=highway_type)
            lines[highway_type] = collection

    rax = plt.axes([0.02, 0.3, 0.1, 0.3])
    labels = list(lines.keys())
    visibility = [collection.get_visible() for collection in lines.values()]
    check = CheckButtons(rax, labels, visibility)

    def func(label):
        idx = list(lines.keys()).index(label)
        ax.collections[idx].set_visible(not ax.collections[idx].get_visible())
        plt.draw()

    check.on_clicked(func)

    plt.legend()
    plt.show()


# Запуск функции с указанием названия региона
# plot_highways('Smolensk Oblast, Russia')
plot_highways('Kaliningrad region, Russia')
