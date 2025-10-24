import geojson
from shapely.geometry import Point, mapping
from shapely.geometry.polygon import Polygon
import math
from pyproj import Transformer

WGS84 = 'EPSG:4326'
UTM = 'EPSG:32646'

def circle_points(cx, cy, radius, num_points):
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    return points

def create_circle(radius, lat, lon):
    wgs2utm = Transformer.from_crs(WGS84, UTM, always_xy=True)
    utm2wgs = Transformer.from_crs(UTM, WGS84, always_xy=True)

    centr = wgs2utm.transform(lon, lat)
    points_utm = circle_points(centr[0], centr[1] , radius, 1000)
    points_wgs = [utm2wgs.transform(x, y) for (x, y) in points_utm]
    
    return Polygon(points_wgs)

def save_geojson(geometry, filename="circle.geojson"):
    """Сохраняет геометрию в файл GeoJSON."""
    feature = geojson.Feature(geometry=mapping(geometry))
    feature_collection = geojson.FeatureCollection([feature])
    with open(filename, "w") as f:
        geojson.dump(feature_collection, f)
    print(f"Контур сохранён в {filename}.")

def main():
    lat = 43.348397
    lon = 42.454421
    radius = 50000
    
    circle = create_circle(radius, lat, lon)
    save_geojson(circle)

if __name__ == "__main__":
    main()