import geopandas as gpd
import matplotlib.pyplot as plt
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from shapely.geometry import shape

# Инициализируем сервисы Overpass и Nominatim
overpass = Overpass()
nominatim = Nominatim()

output_filename = "лаптевых.geojson"
area = nominatim.query("лаптевых море", limit=1).toJSON()[0]

# output_filename = 'Карское.geojson'
# area = nominatim.query('карское море', limit=1).toJSON()[0]

# output_filename = 'barentzevo_sea_contour.geojson'
# area = nominatim.query('баренцево море', limit=1).toJSON()[0]

# output_filename = 'white_sea_contour.geojson'
# area = nominatim.query('белое море', limit=1).toJSON()[0]

# output_filename = 'fin_sea_contour.geojson'
# area = nominatim.query('финский залив', limit=1).toJSON()[0]

# output_filename = 'azov_sea_contour.geojson'
# area = nominatim.query('азовское море', limit=1).toJSON()[0]

# output_filename = 'kasp_sea_contour.geojson'
# area = nominatim.query('каспийское море', limit=1).toJSON()[0]

# output_filename = 'black_sea_contour.geojson'
# area = nominatim.query('черное море', limit=1).toJSON()[0]

osm_id = area["osm_id"]
osm_type = area['osm_type'] # "relation"
overpass_query = f"""
{osm_type}({osm_id});
out geom;
"""

result = overpass.query(overpass_query)
element = result.elements()[0]
geometry = element.geometry()

sea_shape = shape(geometry)
geo_df = gpd.GeoDataFrame({"geometry": [sea_shape]}, crs="EPSG:4326")
geo_df.to_file(output_filename, driver="GeoJSON")

geo_df.plot(figsize=(10, 10), edgecolor="blue", facecolor="lightblue")
plt.xlabel("Долгота")
plt.ylabel("Широта")
plt.show()
