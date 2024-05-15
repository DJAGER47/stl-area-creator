# import os

# import elevation
# import numpy as np
# import richdem as rd


# def get_elevation(lat, lon):
#     # Задать имя файла для хранения данных о высоте
#     cache_dir = 'srtm-cache'
#     os.makedirs(cache_dir, exist_ok=True)
#     dem_path = os.path.join(cache_dir, 'elevation.tif')

#     # Загрузить данные SRTM для заданной области
#     elevation.clip(bounds=(lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05), output=dem_path)

#     # # Загрузить DEM с помощью RichDEM и получить высоту в конкретной точке
#     # dem = rd.LoadGDAL(dem_path)
#     # # Преобразовать широту и долготу в индексы массива
#     # scale = 1 / (360 / dem.shape[0])  # Определение масштаба на основе размера области охвата данных
#     # x_index = int((lon + 180) * scale)
#     # y_index = int((90 - lat) * scale)
#     # elevation = dem[y_index, x_index]

#     return elevation


# # Пример использования:
# latitude = 48.8588443  # Пример для координат близ Эйфелевой башни
# longitude = 2.2943506
# print(f"Elevation at Eiffel Tower: {get_elevation(latitude, longitude)} meters")


# import requests


# def get_elevation(lat, lon):
#     url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
#     response = requests.get(url)
#     data = response.json()
#     return data['results'][0]['elevation']


# # Пример запроса
# latitude = 40.7128
# longitude = -74.0060
# elevation = get_elevation(latitude, longitude)
# print(f"The elevation at New York City is {elevation} meters.")
