import geopandas as gpd
import matplotlib.pyplot as plt
import lib


origing = gpd.read_file("data/gadm41_RUS.gpkg",
                        layer="ADM_ADM_1")  # ADM_ADM_0-3
# print(origing.columns)
# print(origing.head())
# print(origing["NAME_1"])
# print(origing["geometry"])

for i in range(0, len(origing["NAME_1"])):
    print(f'{i:02}\t{origing["NAME_1"][i]:<20}\t{origing["NL_NAME_1"][i]}')


# 00	Adygey              	Республика Адыгея
# 01	Altay               	Алтайский край
# 02	Amur                	Амурская область
# 03	Arkhangel'sk        	Архангельская область
# 04	Astrakhan'          	Астраханская область
# 05	Bashkortostan       	Республика Башкортостан
# 06	Belgorod            	Белгородская область
# 07	Bryansk             	Брянская область
# 08	Buryat              	Республика Бурятия
# 09	Chechnya            	Республика Чечено-Ингушская
# 10	Chelyabinsk         	Челябинская область
# 11	Chukot              	Чукотский АОк
# 12	Chuvash             	Чувашская Республика
# 13	City of St. Petersburg	Санкт-Петербург (горсовет)
# 14	Dagestan            	Республика Дагестан
# 15	Gorno-Altay         	Республика Алтай
# 16	Ingush              	Респу́блика Ингуше́тия
# 17	Irkutsk             	Иркутская область
# 18	Ivanovo             	Ивановская область
# 19	Kabardin-Balkar     	Кабардино-Балкарская Республика
# 20	Kaliningrad         	Калининградская область
# 21	Kalmyk              	Республика Калмыкия
# 22	Kaluga              	Калужская область
# 23	Kamchatka           	Камчатская край
# 24	Karachay-Cherkess   	Карачаево-Черкессия Республика
# 25	Karelia             	Республика Карелия
# 26	Kemerovo            	Кемеровская область
# 27	Khabarovsk          	Хабаровский край
# 28	Khakass             	Республика Хакасия
# 29	Khanty-Mansiy       	Ханты-Мансийский АОк
# 30	Kirov               	Кировская область
# 31	Komi                	Республика Коми
# 32	Kostroma            	Костромская область
# 33	Krasnodar           	Краснодарский край
# 34	Krasnoyarsk         	Красноярский край
# 35	Kurgan              	Курганская область
# 36	Kursk               	Курская область
# 37	Leningrad           	Ленинградская область
# 38	Lipetsk             	Липецкая область
# 39	Magadan             	Магадан|Магаданская область
# 40	Mariy-El            	Республика Марий Эл
# 41	Mordovia            	Республика Мордовия
# 42	Moscow City         	NA
# 43	Moskva              	Московская область
# 44	Murmansk            	Мурманская область
# 45	Nenets              	Ненецкий АОк
# 46	Nizhegorod          	Нижегородская область
# 47	North Ossetia       	Республика Северная Осетия-Алани
# 48	Novgorod            	Новгородская область
# 49	Novosibirsk         	Новосибирская область
# 50	Omsk                	Омская область
# 51	Orel                	Орловская область
# 52	Orenburg            	Оренбургская область
# 53	Penza               	Пензенская область
# 54	Perm'               	Пермская край
# 55	Primor'ye           	Приморский край
# 56	Pskov               	Псковская область
# 57	Rostov              	Ростовская область
# 58	Ryazan'             	Рязанская область
# 59	Sakha               	Республика Саха
# 60	Sakhalin            	Сахалинская область
# 61	Samara              	Самарская область
# 62	Saratov             	Саратовская область
# 63	Smolensk            	Смоленская область
# 64	Stavropol'          	Ставропольский край
# 65	Sverdlovsk          	Свердловская область
# 66	Tambov              	Тамбовская область
# 67	Tatarstan           	Республика Татарстан
# 68	Tomsk               	Томская область
# 69	Tula                	Тульская область
# 70	Tuva                	Республика Тыва
# 71	Tver'               	Тверская область
# 72	Tyumen'             	Тюменская область
# 73	Udmurt              	Удмуртская Республика
# 74	Ul'yanovsk          	Ульяновская область
# 75	Vladimir            	Владимирская область
# 76	Volgograd           	Волгоградская область
# 77	Vologda             	Вологодская область
# 78	Voronezh            	Воронежская область
# 79	Yamal-Nenets        	Ямало-Ненецкий АОк
# 80	Yaroslavl'          	Ярославская область
# 81	Yevrey              	Eврейская АОб
# 82	Zabaykal'ye         	Забайкальский край
