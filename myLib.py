

    # # -----------------------------------------------------------------
    # #  STL
    # # -----------------------------------------------------------------
    # list_stl = make_stl_obl(utm_contour, utm_mesh, utm_contour_zero)
    # # info_city = get_city_coordinates(name_oblast)
    # # stl_city = make_city_stl(info_city, step_m)

    # # for i, obl in enumerate(list_stl):
    # #     if stl_city[i] is None:
    # #         obl.update_normals()
    # #         obl.save(f'{path_save}{obl_name}_{step_m}_{i}.stl')
    # #     else:
    # #         combined_mesh_data = np.concatenate([obl.data, stl_city[i].data])
    # #         combined_mesh = mesh.Mesh(combined_mesh_data)
    # #         combined_mesh.update_normals()
    # #         combined_mesh.save(f'{path_save}{obl_name}_{step_m}_{i}.stl')
    # for i, obl in enumerate(list_stl):
    #     obl.update_normals()
    #     obl.save(f'{path_save}{obl_name}_{step_m}_{i}.stl')


    # # combined_mesh_data = np.concatenate([list_stl[0].data] + [city.data for city in stl_city])
    # # combined_mesh = mesh.Mesh(combined_mesh_data)
    # # combined_mesh.update_normals()
    # # combined_mesh.save(f'{path_save}{obl_name}_{step_m}_{i}.stl')

    # print(f"All done | time {((time.perf_counter() - _start_time_total)/60):.2f}m")
    # return

def circle_points(cx, cy, radius, num_points):
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    return points


def get_city_coordinates(obl_name):
    # Инициализация Overpass и Nominatim
    overpass = Overpass()
    nominatim = Nominatim()

    # Находим ID области при помощи Nominatim
    # - admin_level=2 - страна
    # - admin_level=4 - регион/область
    # - admin_level=6 - район
    # - admin_level=8 - поселения.
    area = nominatim.query(obl_name, featuretype='relation', adminLevel=4) # используйте название вашей области
    areaId = area.areaId()

    # Создаем запрос для Overpass API
    query = overpassQueryBuilder(
        area=areaId,
        elementType='node',
        selector='place~"city|town"',
        includeGeometry=False
    )

    # Выполняем запрос и обрабатываем результат
    result = overpass.query(query)

    # Извлечение и печать всех найденных городов
    print(f"Количество районов {len(result.elements())}")
    # for element in result.elements():
    #     print(element.tags().get('name'))  # Отображаем название
    return result.elements()


def make_city_stl(city_data, step):
    wgs2utm = Transformer.from_crs(WGS84, UTM, always_xy=True)
    utm2wgs = Transformer.from_crs(UTM, WGS84, always_xy=True)

    stl_city = list()
    for city in city_data:
        print(city.tags().get('name'))  # Отображаем название

        centr = wgs2utm.transform(city.lon(), city.lat())
        points_utm = circle_points(centr[0], centr[1] , 5000, 20)
        points_wgs = [utm2wgs.transform(x, y) for (x, y) in points_utm]
        circle = GetHeight(points_wgs, step)
        circle = [(x1, y1, h2) for (x1, y1), (x2, y2, h2) in zip(points_utm, circle)]

        max_z = max(circle, key=lambda x: x[2])[2]
        circle_bot = np.array(scale_elevation([(x, y, 0) for (x, y, _) in circle]))
        circle_top = np.array(scale_elevation([(x, y, max_z + 100) for (x, y, _) in circle]))

        tri = Delaunay(circle_bot[:, :2])

        bot_mesh = mesh.Mesh(np.zeros(tri.simplices.shape[0], dtype=mesh.Mesh.dtype))
        top_mesh = mesh.Mesh(np.zeros(tri.simplices.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(tri.simplices):
            for j in range(3):
                bot_mesh.vectors[i][j] = circle_bot[f[j], :]
                top_mesh.vectors[i][j] = circle_top[f[j], :]

        wall_mesh = wall_stl(circle_bot, circle_top)
        stl_city.append(combined_stl(top_mesh, wall_mesh, bot_mesh))

    return stl_city