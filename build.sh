source .venv/bin/activate

python3 preparation/make_hexagon_contour.py --track path.gpx --rotation 15 --scale 1.1 #--plot
python3 generate_ground.py -c contour.geojson -s 100 --shift-min 50

python3 generate_route.py --wpt-radius 700 --track-thickness 500 --overlay 500 --add-height 150 

# python3 preparation/make_glaciers_contour.py  #--plot
# python3 preparation/make_glaciers_contour_osm.py --buffer 0.003 --min-area 0.001 --simplify 0.001 --smooth 3 --plot
python3 preparation/make_glaciers_contour_osm.py --buffer 0.005 --min-area 0.001 #--plot
python3 generate_ground.py -c glaciers_contour.geojson -s 100 --overlay 500 --add-height 100 

# python3 preparation/make_forest_contour.py --buffer 0.003 --min-area 0.001 --simplify 0.001 --smooth 3 #--plot
python3 preparation/make_forest_contour.py --buffer 0.003 --min-area 0.001
python3 generate_ground.py -c forest_contour.geojson -s 100 --overlay 500 --add-height 100 
