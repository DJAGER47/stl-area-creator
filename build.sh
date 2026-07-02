source .venv/bin/activate

python3 preparation/make_hexagon_contour.py --track path.gpx --rotation 20 --plot
python3 generate_ground.py -c contour.geojson -s 100

python3 generate_route.py --wpt-radius 700 --track-thickness 500 --add-height 150 --overlay 500

python3 preparation/make_glaciers_contour.py  --plot
python3 generate_ground.py -c glaciers_contour.geojson -s 100 -o 500 --add-height 100 

python3 preparation/make_forest_contour.py --buffer 0.003 --min-area 0.001 --simplify 0.001 --smooth 3 --plot
python3 generate_ground.py -c forest_contour.geojson -s 100 -o 500 --add-height 100 
