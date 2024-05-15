#!/bin/bash

# obl=Smolensk
# obl=Kabardin-Balkar
# obl=Karachay-Cherkess
obl=Sakha
# obl=Bryansk

step=500

# python 1_contour.py $obl
python 2_mesh.py $obl $step
python 3_height.py $obl $step
python 4_stl.py $obl $step
