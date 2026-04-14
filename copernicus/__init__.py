# -*- coding: utf-8 -*-

"""
Copernicus DEM (GLO-30) - Global Digital Elevation Model
Модуль для работы с данными высот Copernicus DEM с разрешением 30 м
"""

from .main import get_data
from .data import GeoElevationData

__all__ = ['get_data', 'GeoElevationData']
