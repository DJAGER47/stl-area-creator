# -*- coding: utf-8 -*-

"""
Main module for Copernicus DEM data access.
Provides a unified interface similar to SRTM module.
"""

import os as mod_os
import os.path as mod_path

from . import data as mod_data
from . import utils as mod_utils

from typing import *

# Copernicus DEM доступен через Copernicus Data Space
# Для загрузки требуется использовать API или прямые ссылки на файлы
COPERNICUS_BASE_URL = 'https://elevation.s3.amazonaws.com/'
# Альтернативный источник: https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/

def get_data(file_handler: Optional[mod_utils.FileHandler] = None,
             local_cache_dir: str = "", timeout: int = 0) -> mod_data.GeoElevationData:
    """
    Get the utility object for querying elevation data from Copernicus DEM.

    All data files will be stored locally (note that it may be
    gigabytes of data so clean it from time to time).

    On first run, files will be downloaded from the Copernicus DEM S3 bucket.
    For every elevation query, if the file is not found locally, it will be
    retrieved and saved.

    Args:
        file_handler: Custom file handler for saving/loading files.
                     If None, default FileHandler will be used.
        local_cache_dir: Directory for local cache. If empty, default
                        ~/.cache/copernicus-dem will be used.
        timeout: Timeout for HTTP requests in seconds. 0 means default timeout.

    Returns:
        GeoElevationData object for querying elevations.

    Example:
        >>> from copernicus import get_data
        >>> elevation_data = get_data()
        >>> elevation = elevation_data.get_elevation(55.7558, 37.6173)  # Moscow
        >>> print(f"Elevation: {elevation} meters")
    """
    if not file_handler:
        file_handler = mod_utils.FileHandler(local_cache_dir)

    return mod_data.GeoElevationData(file_handler=file_handler, timeout=timeout)
