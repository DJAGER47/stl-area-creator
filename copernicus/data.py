# -*- coding: utf-8 -*-

"""
Classes containing parsed elevation data from Copernicus DEM.
"""

import logging as mod_logging
import math as mod_math
import struct as mod_struct
import requests as mod_requests
import numpy as np
from typing import *

from . import utils as mod_utils

class GeoElevationFile:
    """
    Represents a single Copernicus DEM tile (1°×1°).
    """

    def __init__(self, file_name: str, data: bytes, geo_elevation_data: "GeoElevationData") -> None:
        self.file_name = file_name
        self.data = data
        self.geo_elevation_data = geo_elevation_data

        # Copernicus DEM GLO-30 has 3600×3600 pixels per 1°×1° tile
        self.size = 3600

        # Parse the data
        self._parse_data()

    def _parse_data(self) -> None:
        """
        Parse GeoTIFF data from Copernicus DEM.
        Copernicus DEM is in GeoTIFF format, not HGT format like SRTM.
        """
        try:
            # Try to use rasterio if available
            import rasterio
            from io import BytesIO

            with rasterio.open(BytesIO(self.data)) as src:
                # Read the first band (elevation)
                self.elevations = src.read(1)
                self.transform = src.transform
                self.crs = src.crs
                self.nodata = src.nodata

                # Replace nodata values with 0
                if self.nodata is not None:
                    self.elevations[self.elevations == self.nodata] = 0

        except ImportError:
            # Fallback: try to parse as raw data (simplified)
            mod_logging.warning('rasterio not available, using simplified parsing')
            # This is a simplified fallback - in production, use rasterio
            self.elevations = np.zeros((self.size, self.size), dtype=np.int16)
            self.transform = None
            self.crs = None
            self.nodata = None

    def get_elevation(self, latitude: float, longitude: float, approximate: bool = False) -> Optional[float]:
        """
        Get elevation at the specified latitude and longitude.
        """
        if self.transform is None:
            return None

        # Convert lat/lon to pixel coordinates
        # Copernicus DEM uses standard GeoTIFF transform
        lon_pixel, lat_pixel = ~self.transform * (longitude, latitude)
        lon_pixel = int(lon_pixel)
        lat_pixel = int(lat_pixel)

        # Check bounds using actual array dimensions
        height, width = self.elevations.shape
        if lon_pixel < 0 or lon_pixel >= width or lat_pixel < 0 or lat_pixel >= height:
            return None

        elevation = self.elevations[lat_pixel, lon_pixel]

        if elevation == self.nodata:
            return None

        return float(elevation)

    def _InverseDistanceWeighted(self, latitude: float, longitude: float, radius: int = 1) -> Optional[float]:
        """
        Return the interpolated elevation at a point using IDW.
        """
        if self.transform is None:
            return None

        # Convert lat/lon to pixel coordinates
        lon_pixel, lat_pixel = ~self.transform * (longitude, latitude)

        # Collect nearby elevations
        elevations = []
        weights = []

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue

                px = int(lon_pixel) + dx
                py = int(lat_pixel) + dy

                height, width = self.elevations.shape
                if 0 <= px < width and 0 <= py < height:
                    elev = self.elevations[py, px]
                    if elev != self.nodata:
                        distance = mod_math.sqrt(dx*dx + dy*dy)
                        weight = 1.0 / (distance ** 2)
                        elevations.append(elev)
                        weights.append(weight)

        if not elevations:
            # If no nearby elevations, return the center point
            return self.get_elevation(latitude, longitude)

        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(e * w for e, w in zip(elevations, weights))

        return weighted_sum / total_weight


class GeoElevationData:
    """
    The main class with utility methods for elevations from Copernicus DEM.
    """

    def __init__(self, file_handler: mod_utils.FileHandler, timeout: int = 0) -> None:
        self.file_handler = file_handler
        self.timeout = timeout

        # Lazy loaded files used in current app:
        self.files: Dict[str, GeoElevationFile] = {}

        # Set of tiles that failed to download (to avoid repeated attempts)
        self.failed_tiles: Set[str] = set()

        # Copernicus DEM base URL
        # Используем публичный S3 бакет с данными
        self.base_url = 'https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/'

    def get_elevation(self, latitude: float, longitude: float, approximate: bool = False) -> Optional[float]:
        """
        Get elevation at the specified latitude and longitude.
        Returns 0 if the tile failed to download.
        """
        geo_elevation_file = self.get_file(float(latitude), float(longitude))

        if not geo_elevation_file:
            # Return 0 if tile failed to download
            return 0.0

        elevation = geo_elevation_file.get_elevation(float(latitude), float(longitude), approximate)
        if elevation is None:
            return 0.0
        return elevation

    def _IDW(self, latitude: float, longitude: float, radius: float = 1) -> Optional[float]:
        """
        Return the interpolated elevation at a point using IDW.
        """
        tile = self.get_file(latitude, longitude)
        if tile is None:
            return None
        return tile._InverseDistanceWeighted(latitude, longitude, radius)

    def get_file(self, latitude: float, longitude: float) -> Optional[GeoElevationFile]:
        """
        Get the elevation file for the specified coordinates.
        If the file can't be found locally, it will be retrieved from the server.
        """
        file_name = self.get_file_name(latitude, longitude)

        if not file_name:
            return None

        # Check if this tile has failed to download before
        if file_name in self.failed_tiles:
            mod_logging.warning(f'Tile {file_name} previously failed to download for coordinates ({latitude}, {longitude})')
            return None

        if file_name in self.files:
            return self.files[file_name]
        else:
            data = self.retrieve_or_load_file_data(file_name, latitude, longitude)
            if not data:
                return None

            result = GeoElevationFile(file_name, data, self)
            self.files[file_name] = result

            return result

    def retrieve_or_load_file_data(self, file_name: str, latitude: Optional[float] = None, longitude: Optional[float] = None) -> Optional[bytes]:
        """
        Retrieve file data from local cache or download from server.
        """
        # Try to load from local cache
        if self.file_handler.exists(file_name):
            return self.file_handler.read(file_name)

        # Construct URL
        url = f"{self.base_url}{file_name}"

        try:
            mod_logging.info(f'Downloading {url}')
            r = mod_requests.get(url, timeout=self.timeout or mod_utils.DEFAULT_TIMEOUT)
        except mod_requests.exceptions.Timeout:
            mod_logging.error(f'Connection to {url} failed (timeout) for coordinates ({latitude}, {longitude})')
            self.failed_tiles.add(file_name)
            return None
        except Exception as e:
            mod_logging.error(f'Failed to download {url} for coordinates ({latitude}, {longitude}): {e}')
            self.failed_tiles.add(file_name)
            return None

        if r.status_code < 200 or r.status_code >= 300:
            mod_logging.error(f'Cannot retrieve {url} (status: {r.status_code}) for coordinates ({latitude}, {longitude})')
            self.failed_tiles.add(file_name)
            return None

        data = r.content
        mod_logging.info(f'Retrieved {url} ({len(data)} bytes)')

        if not data:
            mod_logging.error(f'Empty data received from {url} for coordinates ({latitude}, {longitude})')
            self.failed_tiles.add(file_name)
            return None

        # Save to local cache
        self.file_handler.write(file_name, data)

        return data

    def get_file_name(self, latitude: float, longitude: float) -> Optional[str]:
        """
        Generate the Copernicus DEM file name for the given coordinates.
        Format: Copernicus_DSM_COG_10_N00_00_E006_00_DEM/Copernicus_DSM_COG_10_N00_00_E006_00_DEM.tif
        """
        # Determine latitude direction
        if latitude >= 0:
            lat_dir = 'N'
        else:
            lat_dir = 'S'

        # Determine longitude direction
        if longitude >= 0:
            lon_dir = 'E'
        else:
            lon_dir = 'W'

        # Format: Copernicus_DSM_COG_10_N00_00_E006_00_DEM/Copernicus_DSM_COG_10_N00_00_E006_00_DEM.tif
        lat_deg = str(int(abs(mod_math.floor(latitude)))).zfill(2)
        lon_deg = str(int(abs(mod_math.floor(longitude)))).zfill(3)

        # Формат с подчеркиваниями и нулями
        lat_formatted = f"{lat_dir}{lat_deg}_00"
        lon_formatted = f"{lon_dir}{lon_deg}_00"

        folder_name = f'Copernicus_DSM_COG_10_{lat_formatted}_{lon_formatted}_DEM'
        file_name = f'{folder_name}/{folder_name}.tif'

        return file_name
