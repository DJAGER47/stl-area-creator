# -*- coding: utf-8 -*-

"""
Utility functions for Copernicus DEM module.
"""

import logging as mod_logging
import os as mod_os
import os.path as mod_path
import pathlib as mod_pathlib
from typing import *

DEFAULT_TIMEOUT = 30

class FileHandler:
    """
    File handler for caching Copernicus DEM files locally.
    Similar to SRTM FileHandler but for Copernicus DEM.
    """

    def __init__(self, local_cache_dir: Optional[str] = None) -> None:
        if local_cache_dir:
            self.local_cache_dir = local_cache_dir
        else:
            home_dir = str(mod_pathlib.Path.home()) or mod_os.environ.get("HOME") or mod_os.environ.get("HOMEPATH") or ""
            if not home_dir:
                raise Exception('No default HOME directory found')
            self.local_cache_dir = mod_os.sep.join([home_dir, '.cache', 'copernicus-dem'])

        if not mod_path.exists(self.local_cache_dir):
            mod_logging.info(f"Creating {self.local_cache_dir}")
            try:
                mod_os.makedirs(self.local_cache_dir)
            except Exception as e:
                mod_logging.error(f"Local cache dir: {self.local_cache_dir}")
                raise Exception(f"Error creating directory {self.local_cache_dir}: {e}")

    def exists(self, file_name: str) -> bool:
        """Check if file exists in cache."""
        return mod_path.exists(mod_os.path.join(self.local_cache_dir, file_name))

    def write(self, file_name: str, contents: bytes) -> None:
        """Write file to cache. Creates subdirectories if needed."""
        fn = mod_os.path.join(self.local_cache_dir, file_name)
        
        # Create directory if it doesn't exist
        directory = mod_path.dirname(fn)
        if directory and not mod_path.exists(directory):
            try:
                mod_os.makedirs(directory, exist_ok=True)
                mod_logging.debug(f"Created directory: {directory}")
            except Exception as e:
                mod_logging.error(f"Error creating directory {directory}: {e}")
                raise
        
        with open(fn, 'wb') as f:
            n = f.write(contents)
            mod_logging.debug(f"Saved {n} bytes in {fn}")

    def read(self, file_name: str) -> bytes:
        """Read file from cache."""
        with open(mod_os.path.join(self.local_cache_dir, file_name), 'rb') as f:
            return f.read()
