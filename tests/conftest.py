import sys
from pathlib import Path

import pytest

sys.path.append((Path(__file__).parent / "raster" / "mosaic").as_posix())
sys.path.append((Path(__file__).parent / "raster" / "native" / "netcdf").as_posix())