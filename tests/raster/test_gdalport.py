import os
import pytest
import numpy as np
from veranda.raster.gdalport import call_gdal_util
from veranda.raster.native.geotiff import GeoTiffFile


@pytest.fixture
def in_filepath(tmp_path):
    return os.path.join(tmp_path, "in_test.tif")


@pytest.fixture
def out_filepath(tmp_path):
    return os.path.join(tmp_path, "out_test.tif")


def test_gdal_translate(in_filepath, out_filepath):
    with GeoTiffFile(in_filepath, mode='w') as gt_file:
        gt_file.write(np.ones((1, 100, 100)))

    options = {'-of': 'GTiff', '-co': 'COMPRESS=LZW',
               '-mo': ['parent_data_file="%s"' % os.path.basename(in_filepath)], '-outsize': ('50%', '50%'),
               '-ot': 'Byte'}
    succeed, output = call_gdal_util("gdal_translate", src_files=in_filepath, dst_file=out_filepath,
                                     options=options)

    assert succeed
