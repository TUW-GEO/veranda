import pytest
from tempfile import mkdtemp


@pytest.fixture
def tmp_path():
    return mkdtemp()
