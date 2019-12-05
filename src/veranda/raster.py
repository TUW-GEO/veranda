import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from veranda.io.netcdf import NcFile
from veranda.io.geotiff import GeoTiffFile
from veranda.io.timestack import GeoTiffRasterTimeStack
from veranda.io.timestack import NcRasterTimeStack

from geospade.definition import RasterGeometry
from geospade.spatial_ref import SpatialRef

def _file_io_decorator(func):
    """
    Decorator which checks whether parsed file arguments are acutual IO-Classes
    such as NcFile or GeoTiffFile. If string is parsed as file, then
    a corresponding IO-Class object is parsed into the original function
    """
    supported_io_classes = (GeoTiffFile, NcFile)
    tiff_ext = ('.tiff', '.tif', '.geotiff')
    netcdf_ext = ('.nc', '.netcdf', '.ncdf')

    def wrapper(self, *args, **kwargs):
        file = args[1]
        if not isinstance(file, supported_io_classes):
            if file is None:
                # None => wraps write() => args[0] = self, get_orig_io
                file = args[0]._get_orig_io()
                pass
            else:
                # file is probably string
                format = os.path.splitext(file)[1].lower()
                if format in tiff_ext:
                    io_class = GeoTiffFile
                elif format in netcdf_ext:
                    io_class = NcFile
                else:
                    raise IOError('File format not supported.')
                if not isinstance(args[0], RasterData):
                    # wraps from_file()
                    io_class_kwargs = kwargs if kwargs is not None else {}
                    file = io_class(file, **io_class_kwargs)
                else:
                    # wraps write() with given file
                    touch(file)  # make sure file exists
                    file = io_class(file,
                                    mode='w',
                                    count=1,
                                    geotransform=args[0].geometry.gt,
                                    spatialref=args[0].geometry.wkt)

        return func(args[0], file)

    return wrapper


def get_stack_io_class(filepath):
    """
    """
    tiff_ext = ('.tiff', '.tif', '.geotiff')
    netcdf_ext = ('.nc', '.netcdf', '.ncdf')

    # file is probably string
    format = os.path.splitext(filepath)[1].lower()
    if format in tiff_ext:
        io_class = GeoTiffRasterTimeStack
    elif format in netcdf_ext:
        io_class = NcRasterTimeStack
    else:
        raise IOError('File format not supported.')

    return io_class

class RasterData(object):
    """
    This class represents georeferenced raster data. Two main components of this
    data structure is geometry and actual data. The geometry of the raster data,
    defines all geometrical properties of the data like extent, pixel size,
    location and orientation in a spatial reference system. The geometry is a
    RasterGeometry object which implements several geometrical operations.
    The other component is data, which is a simple numpy ndarray, that contains
    the actual values of the raster file. Every RasterData has its own instance
    of some IO-Class such as GeoTiffFile or NcFile which is used for IO
    operations. The advantage of this class is, that we can perform geometrical
    operations without even having to load the pixel values from hard-drive, and
    then later load only data we need.
    """

    def __init__(self, rows, cols, sref, gt,
                 data=None, parent=None, io=None):
        """
        Basic constructor

        Parameters
        ----------
        rows : int
            number of pixel rows
        cols : int
            number pixel columns
        sref: pyraster.spatial_ref.SpatialRef
            spatial reference of the data
        gt : 6-tuple
            GDAL Geotransform 'matrix'
        data : array-like (optional)
            Image pixel values (data)
        parent : RasterData ,optional (default: None)
            This attribute is used to trace back to the actual file for the
            io operations. For example by cropping existing RasterData object,
            that was created from existing file, a new RasterData object is
            created with parent set to the RasterData object we are cropping.
            This way we can access the file, that contains data in our
            particular region of interest, without having to load the complete
            file into memory only to discard most of it later. If parent is
            None, then this object is associated with the file that contains the
            data.
        io : pyraster.geotiff.GeoTiffFile or pyraster.netcdf.NcFile
            Instance of a IO Class that is associated with a file that contains
            the data.

        """
        self.geometry = RasterGeometry(rows, cols, sref, gt)
        self.data = data
        self.parent = parent
        self.io = io
        #self.filename = self.io.filename if io is not None else None

    @classmethod
    def from_array(cls, sref, gt, data, parent=None):
        """
        Creates a RasterData object from an array-like object.

        Parameters
        ----------
        sref : pyraster.spatial_ref.SpatialRef
            Spatial reference of the data geometry
        gt : 6-tuple
            GDAL geotransform 'matrix'
        data : array-like
            pixel values
        parent : RasterData or None, optional
            parent RasterData object


        Returns
        -------
        RasterData

        """
        rows = data.shape[-2]
        cols = data.shape[-1]
        return cls(rows, cols, sref, gt, data, parent=parent)

    @classmethod
    @_file_io_decorator
    def from_file(cls, file, read=False, io_kwargs=None):
        """
        Creates a RasterData object from a file

        Parameters
        ----------
        file : string or GeoTiffFile or NcFile
            path to the file or IO Classes instance. The decorator deals with
            the distinction
        io_kwargs: dict
            potential arguments for IO Class instance. (Happens in decorator)

        Returns
        -------
        RasterData

        """
        sref = SpatialRef(file.spatialref)
        geotransform = file.geotransform
        data = file.read(return_tags=False) if read else None
        rows, cols = file.shape
        return cls(rows, cols, sref, geotransform, data=data, io=file)

    def crop(self, geometry, inplace=True):
        """
        Crops the image by geometry. Inplace determines, whether a new object
        is returned or the cropping happens on this current object.
        The resulting RasterData object contains no-data value in pixels that
        are not contained in the original RasterData object

        Parameters
        ----------
        geometry : RasterGeometry or shapely geometry
            geometry to which current RasterData should be cropped
        inplace : bool
            if true, current object gets modified, else new object is returned

        Returns
        -------
        RasterData or None
            Depending on inplace argument

        """
        nodata = -9999  # ? set / externalize ?
        new_geom = self.geometry & geometry
        new_data = np.full((new_geom.rows, new_geom.cols), nodata)
        for r in range(new_geom.rows):
            for c in range(new_geom.cols):
                x, y = new_geom.rc2xy(r, c, center=True)
                if (x, y) in self.geometry:
                    orig_r, orig_c = self.geometry.xy2rc(x, y)
                    new_data[r][c] = self.data[orig_r][orig_c]

        if inplace:
            self.geometry = new_geom
            self.data = new_data
            self.parent = self
        else:
            return RasterData.from_array(self.geometry.sref,
                                         new_geom.gt,
                                         new_data,
                                         parent=self)

    def load(self):
        """
        Reads the data from disk and assigns the resulting array to the
        self.data attribute
        Returns
        -------
        None
        """
        self.io = self._get_orig_io()
        g = self.geometry
        sub_rect = (g.ll_x, g.ll_y, g.width, g.height)
        self.data = self.io.read(sub_rect=sub_rect)

    @file_io_decorator
    def write(self, file=None):
        """
        Writes data to on hard-drive into target file or into file associated
        with this object.

        Parameters
        ----------
        file: IO-Class or string
            either initialized instance of one of the IO Classes (GeoTiffFile,
            NcFile ..) or string representing path to the file on filesystem

        Returns
        -------
        None
        """
        # nodata ??? set / externalize ???
        file.write(self.data, band=1)

    def _get_orig_io(self):
        """
        Follows the parent links to until it reaches a RasterData object with
        valid io attribute
        Returns
        -------
        IO Class
            instance of an IO Class of an RasterData object that is associated
            with an file

        """
        parent = self.parent
        while parent.io is None:
            parent = parent.parent
        return parent.io

    def plot(self, fig=None, map_crs=None, extent=None, extent_crs=None,
             cmap='viridis'):
        """
        Plots the raster data to a map that uses map_crs projection
        if provided. When not, the map projection defaults to spatial reference
        in which the data are provided (self.geometry.sref). The extent of the
        data is specified by 'extent'. If extent is not provided, it defaults
        to the bbox of the data's geometry. If provided, one can also specify
        spatial reference of the extent that is beeing parsed, otherwise is is
        assumed that the spatial reference of the extent is same as the spatial
        reference of the data.
        :param fig: pyplot Figure (optional)
            Target figure, into which will the data be plotted
        :param map_crs: cartopy.crs.Projection or its subclass (optional)
            Spatial reference of the map. The Figure is going to be drawn in
            this spatial reference. If omitted, the spatial reference in which
            the data are present is assumed.
        :param extent: 4 - tuple (optional)
            Extent of the data. If omitted, geometry bbox is assumed.
        :param extent_crs: cartopy.crs.Projection or its subclass (optional)
            Spatial reference of parsed extent. If omitted, the spatial
            reference in which the data are present is assumed.
        :param cmap matplotlib.colors.Colormap or string
            Colormap which is to be used to plot the data
        """

        data_crs = self.geometry.to_cartopy_crs()

        if map_crs is None:
            map_crs = data_crs
        if fig is None or len(fig.get_axes()) == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=map_crs)
        else:
            ax = fig.gca()

        if extent_crs is None:
            extent_crs = data_crs
        if extent is None:
            ll_x, ll_y, ur_x, ur_y = self.geometry.geometry.bounds
            extent = ll_x, ur_x, ll_y, ur_y

        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)

        ax.set_extent(extent, crs=extent_crs)
        ax.coastlines()

        ax.imshow(self.data, extent=extent, origin='upper', transform=data_crs,
                  cmap=cmap)
        ax.set_aspect('equal', 'box')

        return fig


class RasterStack(RasterData):

    def __init__(self, geometry):
        self.geometry = geometry

    def plot(self, id=None):


class RasterMosaic(object):

    def __init__(self, arg, grid=None, aggregator=None):
        self.grid = grid
        self.aggregator = aggregator
        self._structure = self.__init_structure(arg)
        self._ds = {}
        sres =
        self.geom =

    def __init_structure(self, arg):
        if isinstance(arg, pd.DataFrame):
            structure = arg
        elif isinstance(arg, dict):
            structure = self.__dict2structure(arg)
        elif isinstance(arg, list):
            arg_dict = dict()
            for i, layer in enumerate(arg):
                arg_dict[i] = layer
            structure = self.__dict2structure(arg_dict)
        else:
            raise Exception("")

        return structure

    def __dict2structure(self, struct_dict):
        structure = dict()
        structure["rd"] = []
        structure["spatial_id"] = []
        structure["layer_id"] = []
        geoms = []
        for layer_id, layer in struct_dict.items():
            for elem in layer:
                structure['layer_id'].append(layer_id)
                if isinstance(elem, RasterData):
                    rd = elem
                elif isinstance(elem, str) and os.path.exists(elem):
                    rd = RasterData.from_file(elem)
                else:
                    raise Exception("")

                structure['rd'].append(rd)
                if rd.geometry.id is not None:
                    structure['spatial_id'].append(rd.geometry.id)
                else:
                    spatial_id = None
                    for i, geom in enumerate(geoms):
                        if geom == rd.geometry:
                            spatial_id = i
                            break
                    if spatial_id is None:
                        spatial_id = len(geoms)
                        geoms.append(rd.geometry)
                    structure['spatial_id'].append(spatial_id)

        return pd.DataFrame(structure)

    def __build_stack(self):
        struct_groups = self._structure.groupby(by="spatial_id")
        for struct_group in struct_groups:
            self._ds


    def __filepaths2stack(self, filepaths):
        io_class = get_stack_io_class(filepaths[0])
        df = pd.DataFrame({'filenames': filepaths})
        return io_class(mode='r', file_ts=df)

    def read_by_geom(self, geom):
        pass

    def read_by_pixels(self, geom):
        pass

    def write(self):
        pass
