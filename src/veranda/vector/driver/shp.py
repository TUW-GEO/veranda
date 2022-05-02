from osgeo import ogr


def write_geometry(geom, fname, dformat="ESRI Shapefile"):
    """
    Write an geometry to a vector file.

    parameters
    ----------
    geom : Geometry
        geometry object
    fname : string
        full path of the output file name
    format : string
        format name. currently only shape file is supported
    """
    drv = ogr.GetDriverByName(dformat)
    dst_ds = drv.CreateDataSource(fname)
    srs = geom.GetSpatialReference()

    dst_layer = dst_ds.CreateLayer("out", srs=srs)
    fd = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(fd)

    feature = ogr.Feature(dst_layer.GetLayerDefn())
    feature.SetField("DN", 1)
    feature.SetGeometry(geom)
    dst_layer.CreateFeature(feature)

    feature.Destroy()
    dst_ds.Destroy()