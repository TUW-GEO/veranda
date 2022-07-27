from osgeo import ogr


def write_geometry(geom, filepath, output_format="ESRI Shapefile"):
    """
    Write a geometry to a vector file.

    Parameters
    ----------
    geom : ogr.Geometry
        OGR geometry object.
    filepath : str
        Full system path to the output file.
    output_format : str, optional
        Vector format name. Defaults to "ESRI Shapefile".

    """
    drv = ogr.GetDriverByName(output_format)
    dst_ds = drv.CreateDataSource(filepath)
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