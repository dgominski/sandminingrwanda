import warnings
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = 1000000000     # to avoid error "https://github.com/zimeon/iiif/issues/11"
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # to avoid error "https://github.com/python-pillow/Pillow/issues/1510"
import os
import os.path
import glob
import json
import rasterio
import geopandas as gpd
from rasterio import features, windows
from rasterio import transform as rt
from shapely.ops import transform as transform_polygon
from shapely.geometry import Polygon, Point, box
from shapely.affinity import translate, scale
from rasterio.merge import merge
import tqdm
import matplotlib.pyplot as plt
import pyproj
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from rasterio.features import is_valid_geom
import math


def calculate_boundary_weights(polygons, scale):
    """Find boundaries between close polygons.

    Scales up each polygon, then get overlaps by intersecting. The overlaps of the scaled polygons are the boundaries.
    Returns geopandas data frame with boundary polygons.
    """
    # Scale up all polygons around their center, until they start overlapping
    # NOTE: scale factor should be matched to resolution and type of forest
    scaled_polys = gpd.GeoDataFrame(
        {"geometry": polygons.geometry.scale(xfact=scale, yfact=scale, origin='center')})

    # Get intersections of scaled polygons, which are the boundaries.
    boundaries = []
    for i in range(len(scaled_polys)):

        # For each scaled polygon, get all nearby scaled polygons that intersect with it
        try:
            nearby_polys = scaled_polys[scaled_polys.geometry.intersects(scaled_polys.iloc[i].geometry)]
        except:
            continue

        # Add intersections of scaled polygon with nearby polygons [except the intersection with itself!]
        for j in range(len(nearby_polys)):
            if nearby_polys.iloc[j].name != scaled_polys.iloc[i].name:
                try:
                    boundaries.append(scaled_polys.iloc[i].geometry.intersection(nearby_polys.iloc[j].geometry))
                except:
                    boundaries.append(Point(0.0, 0.0))

    # Convert to df and ensure we only return Polygons (sometimes it can be a Point, which breaks things)
    boundaries = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(boundaries)}, crs=polygons.crs).explode(index_parts=True)
    boundaries = boundaries[boundaries.type == "Polygon"]

    # If we have boundaries, difference overlay them with original polygons to ensure boundaries don't cover labels
    if len(boundaries) > 0:
        boundaries = gpd.overlay(boundaries, polygons, how='difference')
    else:
        boundaries = boundaries.append({"geometry": box(0, 0, 0, 0)}, ignore_index=True)

    return boundaries


def preprocess_all(args, starting_index=0):
    i = starting_index
    rasters = glob.glob(os.path.join(args.raster_dir, f"*.{args.raster_ext}"))
    rasterfiles = np.array(rasters)

    shp = gpd.read_file(args.polygon_file)
    if shp.crs is None:
        shp = shp.set_crs(args.crs)
    shp = shp.to_crs(args.crs)
    shp = shp.explode(column="geometry", index_parts=False, ignore_index=True)  # remove potential multipolygons
    shp = shp[~shp.is_empty]  # remove potential empty geometries
    invalid_geoms = shp["geometry"].apply(is_valid_geom)
    print(f"found {(~invalid_geoms).sum()} invalid geometries")

    if "Class" in shp.columns:
        classnames = shp["Class"].unique()
        print(f"Found {len(classnames)} unique class names in the Class column of the polygon shapefile")
        class_quantizer = {name: v for v, name in enumerate(classnames)}
        shp["qclass"] = shp["Class"].apply(lambda x: class_quantizer[x])
        print("###################################")
        print("Rasterizing polygons to quantized class values (multiclass masks). \n"
              "Unlabeled areas within rectangles considered negative (-1). \n"
              "Areas outside rectangles considered invalid (separate band).")
        print(f"Class definition:")
        [print(f"{name}: {v}") for v, name in enumerate(classnames)]
        print("###################################")
        unlabeled_value = -1
    else:
        print("Did not find class information.")
        print("Rasterizing polygons to binary class values (binary masks). \n "
              "Unlabeled areas considered negative (0). \n"
              "Areas outside rectangles considered invalid (separate band).")
        shp["qclass"] = 1
        unlabeled_value = 0

    footprints = []
    raster_crs = None
    for rasterpath in rasters:
        raster = rasterio.open(rasterpath)
        if raster_crs is None:
            raster_crs = raster.crs
        else:
            if raster.crs != raster_crs:
                raise ValueError(f"Please ensure the rasters are all in the same CRS (needed for reading data windows across rasters). \n "
                                 f"CRS read with the first raster: {raster_crs} \n "
                                 f"Raster {rasterpath} has a different CRS: {raster.crs}")
        footprints.append(transform_bounds(raster.crs, CRS.from_epsg(args.crs), *raster.bounds))

    # get extent of rasters
    raster_polys = [box(*b) for b in footprints]

    rectangles = gpd.read_file(args.rectangle_file)
    if rectangles.crs is None:
        rectangles = rectangles.set_crs(args.crs)
    rectangles = rectangles.to_crs(args.crs)
    rectangles = rectangles[~rectangles.is_empty]

    # browse once to get bigger rectangles
    reading_bounds = []
    without_raster = 0
    for index, row in tqdm.tqdm(rectangles.iterrows(), desc="Browsing database to generate bounds with target frame size", total=len(rectangles.index)):  # Looping over all points
        geom = row["geometry"]
        bounds = geom.bounds
        annotation_zone_polygon = box(*bounds)

        intersects = np.array([annotation_zone_polygon.intersects(poly) for poly in raster_polys])
        if intersects.sum() == 0:
            without_raster += 1
            continue
        rst = rasterio.open(rasterfiles[intersects][0])
        window = windows.from_bounds(*transform_bounds(CRS.from_epsg(args.crs), raster_crs, *geom.bounds), transform=rst.transform)
        windowparams = [window.col_off, window.row_off, window.width, window.height]
        if args.round_up:
            target_patch_size = (args.patch_size * math.ceil(window.height / args.patch_size), args.patch_size * math.ceil(window.width / args.patch_size))
        else:
            target_patch_size = (args.patch_size, args.patch_size)
        if window.height < target_patch_size[0]:
            diff = target_patch_size[0] - window.height
            windowparams[1] = window.row_off - diff / 2
            windowparams[3] = target_patch_size[0]
        if window.width < target_patch_size[1]:
            diff = target_patch_size[1] - window.width
            windowparams[0] = window.col_off - diff / 2
            windowparams[2] = target_patch_size[1]
        window = windows.Window(*windowparams)
        reading_bounds.append(transform_bounds(raster_crs,  CRS.from_epsg(args.crs), *windows.bounds(window, transform=rst.transform)))

    print(f"{without_raster}/{len(rectangles.index)} annotation zones does not have any corresponding raster(s)")

    processing_log = tqdm.tqdm(total=len(reading_bounds), position=1, desc="Extracting raster data for annotation zones")
    saving_log = tqdm.tqdm(total=len(reading_bounds), position=2, bar_format='{desc}')
    for b in reading_bounds:
        p = Polygon.from_bounds(*b)

        # find intersecting rasters
        intersects = np.array([p.intersects(poly) for poly in raster_polys])

        if sum(intersects) == 0:
            raise ValueError("annotation zone does not fall inside any raster")
        elif sum(intersects) == 1:
            rst = rasterio.open(rasterfiles[intersects].item())
            window = windows.from_bounds(*transform_bounds(CRS.from_epsg(args.crs), rst.crs, *p.bounds), transform=rst.transform)
            npraster = rst.read(window=window)
        elif sum(intersects) > 1:
            to_merge = [rasterio.open(r, mode="r") for r in rasterfiles[intersects].tolist()]
            npraster, _ = merge(to_merge, bounds=transform_bounds(CRS.from_epsg(args.crs), raster_crs, *b))
        else:
            raise ValueError()

        transform = rt.from_bounds(p.bounds[0], p.bounds[1], p.bounds[2], p.bounds[3], npraster.shape[2], npraster.shape[1])
        annotation_band = unlabeled_value * np.ones(npraster.shape[1:])
        validity_band = 0 * np.ones(npraster.shape[1:])
        boundary_band = 0 * np.ones(npraster.shape[1:])
        try:
            labeled_polygons = ((geom, value) for geom, value in zip(shp.geometry, shp.qclass))
            polygon_mask = features.rasterize(shapes=labeled_polygons, fill=-1, out=annotation_band, transform=transform, dtype=rasterio.int16, all_touched=False)[None,]
            validity_rectangles = (geom for geom in rectangles.geometry)
            validity_mask = features.rasterize(shapes=validity_rectangles, default_value=1, fill=-1, out=validity_band, transform=transform, dtype=rasterio.int16, all_touched=True)[None,]
            if args.get_boundary_weights:
                polygons_in_area = shp[shp.within(box(*p.bounds))]
                boundary_df = calculate_boundary_weights(polygons_in_area, scale=1.5)
                boundary_polygons = (geom for geom in boundary_df.geometry)
                boundary_weights = features.rasterize(shapes=boundary_polygons, default_value=10, fill=1, out=boundary_band, transform=transform, dtype=rasterio.int16, all_touched=True)[None,]
        except ValueError as ve:
            print(f"got error {ve} when rasterizing")
            continue

        if args.get_boundary_weights:
            out_array = np.concatenate((npraster, polygon_mask, validity_mask, boundary_weights))
        else:
            out_array = np.concatenate((npraster, polygon_mask, validity_mask))

        output_fp = os.path.join(args.output_dir, f"{i}.npy")
        np.save(output_fp, out_array)
        saving_log.set_description_str(f"saving np array {output_fp}, shape {out_array.shape}")
        processing_log.update()
        i += 1


def preprocess_img(img, to_int=True):

    if isinstance(img, Image.Image):
        img = np.array(img)
    # if isinstance(img, accimage.Image):
    #     tmpimg = img
    #     img = np.zeros([img.channels, img.height, img.width], dtype=np.float32)
    #     tmpimg.copyto(img)
    if isinstance(img, np.ndarray):
        if img.dtype == bool:
            img = img.astype(int)
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[0] == 1:
            img = img.squeeze(axis=0)
        if to_int:
            img = 255 * (img - np.min(img)) / np.ptp(img)
            img = img.astype(int)
    else:
        raise ValueError("Unknown image type")
    return img


def display(img, overlay=None, to_int=True, hold=False, title=None):
    img = preprocess_img(img, to_int=to_int)
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.imshow(img)
    if overlay is not None:
        overlay = preprocess_img(overlay, to_int=to_int)
        plt.imshow(overlay, cmap='jet', alpha=0.2)
    if not hold:
        plt.show()
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert rasters and annotation files into DL-ready patches')
    parser.add_argument('--raster-dir', type=str, required=True, help='target directory with preprocessed rasters')
    parser.add_argument('--output-dir', type=str, required=True, help='output directory to save patches')
    parser.add_argument('--polygon-file', type=str, required=True, help='polygon annotation file')
    parser.add_argument('--rectangle-file', type=str, required=True, help='rectangle annotation zones file')
    parser.add_argument('--patch-size', type=int, required=True, help='patch size')
    parser.add_argument('--round-up', action='store_true', default=False, help='optionally, round up window size to nearest multiple of patch size')
    parser.add_argument('--starting-index', type=int, required=False, default=0, help='starting int value for naming patches')
    parser.add_argument('--crs', type=str, required=True, default="25832", help='equal area CRS to use')
    parser.add_argument('--get-boundary-weights', action='store_true', default=False, help='whether to compute boundary weights between polygons')
    parser.add_argument('--raster-ext', type=str, required=True, help='raster files extension')

    args = parser.parse_args()

    preprocess_all(args, starting_index=args.starting_index)