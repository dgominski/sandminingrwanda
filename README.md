[![DOI](https://zenodo.org/badge/946689635.svg)](https://doi.org/10.5281/zenodo.15006791)

# Sand Mining in Rwanda with TransUNet (PyTorch) 

This repository contains the code for training a TransUNet model for segmenting sand mining sites in Rwanda, with high resolution aerial imagery.

### Dependencies:

You can use the provided .yml Conda environment file to install dependencies into a separate environment:
```
# From repo root
conda env create -f env.yml
conda activate sandmining
```
Make sure Pytorch and CUDA versions are compatible with your system.

---

### Preprocessing:

`preprocessing/rasterize_polygons.py` takes as inputs a folder with areial imagery rasters, a vector file with point or polygon annotations, and a vector file with rectangles indicating what zones were annotated.
The rasters are cropped to the rectangles, polygon annotations for the corresponding areas are rasterized, and the results are saved together in a npy frame.

**Features**:

_--patch-size_ defines the minimal patch size in pixels. If the labeled rectangle is smaller than this, it will be extended to this size, but pixels outside of the annotation rectangle will be considered invalid and not used for computing the loss.

_--starting-index_ optionally makes the names of output frames start at a specific value. Useful if you preprocess the data in multiple batches.

The script handles annotation rectangles covering multiple rasters (using rasterio.merge).

 ⚠️ There might be issues if there is a mismatch between the CRS of the rasters and the CRS of the annotation files. If you encounter this, please convert them to the same CRS, 
or uncomment the lines converting geometries to a common provided --crs 


---

### Training:

Once the frames are ready, split them in train and val folders, modify the hard-coded data root folder in `data/sandmining_dataset.py` and run the training script.

Example:

`python3 train.py --model transunet --train-dataset sandmining --batch-size 4 --gpu-ids 0 --lr 5e-7 --nepoch 200 --lr-policy poly --checkpoints-dir $CKPT_FOLDER --val-freq 2000 --print-freq 100 --imsize 384 --num-threads 16 --ckpt imagenet`
