"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def preprocess_img(img, to_int=True):
    if isinstance(img, str):
        maxsize = [512, 512]
        try:
            img = Image.open(img)
            img.thumbnail(maxsize, Image.ANTIALIAS)
        except:
            print("Image not readable with PIL, trying with GDAL")
            im = gdal.Open(img, gdal.GA_ReadOnly)
            nbands = im.RasterCount
            for i in range(1, nbands):
                rb = im.GetRasterBand(i)
                array = rb.ReadAsArray()
                display(array)
            return

    if isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img[0, :, :, :]
        img = img.detach().cpu().numpy()
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


def denormalize_image(image):
    "denormalizes the input image for displaying"
    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image - np.min(image))/np.ptp(image)
    return image


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for i in range(image2.shape[0]):
            for t, m, s in zip(image2[i], self.mean, self.std):
                t.mul_(s).add_(m)
        image2 = (255 * (image2 - image2.amin(dim=[0, 2, 3], keepdim=True)) / (image2.amax(dim=[0, 2, 3], keepdim=True) - image2.amin(dim=[0, 2, 3], keepdim=True))).long()
        return 255 - image2


def load_state_dict(net, state_dict):
    stlist = list(net.state_dict())
    if not stlist:
        net.load_state_dict(state_dict)
        return
    if "module" in list(net.state_dict())[0]:
        if "module" in list(state_dict)[0]:
            net.load_state_dict(state_dict, strict=False)
        else:
            state_dict = {"module."+x: y for x, y in state_dict.items()}
            net.load_state_dict(state_dict)
    else:
        if "module" in list(state_dict)[0]:
            state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
            net.load_state_dict(state_dict)
        else:
            net.load_state_dict(state_dict)
    return


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()