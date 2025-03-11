import warnings
import shutil
import numpy as np
import torchvision
torchvision.set_image_backend('accimage')
import torchvision.transforms as transforms
import os
import os.path
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import cv2
from util.utils import display
import tqdm
import glob
import os

ROOT = "PUT YOUR FRAMES ROOT FOLDER HERE"  # should contain train and val folders


class SandMiningDataset(torch.utils.data.Dataset):
    def __init__(self, opt=None, mode="train"):
        self.dataroot = os.path.join(ROOT, mode)
        self.mode = mode
        self.opt = opt
        super().__init__()

        #normalizer for bands
        self.normalizer = transforms.Normalize(mean=(96, 103, 86),
                                             std=(43, 40, 39))

        self.cropper = A.Compose([
            A.RandomResizedCrop(size=(opt.imsize, opt.imsize), scale=(.8, 1.), interpolation=cv2.INTER_LINEAR),
            A.ShiftScaleRotate(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Transpose()], additional_targets={'valid': 'image'})
        self.totensor = A.Compose([
            ToTensorV2(),
        ])
        self.load_data()

    def __getitem__(self, index):
        # img, target, valid = self.get_frame()
        img = np.zeros((3, 256, 256))
        target = np.zeros((256, 256))
        valid = np.zeros((256, 256))
        cropped = self.cropper(image=img.transpose(1, 2, 0), mask=target, valid=valid.astype('float'))

        if self.mode == "train":
            transformed = self.totensor(**cropped)
            im = self.normalizer(transformed["image"].float())
            target = transformed["mask"].long()
            valid = transformed["valid"]
        elif self.mode == "val":
            im = self.normalizer(torch.from_numpy(cropped["image"].transpose(2, 0, 1)).float())
            target = cropped["mask"].astype(int)
            valid = cropped["valid"]

        rawim = cropped["image"].transpose(2, 0, 1)
        return im, target, rawim, valid.astype(bool)

    def get_frame(self):
        # randomly pick frame
        picked = np.random.randint(len(self.frames))
        data = self.frames[picked].astype(int)
        img, target, valid = data[:3], data[3], data[4]
        return img.astype(np.uint8), target, valid

    def load_data(self):
        """ Loads dataset from disk. Frames are kept in memory, only works for small datasets.
        """
        self.frames = []
        all_files = sorted(glob.glob(os.path.join(self.dataroot, "*.npy")))
        dropped_frames = 0
        for f in tqdm.tqdm(all_files, desc="loading frames"):
            frame = np.load(f)
            # only keep frames bigger than imsize
            if frame.shape[1] >= self.opt.imsize and frame.shape[2] >= self.opt.imsize:
                self.frames.append(frame)
            else:
                dropped_frames += 1
        warnings.warn(f"{dropped_frames} frame will be ignored because they are smaller than the requested patch size")
        return

    def __len__(self):
        if self.mode == "train":
            return 10000
        else:
            return 2000


def plot_training_patches():
    "plots the training patches as they will be fed to the model, including augmentations"
    opt = Options().parse()  # get training options
    dataset = create_dataset(opt, name="sandmining",
                             mode="train")  # create a dataset given opt.train_dataset_mode and other options
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads), pin_memory=True
    )
    print(len(loader))
    for i, (im, target, rawimg, valid) in tqdm.tqdm(enumerate(loader), total=len(
            loader)):  # inner loop within one epoch        d = dataset[i*100]
        display(im[0, :3], overlay=target[0], hold=True, title="img + target")
        display(im[0, :3], overlay=valid[0], hold=True, title="img + valid")
        display(target[0])


def plot_frames(directory):
    "plots the frames as they are saved on the disk, without augmentations"
    # browsing the frames for visual inspection
    all_files = glob.glob(os.path.join(directory, "*.npy"))
    for f in all_files:
        array = np.load(f)
        target = array[3]
        valid = array[4]
        display(array[:3], hold=True, overlay=target, title=f)
        display(array[:3], overlay=valid, hold=True)
        display(target, to_int=False, title= valid.shape)


if __name__ == "__main__":
    from options.options import Options
    from data import create_dataset

    mode = "val"  # "train"
    plot_frames(os.path.join(ROOT, mode))
