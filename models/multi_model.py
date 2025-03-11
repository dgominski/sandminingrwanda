import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from models.transunet_model import TransUnetModel


class MultiModel(torch.nn.Module):
    def __init__(self, opt, mode):
        torch.nn.Module.__init__(self)
        print(f"## MultiGDet model")

        model_dir = opt.load_from
        paths = [name for name in os.listdir(opt.load_from) if os.path.isdir(os.path.join(opt.load_from, name))]
        if len(paths) > 3:
            print(f"## Warning: more than 3 models found for ensembling in {opt.load_from}, might lead to OOM errors")

        self.models = []
        for p in paths:
            opt.load_from = os.path.join(model_dir, p)
            m = TransUnetModel(opt, mode="test")
            m.eval()
            m.setup(opt)
            self.models.append(m)

        self.device = self.models[0].device

        self.opt = opt

    def setup(self, opt):
        pass

    def __call__(self, image, rawimg):
        all_logits = []
        for m in self.models:
            m.forward(image)
            all_logits.append(m.probs)
        self.heatmap = torch.stack(all_logits).mean(0)
        self.probs = torch.softmax(self.heatmap, dim=1)
        self.predicted_mask = torch.argmax(self.probs, dim=1)
        self.visual_pred = self.predicted_mask.unsqueeze(1)
        return self.heatmap, self.predicted_mask


if __name__ == "__main__":
    from options.options import Options
    from models import create_model
    import rasterio
    from rasterio.windows import Window
    import torchvision.transforms as transforms
    from util.utils import display

    opt = Options()
    opt = opt.parse()
    model = create_model(opt, mode='test')
    raster = rasterio.open("/bigdata/users/Dimitri/files_ke/KE_TransUnet_Aerial/Rwanda_RGB/Piece4BSE3R.tif")
    for i in range(20):
        data = raster.read(window=Window(256*i, 0, 256, 256))
        print(data.shape)
        [print(f"mean {data[b].mean()}, std {data[b].std()}") for b in range(3)]
        data = torch.from_numpy(data).unsqueeze(0).float()
        tf = transforms.Normalize(mean=(96, 103, 86),
                             std=(43, 40, 39))
        norm_data = tf(data)
        probs, mask = model(norm_data)
        display(data[0], hold=True)
        display(probs, hold=True, to_int=False, title="background")
        display(probs[:, 1], hold=True, to_int=False, title="positive")
        display(mask[:, 0])