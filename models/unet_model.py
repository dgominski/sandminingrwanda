import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from models.base_model import BaseModel
import torch.nn.functional as F
from models import create_model
import segmentation_models_pytorch as smp
import math
from layers.loss import FocalLoss, TverskyLoss, DiceLoss
from torchmetrics.functional import dice, jaccard_index
import numpy as np
from torch.nn import CrossEntropyLoss


class UnetModel(BaseModel, torch.nn.Module):
    """
    Unet model relying on segmentation_models_pytorch library to allow loading pretrained segmentation model such as
    ResNet50 pretrained on ImageNet.
    """
    def __init__(self, opt, mode):
        BaseModel.__init__(self, opt)
        torch.nn.Module.__init__(self)
        self.mode = mode
        print(f"## Unet model")
        self.loss_names = ['pixel', 'region', 'total']
        self.scalar_names = ["jaccard", "dice"]
        self.model_names = ['unet']
        self.visual_names = ["visual_target", "visual_pred"]

        if opt.ckpt == "imagenet":
            self.unet = smp.Unet("resnet50", in_channels=3, encoder_weights='imagenet', classes=2)
        elif opt.ckpt == "None":
            self.unet = smp.Unet("resnet50", in_channels=3, encoder_weights=None, classes=2)
        else:
            raise NotImplementedError("Only imagenet weights are supported for now")

        self.unet.to(self.device)

        self.criterion_region = DiceLoss(mode="binary", from_logits=True)
        self.criterion_pixel = CrossEntropyLoss() # in the begnining is the crossEntropyLoss

        self.optimizers["main"] = torch.optim.Adam(list(self.unet.parameters()), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    def forward(self, img):
        self.heatmap = self.unet(img.to(self.device))
        self.probs = torch.softmax(self.heatmap, dim=1)
        self.predicted_mask = torch.argmax(self.probs, dim=1)
        self.visual_pred = self.predicted_mask.unsqueeze(1)
        return self.predicted_mask

    def __call__(self, img, rawimg):
        self.forward(img.float())
        return self.probs, self.predicted_mask.unsqueeze(1)

    def compute_loss(self, target, valid):
        self.visual_target = target.unsqueeze(1)
        self.loss_region = self.criterion_region(self.probs, target)
        self.loss_pixel = (valid * self.criterion_pixel(self.probs, target)).mean([1, 2]).mean()
        self.loss_total = self.opt.alpha * self.loss_region + (1 - self.opt.alpha) * self.loss_pixel
        return self.loss_total

    def optimize_parameters(self, target, valid):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.compute_loss(target.to(self.device), valid.to(self.device))
        for name in self.optimizers:
            self.optimizers[name].zero_grad()
        self.loss_total.backward()
        for name in self.optimizers:
            self.optimizers[name].step()
        return

    def compute_scalars(self, predicted_mask, target, suffix="raw"):
        res = {
            f"dice_{suffix}": dice(predicted_mask, target, num_classes=1, multiclass=False).cpu().item(),
            f"jaccard_{suffix}": jaccard_index(predicted_mask, target, num_classes=1, task='binary').cpu().item()
        }
        if suffix == "raw":
            self.jaccard = res["jaccard_raw"]
            self.dice = res["dice_raw"]
        return res

    def val(self, img, target, rawimg, valid):
        with torch.no_grad():
            predicted_mask = self.forward(img)
            results = self.compute_scalars(predicted_mask, target.to(self.device))
            self.compute_loss(target.to(self.device), valid.to(self.device))
            results["loss_pixel"] = self.loss_pixel.cpu().item()
            results["loss_region"] = self.loss_region.cpu().item()
            results["loss_total"] = self.loss_total.cpu().item()
        return results

    def test(self, dataset):
        pass


if __name__ == '__main__':
    from options.options import Options
    from data import create_dataset
    from util.evaluate import DictAverager
    import random
    import imgaug
    import numpy as np

    opt = Options()
    opt = opt.parse()
    model = create_model(opt, mode='test')
    model.setup(opt)
    model.eval()

    best_dice = 0

    dataset = create_dataset(opt, name=opt.train_dataset, mode='train')
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads), pin_memory=True
    )

    averager = DictAverager()

    random.seed(42)
    imgaug.random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    for i, (im, target, rawimg, valid) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        model.forward(im)
        results = model.val(im, target, rawimg, valid)
        averager.update(results)
    res_str = f"{averager.get_avg()}"
    print(res_str)

