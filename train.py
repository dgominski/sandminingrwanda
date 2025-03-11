import os
from torch.utils.tensorboard import SummaryWriter
from options.options import Options
from data.sandmining_dataset import SandMiningDataset
from models import create_model
from util.evaluate import AverageMeter, DictAverager
import tqdm
import torch
from torchvision.utils import make_grid, save_image
from util.utils import UnNormalize


if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()

    opt = Options().parse()  # get training options

    trainset = SandMiningDataset(opt, mode='train')
    valset = SandMiningDataset(opt, mode='val')

    model = create_model(opt, mode='train')  # create a model given opt.model and other options
    model.setup(opt, mode="train")  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    val_iters = 0

    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "tbx")) if not opt.debug else SummaryWriter('TMP')
    unnorm = UnNormalize(mean=(96, 103, 86), std=(43, 40, 39))  # unnormalize images for visualization

    best_jaccard = 0
    for epoch in range(1, opt.nepoch + opt.nepoch_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        dataloader = torch.utils.data.DataLoader(
                    trainset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads), pin_memory=True
                )
        dataset_size = len(dataloader)  # get the number of images in the dataset.
        print('Train split size: {}'.format(len(trainset)))
        print('>> Epoch {}/{}: training...'.format(epoch, opt.nepoch+opt.nepoch_decay))

        for i, (im, target, rawimg, valid) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):  # inner loop within one epoch
            total_iters += opt.batch_size
            model.forward(im)
            model.optimize_parameters(target, valid)  # calculate loss functions, get gradients, update network weights

            if total_iters//opt.batch_size % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                writer.add_scalars("train/loss", losses, global_step=total_iters)
                model.compute_scalars(model.predicted_mask, target.to(model.device))
                scalars = model.get_current_scalars()
                writer.add_scalars("train/scalars", scalars, global_step=total_iters)
                writer.add_image(f"train/target", make_grid(model.visual_target), global_step=total_iters)
                writer.add_image(f"train/pred", make_grid(model.visual_pred), global_step=total_iters)
                writer.add_image(f"train/raw_input", make_grid(rawimg), global_step=total_iters)
                writer.add_image(f"train/aug_input", make_grid(unnorm(im)), global_step=total_iters)

            if total_iters//opt.batch_size % opt.val_freq == 0 and opt.val:
                model.eval()
                print('Running validation...')
                val_loader = torch.utils.data.DataLoader(
                    valset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads), pin_memory=True
                )
                averager = DictAverager()
                for i, (im, target, rawimg, valid) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):  # inner loop within one epoch
                    model.forward(im)
                    results = model.val(im, target, rawimg, valid)
                    averager.update(results)

                res = averager.get_avg()
                writer.add_scalars("val",res, global_step=val_iters)
                if res["jaccard_raw"] > best_jaccard:
                    best_jaccard = res["jaccard_raw"]
                    model.save_networks('best')
                val_iters += 1
                model.train()

        if (epoch+1) % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch {} / {}'.format(epoch, opt.nepoch+opt.nepoch_decay))

        model.update_learning_rate()  # update learning rates at the end of every epoch.

    print('saving the model at last epoch, iters %d' % (total_iters))
    model.save_networks('latest')
