from torch.optim import lr_scheduler
import math


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.nepoch> epochs
    and linearly decay the rate to zero over the next <opt.nepoch_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.nepoch) / float(opt.nepoch_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, threshold=0.1, patience=3)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.nepoch, eta_min=0)
    elif opt.lr_policy == 'exp':
        exp_decay = math.exp(-0.01)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)
    elif opt.lr_policy == "cyclical":
        scheduler = lr_scheduler.CyclicLR(optimizer, max_lr=optimizer.param_groups[0]['lr'], base_lr=0.25*optimizer.param_groups[0]['lr'], cycle_momentum=False, step_size_up=100, step_size_down=100)
    elif opt.lr_policy == "poly":
        f = lambda epoch: (1-epoch/opt.nepoch)**0.9
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
