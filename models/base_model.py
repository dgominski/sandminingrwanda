import os
import numpy as np
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
import util.utils as util
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from models import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.loss_names = []
        self.scalar_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = {}
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.mode = "train"

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt, mode):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if mode == 'train':
            self.schedulers = [networks.get_scheduler(self.optimizers[key], opt) for key in self.optimizers]
        else:
            self.eval()
        if opt.load_from:
            self.load_networks(opt.load_epoch) if opt.load_epoch is not None else self.load_networks("latest")
        self.print_networks(opt.verbose)
        self.mode = mode

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if net is not None:
                    net.eval()
        self.mode = "eval"

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if net is not None:
                    net.train()
        self.mode = "train"

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def compute_scalars(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self, verbose=True):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        for key in self.optimizers:
            lr = self.optimizers[key].param_groups[0]['lr']
            if verbose:
                print('optimizer {} :learning rate = {}'.format(key, lr))

    def get_current_visuals_original(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_visuals(self):
        nim = self.opt.batch_size
        if nim == 1:
            nim += 1
        all =[]
        for i in range(0,min(nim-1,5)):
            row=[]
            for name in self.visual_names:
                if isinstance(name, str):
                    if hasattr(self,name):
                        im = util.tensor2im(getattr(self, name).data[i:i+1,:,:,:],mean=self.opt.mean,std=self.opt.std)
                        row.append(im)
            row=tuple(row)
            all.append(np.hstack(row))
        all = tuple(all)
        allim = np.vstack(all)
        return OrderedDict([(self.opt.name,allim)])

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                if hasattr(self, 'loss_' + name):
                    errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
                else:
                    warnings.warn('loss_' + name + " not found")
        return errors_ret

    def get_current_scalars(self):
        """Return scalar values, such as parameters or classification accuracies. train.py will log these values"""
        params_ret = OrderedDict()
        for name in self.scalar_names:
            if isinstance(name, str):
                if hasattr(self, name):
                    params_ret[name] = float(getattr(self, name))
                else:
                    warnings.warn('scalar ' + name + ' not found')
        return params_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)
                if net is None:
                    continue
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

        for optimizer_name in self.optimizers:
            save_filename = '%s_optim_%s.pth' % (epoch, optimizer_name)
            save_path = os.path.join(self.save_dir, save_filename)
            optim = self.optimizers[optimizer_name]
            if optim is None:
                continue
            torch.save(optim.state_dict(), save_path)

        for i, scheduler in enumerate(self.schedulers):
            save_filename = '%s_scheduler_%s.pth' % (epoch, i)
            save_path = os.path.join(self.save_dir, save_filename)
            scheduler = scheduler
            if scheduler is None:
                continue
            torch.save(scheduler.state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.opt.load_from, load_filename)
                net = getattr(self, name)
                print('loading the model from %s' % load_path)
                try:
                    state_dict = torch.load(load_path, map_location=str(self.device))
                except Exception as e:
                    raise FileNotFoundError("Model {} not loaded, exception {}".format(name, e))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                util.load_state_dict(net, state_dict)

        if self.mode == "train":
            for optimizer_name in self.optimizers:
                if isinstance(optimizer_name, str):
                    load_filename = '%s_optim_%s.pth' % (epoch, optimizer_name)
                    load_path = os.path.join(self.opt.load_from, load_filename)
                    if os.path.exists(load_path):
                        optim = self.optimizers[optimizer_name]
                        print('loading the optimizer from %s' % load_path)
                        state_dict = torch.load(load_path)
                        if hasattr(state_dict, '_metadata'):
                            del state_dict._metadata
                        optim.load_state_dict(state_dict)
                    else:
                        print('optimizer %s not found' % load_path)

            for i, scheduler in enumerate(self.schedulers):
                load_filename = '%s_scheduler_%s.pth' % (epoch, i)
                load_path = os.path.join(self.opt.load_from, load_filename)
                if os.path.exists(load_path):
                    scheduler = self.schedulers[i]
                    print('loading the scheduler from %s' % load_path)
                    state_dict = torch.load(load_path)
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                    scheduler.load_state_dict(state_dict)
                else:
                    print('scheduler %s not found' % load_path)

            for key in self.optimizers:
                lr = self.optimizers[key].param_groups[0]['lr']
                if lr < 1e-9:
                    warnings.warn("Learning rate is very low (lr < 1e-9), if retraining a model that has finished training, it is recommended to not load the optimizers and schedulers (can be moved to another folder or deleted)")

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                if net is None:
                    print('[Network %s] not defined' % (name))
                    continue
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_requires_grad_paramlist(self, paramlist, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(paramlist, list):
            raise ValueError("Taking only a list of parameters as input, please call self.set_requires_grad for setting"
                             "on networks")
        for param in paramlist:
            param.requires_grad = requires_grad

    def plot_grad_flow_v2(self):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in self.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.detach().cpu().abs().mean())
                max_grads.append(p.grad.detach().cpu().abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.show()


