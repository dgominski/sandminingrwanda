import argparse
import os
from util import utils
import torch
import models
import data
import datetime


class Options():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot',  help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime("%m%d_%H%M"), help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints-dir', type=str, default='/tmp/', help='models are saved here')
        parser.add_argument('--load-from', type=str, default=None) # pretrain if the model stopped earlier than expect due to some reason
        parser.add_argument('--load-epoch', type=str, default=None)
        parser.add_argument('--print-freq', type=int, default=100, help='frequency of showing training results on console')

        # model parameters
        parser.add_argument('--model', type=str, default='transunet', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--alpha', type=float, default=1.0, help='parameter for loss weighing')
        parser.add_argument('--ckpt', type=str, default='bigearthnet', help='what pretrained model to use') #imagenet

        # dataset parameters
        parser.add_argument('--dataset-size', type=int, default=None, help='number of samples to take to build a subset')
        parser.add_argument('--train-dataset', type=str, default='peru', help='training dataset name')
        parser.add_argument('--mean', default=[0.485, 0.456, 0.406], help='dataset image mean')
        parser.add_argument('--std', default=[0.229, 0.224, 0.225], help='dataset image std')
        parser.add_argument('--test-dataset', '-td', default=None)
        parser.add_argument('--num-threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch-size', type=int, default=1, help='input batch size')
        parser.add_argument('--load-size', type=int, default=286, help='loading image size')
        parser.add_argument('--imsize', type=int, default=256, help='final image size')

        # additional parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--debug', action='store_true', default=False, help='debug mode: everything goes to TMP dir')
        parser.add_argument('--no-val', dest='val', action='store_false', help='do not run validation')

        ################################# TRAIN ########################################################################
        # visualization parameters
        # network saving and loading parameters
        parser.add_argument('--save-epoch-freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        # training parameters
        parser.add_argument('--nepoch', type=int, default=5, help='# of iter at starting learning rate')
        parser.add_argument('--nepoch-decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
        parser.add_argument('--lr-policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine | exp]')
        # validation parameters
        parser.add_argument('--val-freq', type=int, default=4000, help='frequency of running validation step')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        if not opt.debug:
            # save to the disk
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            utils.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
