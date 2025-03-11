import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DictAverager(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = {}

    def update(self, dict):
        for k in dict:
            if k not in self.vals:
                # declare
                self.vals[k] = []
            if isinstance(dict[k], int) or isinstance(dict[k], float):
                self.vals[k].append(dict[k])
            if isinstance(dict[k], list) or isinstance(dict[k], tuple):
                self.vals[k].extend(dict[k])

    def get_avg(self):
        output_dict = {}
        for k in self.vals:
            output_dict[k] = np.round(np.nanmean(self.vals[k]), decimals=4)
        return output_dict
