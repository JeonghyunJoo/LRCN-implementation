from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import math
import numpy as np

import torch.nn as nn


class MyDataParallel(nn.DataParallel):
    """
    Wrapper class for nn.DataParallel

    This class provides direct accesses to attributes for the original module
    """
    def __init__(self, model, **kargs):
        super(MyDataParallel, self).__init__(model, **kargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name) # Direct access to attributes belonging to the original module


class TimeChecker():
    def __init__(self):
        self.start_time_list = dict()
        self.time_accum = dict()
        self.start_time_list['default'] = 0
        self.time_accum['default'] = 0
        #self.startTime = 0
        self.epoch = 0

    def reset(self, k = 'default'):
        self.start_time_list[k] = 0
        self.time_accum[k] = 0

    def start(self, k = 'default'):
        self.start_time_list[k] = time.time()

    def elapsed(self, k = 'default'):
        interval_time = time.time() - self.start_time_list[k]
        self.time_accum[k] = self.time_accum.get(k, 0) + interval_time
        return time_format(interval_time)

    def get_time(self, k = 'default'):
        return time_format( self.time_accum.get(k, 0) )
    def epoch_count(self):
        self.epoch += 1


def time_format(s):
    h = math.floor(s / 3600)
    m = math.floor((s-3600*h) / 60)
    s = s - h*3600 - m*60
    return '%dh %dm %ds' % (h, m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (time_format(s))
