import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))


class Config(object):
    def __init__(self):
        self.GAMMA = 0.95
        self.a_lr = 1e-4
        self.c_lr = 2e-4
        self.cs_lr = 1e-4
        self.n_action = None
