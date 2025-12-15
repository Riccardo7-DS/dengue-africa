# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import yaml
import torch
from definitions import ROOT_DIR


class BaseConfig():
    def __init__(self):
        self.data_dir = os.path.join(ROOT_DIR,  "..",'data')
        self.output_dir = os.path.join(ROOT_DIR, "..", 'output')
        self.model_dir = os.path.join(self.output_dir, 'model')
        self.log_dir = os.path.join(self.output_dir, 'log')
        for path in [self.data_dir, self.output_dir, self.log_dir, self.model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

class ConfigTransf(BaseConfig):

    num_frames_output = 1 ### means 1 frame as output
    output_channels = 1
    include_lag = True
    num_samples = 9 #### the number of channels to use

    epochs = 50
    patience = 10
    learning_rate = 0.1
    batch_size= 8
    dim = 64

    masked_loss = True

    ### Parameters specific to GWNET
    weight_decay = 0.0001
    in_dim = 10
    out_dim = 1
    dropout = 0.3
    nhid = 16
    blocks = 6
    layers = 4
    print_every = 100

    scheduler_patience = 3
    scheduler_factor = 0.7
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

config_transf = ConfigTransf()