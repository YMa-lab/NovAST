#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import argparse
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import torch
import torch_geometric
from .transfer import Transfer
import time

def spatial_classification_tool(sp_data, sc_data, args, output_path):
    ''' Spatial classification workflow.

    # Arguments
        config (Config): Configuration parameters.
        data_name (str): Data name.
    '''
    tsfr = Transfer(sp_data, sc_data, output_path)
    # in original paper, filtering only applicable to stereo-seq which they generated
    # if "sc" in args.dataset_train:
    #     sc_data = tsfr.filter(sc_data)

    # # Add noise manually.
    # if dataset != 'sc_data':
    #     drop_factor = (np.random.random(adata.shape) > config['preprocess']['drop_rate']) * 1.
    #     adata.X = adata.X * drop_factor

    ### train DNN model from sc_data
    ### learn_sc include the filtering
    tsfr.learn_sc(ann_key=args.celltype_name_train, batch_size=args.batch_size)

    # transfer from sc to sp
    tsfr.sc2st()

    # train GDAE model to annotate
    tsfr.annotation()