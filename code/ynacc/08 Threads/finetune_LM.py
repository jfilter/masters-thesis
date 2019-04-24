#!/usr/bin/env python

from fastai.text import *
from fastai.datasets import *
from pathlib import Path
import pandas as pd
from fastai.metrics import *
from fastai.train import *
from fastai.imports import nn, torch
from fastai.callbacks import *

from fastai import *
from fastai.text import * 

import random
import math
import datetime
from sacred import Experiment

from sacred.observers import MongoObserver
import fastai

import news_utils.fastai

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp")
parser.add_argument("--device", type=int)
args = parser.parse_args()

torch.cuda.set_device(args.device)

EX_PA = Path('/mnt/data/group07/johannes/ynacc_proc/replicate/' + args.exp)

ex = Experiment(EX_PA.stem)

ex.observers.append(MongoObserver.create(db_name=EX_PA.stem))

print('f')

@ex.config
def my_config():
    bs=64
    epochs_start = math.ceil(random.uniform(0, 2))
    epochs=math.ceil(random.uniform(5, 10))
    drop_mult=random.uniform(0.4, 1)
    layer_factor=random.uniform(2, 3)
    exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")
    backwards = False

@ex.main
def my_main(epochs, drop_mult, exp_id, bs, layer_factor, epochs_start):    
    ex.info['path'] = EX_PA
    
    data_lm = TextLMDataBunch.load(EX_PA, bs=bs)
    learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=drop_mult)
    
    lr = news_utils.fastai.get_optimal_lr(learn, runs=3)
    
    learn.callbacks += [
        news_utils.fastai.SacredLogger(learn, ex),
        SaveModelCallback(learn, name=exp_id),
        EarlyStoppingCallback(learn, patience=3)
    ]
    
    learn.fit_one_cycle(epochs_start, lr)
    
    learn.unfreeze()
    
    learn.fit_one_cycle(epochs, [lr / (layer_factor * (3 - x)) for x in range(3)] + [lr])

print('f')
    
ex.run()
