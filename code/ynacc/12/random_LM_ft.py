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
#parser.add_argument("--do", type=float)
#parser.add_argument("--device", type=int)
args = parser.parse_args()

#torch.cuda.set_device(args.device)

EX_PA = Path('/mnt/data/group07/johannes/ynacc_proc/proper_threads/exp/lm/' + args.exp)

exp_name = str(EX_PA.stem) + 'lm'

ex = Experiment(exp_name)
ex.observers.append(MongoObserver.create(db_name=exp_name))


@ex.config
def my_config():
    bs=128
    #epochs_start = math.ceil(random.uniform(0, 2))
    epochs_start = 1
    epochs=math.ceil(random.uniform(3, 9))
    drop_mult=random.uniform(0.7, 1)
    layer_factor=2.6 #random.uniform(2, 3)
    exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")
    backwards = False
    dont_emb = 0

@ex.main
def my_main(epochs, drop_mult, exp_id, bs, layer_factor, epochs_start):    
    ex.info['path'] = EX_PA
    
    data_lm = TextLMDataBunch.load(EX_PA, bs=bs)
    learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=drop_mult, embed_prevent_first=0)
    
    lr = news_utils.fastai.get_optimal_lr(learn, runs=3)
    
    learn.callbacks += [
        news_utils.fastai.SacredLogger(learn, ex),
        SaveModelCallback(learn, name=exp_id),
        EarlyStoppingCallback(learn, patience=2)
    ]
    
    learn.fit_one_cycle(epochs_start, lr)
    
    learn.unfreeze()
    
    learn.fit_one_cycle(epochs, [lr / (layer_factor * (3 - x)) for x in range(3)] + [lr])
    
ex.run()
