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
from bpemb import BPEmb

import news_utils.clean.german


EX_PA = Path('/mnt/data/group07/johannes/ompc/pplmexp_short2')

ex = Experiment('shortppompclm2')
from sacred.observers import FileStorageObserver

# ex.observers.append(MongoObserver.create(db_name='shortppompclm2'))
ex.observers.append(FileStorageObserver.create('my_runs'))


@ex.config
def my_config():
    bs=128
    epochs_start = 0 
    epochs= 10 #math.ceil(random.uniform(1, 5))
    drop_mult= 0.5559 #random.uniform(0.5, 1)

    exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")
    model_id = '2019_ 3_27_14_30_09_921754' # best model after 5 epochs

@ex.main
def my_main(epochs, drop_mult, exp_id, bs, epochs_start, model_id):    
    print(exp_id)
    ex.info['path'] = EX_PA
    layer_factor = 2.6
        
    data_lm_ft = TextLMDataBunch.load(EX_PA)
    
    learn = language_model_learner(data_lm_ft, drop_mult=drop_mult)
    learn.load_pretrained(Path('/mnt/data/group07/johannes/germanlm/exp_1/models/' + model_id +'.pth'), Path('/mnt/data/group07/johannes/germanlm/exp_1/tmp/itos.pkl'))
    
    # learn.load('2019_ 4_06_18_53_42_028549') 
    lr = news_utils.fastai.get_optimal_lr(learn, runs=1)
    #lr = 0.001
    
    learn.callbacks += [
        news_utils.fastai.SacredLogger(learn, ex),
        SaveModelCallback(learn, name=exp_id),
        EarlyStoppingCallback(learn, patience=1)
    ]
    
    if epochs_start > 0:
        learn.fit_one_cycle(epochs_start, lr)
    
    learn.unfreeze()
    if epochs > 0: 
        #learn.fit_one_cycle(epochs, [lr / (layer_factor * (3 - x)) for x in range(3)] + [lr])
        learn.fit_one_cycle(epochs, lr)
    
ex.run()
