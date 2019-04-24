from fastai.text import *
from fastai.datasets import *
from pathlib import Path
import pandas as pd
from fastai.metrics import *
from fastai.train import *
from fastai.imports import nn, torch
from fastai.callbacks import *

import random
import math
import datetime
from sacred import Experiment

from sacred.observers import MongoObserver
import fastai
import  news_utils.fastai

import news_utils

ex = Experiment('sec lm')

ex.observers.append(MongoObserver.create(db_name='seclm'))

EX_PA = Path('/mnt/data/group07/johannes/ynacc_proc/replicate/lmmodels2')

@ex.config
def my_config():
    bs=64
    epochs=math.ceil(random.uniform(5, 20))
    drop_mult=random.uniform(0.1, 0.5)
    lr=random.uniform(1e-3, 1e-4)
    exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")
    backwards = False
    frozen = 'last' 

@ex.automain
def my_main(epochs, lr, drop_mult, exp_id):    
    ex.info['path'] = EX_PA
    
    data_lm = TextLMDataBunch.load(EX_PA)
    learn = language_model_learner(data_lm, pretrained_fnames=['lstm_wt103', 'itos_wt103'], drop_mult=drop_mult)

    learn.callbacks += [
        news_utils.fastai.SacredLogger(learn, ex),
        SaveModelCallback(learn, name=exp_id),
        EarlyStoppingCallback(learn, patience=5)
    ]
    
    learn.fit_one_cycle(epochs, lr)
    
    ex.info['recorder'] = learn.recorder
    


