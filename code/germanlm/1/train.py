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

import news_utils.fastai

ex = Experiment('germanlm_raw')

ex.observers.append(MongoObserver.create(db_name='germanlm'))

EX_PA = Path('/mnt/data/group07/johannes/germanlm/exp_1')

@ex.config
def my_config():
    lemma = "no"
    vocab = "25k"
    drop_mult = 0
    # lr = 1e-2
    epochs = 5
    bs=128
    exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")

@ex.automain
def my_main(epochs, drop_mult, exp_id, bs):
#    torch.cuda.set_device(3)
    
    data = TextLMDataBunch.load(EX_PA, bs=bs)
    ex.info['path'] = EX_PA
    
    learn = language_model_learner(data, drop_mult=drop_mult)
    learn.unfreeze()

    lr = news_utils.fastai.get_optimal_lr(learn, runs=1)

    learn.callbacks += [
        news_utils.fastai.SacredLogger(learn, ex),
        SaveModelCallback(learn, name=exp_id),
        EarlyStoppingCallback(learn, patience=5)
    ]
    
    learn.fit_one_cycle(epochs, lr)
