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

import news_utils

EX_PA = Path('/mnt/data/group07/johannes/ynacc_proc/replicate/20k_ner')

ex = Experiment(EX_PA.stem)

ex.observers.append(MongoObserver.create(db_name=EX_PA.stem))

@ex.config
def my_config():
    bs=64
    epochs=math.ceil(random.uniform(10, 30))
    drop_mult=random.uniform(0.4, 0.6)
    lr=random.uniform(5e-3, 5e-4)
    layer_factor=random.uniform(2, 3)
    exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")
    backwards = False
    frozen = 'none'
    based_on_model = '2018_11_22_22_09_44_500735'

@ex.automain
def my_main(epochs, lr, drop_mult, exp_id, bs, layer_factor, based_on_model):    
    ex.info['path'] = EX_PA
    
    torch.cuda.set_device(3)

    data_lm = TextLMDataBunch.load(EX_PA, bs=bs)
    learn = language_model_learner(data_lm, drop_mult=drop_mult).load(EX_PA/"models"/based_on_model, device="cuda:3")
    
    learn.callbacks += [
        news_utils.fastai.SacredLogger(learn, ex),
        SaveModelCallback(learn, name=exp_id),
        EarlyStoppingCallback(learn, patience=5)
    ]
        
    learn.unfreeze()
    
    learn.fit_one_cycle(epochs, [lr / (layer_factor * (3 - x)) for x in range(3)] + [lr])
  
