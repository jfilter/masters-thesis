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
import os

from sacred import Experiment
from sacred.observers import MongoObserver
import fastai
import news_utils

ex = Experiment('classify sentiment')

# ex.observers.append(MongoObserver.create(db_name='classify'))

from sacred.observers import FileStorageObserver

ex.observers.append(FileStorageObserver.create('my_runs'))

EX_PA = Path('/mnt/data/group07/johannes/ynacc_proc/replicate/lmmodels')

@ex.config
def my_config():
    bs=50
    drop_mult=random.uniform(1.2, 1.8)
    lr=random.uniform(1e-4, 1e-6)
    layer_factor=random.uniform(2, 4)
    wd = random.choice([1e-7, 1e-6])
    exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")
    encoder = random.choice(['encoder_2018_11_12_00_40_36_447657', 'encoder_2018_11_12_03_55_59_263509', 'encoder_2018_11_12_07_27_32_842777'])


@ex.automain
def my_main(lr, drop_mult, exp_id, bs, layer_factor, wd, encoder):
#     print(os.environ['CUDA_VISIBLE_DEVICES'])
#     torch.cuda.set_device(int(os.environ['CUDA_VISIBLE_DEVICES']))
#     torch.cuda.set_device(1)
    
    ex.info['path'] = EX_PA
    data_clas = TextClasDataBunch.load(EX_PA, 'textclassent', bs=bs)

    lrs = [lr / (layer_factor ** (4 - x)) for x in range(4)] + [lr]

    torch.cuda.set_device(2)

    learn = text_classifier_learner(data_clas, drop_mult=drop_mult)
    learn.load_encoder(encoder)

    learn.metrics += [news_utils.fastai.F1Macro(), news_utils.fastai.F1Weighted()]
    learn.callbacks += [
        news_utils.fastai.SacredLogger(learn, ex),
        SaveModelCallback(learn, name=exp_id),
    ]

    for i in range(1, 4):
        epochs = 5
        if i in [1, 2]:
            learn.freeze_to(-i)
        else:
            learn.unfreeze()
            learn.callbacks += [EarlyStoppingCallback(learn, patience=5)]
            epochs = 40
        learn.fit_one_cycle(epochs, lrs, wd=wd)
