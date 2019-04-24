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

ex = Experiment('first layer LM ynacc, then all layers')

ex.observers.append(MongoObserver.create())

EX_PA = Path('/mnt/data/group07/johannes/ynacc_proc/replicate/lmmodels')

@ex.config
def my_config():
    bs=64
    epochs=math.ceil(random.uniform(5, 20))
    drop_mult=random.uniform(0.15, 0.3)
    lr=random.uniform(1e-3, 1e-7)
    layer_factor=random.uniform(2, 5)
    exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")
    backwards = False
    frozen = 'none'
    based_on_model = random.choice(['2018_11_11_00_32_45_329728', '2018_11_11_16_27_24_932260', '2018_11_11_15_31_08_374724'])

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]   

@ex.automain
def my_main(epochs, lr, drop_mult, exp_id, based_on_model, layer_factor):
    torch.cuda.set_device(1)
    
    ex.info['path'] = EX_PA
    
    data_lm = TextLMDataBunch.load(EX_PA)
    learn = language_model_learner(data_lm, drop_mult=drop_mult).load(EX_PA/"models"/based_on_model)
    
    learn.unfreeze()

    learn.callbacks += [
        SaveModelCallback(learn, name=exp_id),
        CSVLogger(learn, 'history/' + exp_id ),
        EarlyStoppingCallback(learn, patience=5)
    ]
    
    learn.fit_one_cycle(epochs, [lr / (layer_factor * (4 - x)) for x in range(4)])
    
    ex.info['recorder'] = learn.recorder
    
#     save all the losses / metrics

    losses_fixed = [x.item() for x in learn.recorder.losses]
    
    res = []
    for x in learn.recorder.metrics:
        res_inner = []
        for y in x:
            res_inner.append(y.item())
        res.append(res_inner)
    
    for r in res:
        ex.log_scalar("acc", r[0])
     
    for x in learn.recorder.val_losses:
        ex.log_scalar('val loss', x)

    for x in losses_fixed:
        ex.log_scalar('loss', x)
    
    for ch in chunks(losses_fixed, math.ceil(len(losses_fixed)/len(learn.recorder.val_losses))):
        ex.log_scalar('loss avg', sum(ch) / len(ch))

  
