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



EX_PA = Path('/mnt/data/group07/johannes/ompc/lmexp')

ex = Experiment('ompclm')
ex.observers.append(MongoObserver.create(db_name='ompclm'))


@ex.config
def my_config():
    bs=128
    epochs_start = 0 
    epochs=5 #math.ceil(random.uniform(1, 3))
    drop_mult=0  #random.uniform(1, 2)
    exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")
    model_id = '2019_ 3_27_14_30_09_921754' # best model after 5 epochs

@ex.main
def my_main(epochs, drop_mult, exp_id, bs, epochs_start, model_id):    
    ex.info['path'] = EX_PA
    layer_factor = 2
    
    bpemb_de = BPEmb(lang="de", vs=25000, dim=300)
    itos = dict(enumerate(bpemb_de.words + ['xxpad']))
    voc = Vocab(itos)    
    
    df_all = pd.read_pickle('/mnt/data/group07/johannes/ompc/unla.pkl')
    df_all['text_cat'] = df_all.apply(lambda x: (x['Headline'] if not x['Headline'] is None else '') + ' ' + (x['Body'] if not x['Body'] is None else '') + ' xxp ' + ('xxa' if pd.isna(x['ID_Parent_Post']) else 'xxb') , axis=1)

    df_all['text_ids'] = df_all['text_cat'].apply(lambda x: bpemb_de.encode_ids_with_bos_eos(news_utils.clean.german.clean_german(x)))

    df_all_train = df_all[df_all['ID_Article'] < 11500]
    df_all_val = df_all[df_all['ID_Article'] >= 11500]

    data_lm_ft = TextLMDataBunch.from_ids(bs=bs, path=EX_PA,vocab=voc, train_ids=df_all_train['text_ids'], valid_ids=df_all_val['text_ids'])
    
    learn = language_model_learner(data_lm_ft, drop_mult=drop_mult)
    learn.load_pretrained(Path('/mnt/data/group07/johannes/germanlm/exp_1/models/' + model_id +'.pth'), Path('/mnt/data/group07/johannes/germanlm/exp_1/tmp/itos.pkl'))
    
    
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
