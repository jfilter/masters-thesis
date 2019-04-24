#!/usr/bin/env python
# coding: utf-8

# In[34]:



from fastai.text import *
from fastai.text.data import DataBunch
from fastai.datasets import *
from pathlib import Path
import pandas as pd
from fastai.metrics import *
from fastai.train import *
from fastai.vision import *
from fastai.imports import nn, torch
from sklearn import metrics
from fastai.callbacks import *
from fastai.basic_train import get_preds

import sacred
from sacred import Experiment

from sacred.observers import MongoObserver

import sklearn.metrics
import datetime
import news_utils
from pathlib import Path

import fastai
fastai.__version__

torch.cuda.set_device(1)


# In[2]:


EX_PA = Path('/mnt/data/group07/johannes/ynacc_proc/replicate/20k')
model_id = '2018_11_26_13_59_38_368409'# best without 
db_name = '20k_class_best'


# In[4]:


data_lm = TextLMDataBunch.load(EX_PA)


# In[3]:


learn_lm = language_model_learner(data_lm).load(EX_PA/"models"/model_id, device="cpu")
learn_lm.save_encoder('encoder_' + model_id)


# In[5]:


def setup_data(clas):
    split_path = Path('~/data/ynacc_proc/replicate/split')

    data_clas_train = pd.read_csv(split_path/'train_proc_with_ner.csv')
    data_clas_val = pd.read_csv(split_path/'val_proc_with_ner.csv')

    data_clas_train = data_clas_train[[clas, 'text_proc']]
    data_clas_val = data_clas_val[[clas, 'text_proc']]

    data_clas_train = data_clas_train.dropna()
    data_clas_val = data_clas_val.dropna()

    data_clas_train[clas] = data_clas_train[clas].astype(int)
    data_clas_val[clas] = data_clas_val[clas].astype(int)

    data_clas = TextClasDataBunch.from_df(EX_PA, data_clas_train, data_clas_val,
                                          vocab=data_lm.train_ds.vocab, bs=64, text_cols=['text_proc'], label_cols=[clas],)
    return data_clas


# In[28]:


def run_for_class(clas, it=5):
    data_clas = setup_data(clas)
    encoder_name = 'encoder_' + model_id
    drop_mult = 1

    learn = text_classifier_learner(data_clas, drop_mult=drop_mult)
    learn.load_encoder(encoder_name)

    optim_lr = news_utils.fastai.get_optimal_lr(learn, runs=7)

    ex = Experiment(db_name + '_' + clas)
    ex.observers.append(MongoObserver.create(db_name=db_name))

    @ex.config    
    def my_config():
        exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")
        factor = 3
        wd = 1e-7
        moms = (0.8, 0.7)
        full_epochs = 10

    @ex.main
    def run_exp(exp_id, drop_mult, lr, moms, wd, factor, full_epochs):

        lrs = [lr / (factor ** (4 - x)) for x in range(4)] + [lr]
        
        learn = text_classifier_learner(data_clas, drop_mult=drop_mult)
        learn.load_encoder(encoder_name)

        learn.metrics += [news_utils.fastai.F1Macro(), news_utils.fastai.F1Weighted()]
        learn.callbacks += [
            SaveModelCallback(learn, name=exp_id),
        ]

        for i in range(1, 4):
            epochs = 1
            if i in [1, 2]:
                learn.freeze_to(-i)
            else:
                learn.unfreeze()
        #         learn.callbacks += [EarlyStoppingCallback(learn, patience=5)]
                epochs = full_epochs
        #             learn.fit_one_cycle(epochs, np.array(lrs) * 1 / (i ** 4), wd=wd, moms=moms)
            learn.fit_one_cycle(epochs, np.array(lrs), wd=wd, moms=moms)
    
    for _ in range(it):
        ex.run(config_updates={"lr": optim_lr, "drop_mult": drop_mult})


# In[ ]:

for x in ['claudience','clpersuasive','clsentiment','clagreement','cldisagreement','clinformative','clmean','clcontroversial', 'cltopic']:
    run_for_class(x)
