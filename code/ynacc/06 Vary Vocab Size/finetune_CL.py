#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python


# In[10]:


import argparse
import datetime
from pathlib import Path

import fastai
import pandas as pd
import pymongo
import sacred
import sklearn.metrics
from fastai.basic_train import get_preds
from fastai.callbacks import *
from fastai.datasets import *
from fastai.imports import nn, torch
from fastai.metrics import *
from fastai.text import *
from fastai.text.data import DataBunch
from fastai.train import *
from fastai.vision import *
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn import metrics

import news_utils.fastai


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument("--exp")
parser.add_argument("--device", type=int)
parser.add_argument("--start", type=int)
args = parser.parse_args()

EX_PA = Path('/mnt/data/group07/johannes/ynacc_proc/replicate/' + args.exp)

torch.cuda.set_device(args.device)


# In[11]:


print(fastai.__version__)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient[args.exp]

db_name = args.exp + '_cl'
# In[27]:


myresults = mydb["metrics"].aggregate([{
    "$match": {"name": "valid_loss"}  # only consider val loss
},
    {"$unwind": "$values"},
    {"$group":
     {'_id': '$_id',
      'minval': {'$min': "$values"}, 'run_id' : { '$first': '$run_id' }}
     },  # find min values
    {"$sort": {"minval": 1}}  # sort
])

# get best run id in the metrics table
best_run_id = sorted(list(myresults), key=lambda x: x['minval'])[0]['run_id']

# get the exp id for the language model
best_lm_exp_id = list(mydb['runs'].find({'_id': best_run_id}))[0]['config']['exp_id']


# In[ ]:


data_lm = TextLMDataBunch.load(EX_PA)
learn_lm = language_model_learner(data_lm).load(
    EX_PA/"models"/best_lm_exp_id, device="cpu")
learn_lm.save_encoder('encoder_' + best_lm_exp_id)
learn_lm_vocab = data_lm.train_ds.vocab
del data_lm
del learn_lm


def setup_data(clas):
    split_path = Path('~/data/ynacc_proc/replicate/split')
    
    if args.exp.endswith('_ner'):
        data_clas_train = pd.read_csv(split_path/'train_ner.csv')
        data_clas_val = pd.read_csv(split_path/'val_ner.csv')
    else:
        data_clas_train = pd.read_csv(split_path/'train.csv')
        data_clas_val = pd.read_csv(split_path/'val.csv')        

    data_clas_train = data_clas_train[[clas, 'text_proc']]
    data_clas_val = data_clas_val[[clas, 'text_proc']]

    data_clas_train = data_clas_train.dropna()
    data_clas_val = data_clas_val.dropna()

    data_clas_train[clas] = data_clas_train[clas].astype(int)
    data_clas_val[clas] = data_clas_val[clas].astype(int)

    data_clas = TextClasDataBunch.from_df(EX_PA, data_clas_train, data_clas_val,
                                          vocab=learn_lm_vocab, bs=32, text_cols=['text_proc'], label_cols=[clas],)
    return data_clas


def run_for_class(clas, it=5):
    print('work on ' + clas)
    torch.cuda.empty_cache()
    data_clas = setup_data(clas)
    encoder_name = 'encoder_' + best_lm_exp_id
    drop_mult = 1

    learn = text_classifier_learner(data_clas, drop_mult=drop_mult)
    learn.load_encoder(encoder_name)

    optim_lr = news_utils.fastai.get_optimal_lr(learn, runs=3)

    torch.cuda.empty_cache()

    ex = Experiment(db_name + '_' + clas)
    ex.observers.append(MongoObserver.create(db_name=db_name + '_' + clas))

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

        learn.metrics += [news_utils.fastai.F1Macro(),
                          news_utils.fastai.F1Weighted()]
        learn.callbacks += [
            SaveModelCallback(learn, name=exp_id),
            news_utils.fastai.SacredLogger(learn, ex),
        ]

        for i in range(1, 4):
            epochs = 1
            if i in [1, 2]:
                learn.freeze_to(-i)
            else:
                learn.unfreeze()
                epochs = full_epochs
            learn.fit_one_cycle(epochs, np.array(lrs), wd=wd, moms=moms)
            torch.cuda.empty_cache()

    for _ in range(it):
        ex.run(config_updates={"lr": optim_lr, "drop_mult": drop_mult})


i = -1
for x in ['claudience', 'clpersuasive', 'clsentiment', 'clagreement', 'cldisagreement', 'clinformative', 'clmean', 'clcontroversial', 'cltopic']:
    torch.cuda.empty_cache()
    i += 1
    if not args.start is None and args.start > i:
        continue
    run_for_class(x)

