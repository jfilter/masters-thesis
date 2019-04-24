#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python


# In[10]:


import argparse
import datetime
from pathlib import Path
import shutil

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
parser.add_argument("--cl", type=int)
parser.add_argument("--best")
args = parser.parse_args()

EX_PA = Path('/mnt/data/group07/johannes/ynacc_proc/proper_threads/exp/cl/' + args.exp)

# torch.cuda.set_device(args.device)


# In[11]:


print(fastai.__version__)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient[args.exp + 'lm']

db_name = args.exp + '_cl'
# In[27]:

if args.best is None:
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
else:
    best_lm_exp_id = args.best

#In[ ]:


data_lm = TextLMDataBunch.load(Path('/mnt/data/group07/johannes/ynacc_proc/proper_threads/exp/lm/' + args.exp))
learn_lm = language_model_learner(data_lm).load(
    Path('/mnt/data/group07/johannes/ynacc_proc/proper_threads/exp/lm/'+ args.exp  + "/models/" + best_lm_exp_id, device="cpu"))
learn_lm.save_encoder('encoder_' + best_lm_exp_id)
shutil.move('/mnt/data/group07/johannes/ynacc_proc/proper_threads/exp/lm/'+ args.exp  + "/models/"  + 'encoder_' + best_lm_exp_id + '.pth', '/mnt/data/group07/johannes/ynacc_proc/proper_threads/exp/cl/'+ args.exp  + "/models/" + 'encoder_' + best_lm_exp_id + '.pth')
learn_lm_vocab = data_lm.train_ds.vocab
del data_lm
del learn_lm

print('saved enconder, best model id:', best_lm_exp_id)

def setup_data(clas):
    UT = Path('~/data/ynacc_proc/proper_threads/data/cls/' + args.exp)
    
    data_clas_train = pd.read_csv(UT/'train.csv')
    data_clas_val = pd.read_csv(UT/'val.csv')

    data_clas_train = data_clas_train[[clas, 'text_proc']]
    data_clas_val = data_clas_val[[clas, 'text_proc']]

    data_clas_train = data_clas_train.dropna()
    data_clas_val = data_clas_val.dropna()

    data_clas_train[clas] = data_clas_train[clas].astype(int)
    data_clas_val[clas] = data_clas_val[clas].astype(int)

    data_clas = TextClasDataBunch.from_df(EX_PA, data_clas_train, data_clas_val,
                                          vocab=learn_lm_vocab, bs=50, text_cols=['text_proc'], label_cols=[clas],tokenizer=Tokenizer(cut_n_from_behind=1398))
    return data_clas


def run_for_class(clas, it=5):
    print('work on ' + clas)
    torch.cuda.empty_cache()
    data_clas = setup_data(clas)
    encoder_name = 'encoder_' + best_lm_exp_id
    drop_mult = 1

    learn = text_classifier_learner(data_clas, drop_mult=drop_mult, embed_prevent_first=6)
    learn.load_encoder(encoder_name)

    optim_lr = news_utils.fastai.get_optimal_lr(learn, runs=3)

    ex = Experiment(db_name + '_' + clas)
    ex.observers.append(MongoObserver.create(db_name=db_name + '_' + clas))

    @ex.config
    def my_config():
        exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")
        factor = 2.6
        wd = 1e-7
        moms = (0.8, 0.7)
        full_epochs = 20
        bs = 50
        embed_prevent=6
        lm_model_type='trained_6_embed_prevent'

    @ex.main
    def run_exp(exp_id, drop_mult, lr, moms, wd, factor, full_epochs):

        lrs = [lr / (factor ** (4 - x)) for x in range(4)] + [lr]

        learn = text_classifier_learner(data_clas, drop_mult=drop_mult, embed_prevent_first=6)
        learn.load_encoder(encoder_name)

        learn.metrics += [news_utils.fastai.F1Macro(),
                          news_utils.fastai.F1Weighted(), news_utils.fastai.PrecisionMacro(), news_utils.fastai.RecallMacro()]

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

    for _ in range(it):
        ex.run(config_updates={"lr": optim_lr, "drop_mult": drop_mult})


all_classes = ['claudience', 'clpersuasive', 'clsentiment', 'clagreement', 'cldisagreement', 'clinformative', 'clmean', 'clcontroversial', 'cltopic']
run_for_class(all_classes[args.cl])

