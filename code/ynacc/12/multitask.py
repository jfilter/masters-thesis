#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from types import SimpleNamespace

args = SimpleNamespace(**{'exp': 'only_threads_unlimited_30000', 'best': '2019_ 1_16_20_07_47_497762', 'device': 3})

EX_PA = Path('/mnt/data/group07/johannes/ynacc_proc/proper_threads/exp/cl/' + args.exp)

# torch.cuda.set_device(args.device)

print(fastai.__version__)


best_lm_exp_id = args.best

# all_classes = ['claudience', 'clpersuasive', 'clsentiment', 'clagreement', 'cldisagreement', 'clinformative', 'clmean', 'clcontroversial', 'cltopic']
all_classes = ['claudience', 'clpersuasive', 'clagreement', 'cldisagreement', 'clinformative', 'clmean', 'clcontroversial', 'cltopic']

data_lm = TextLMDataBunch.load(Path('/mnt/data/group07/johannes/ynacc_proc/proper_threads/exp/lm/' + args.exp))
learn_lm = language_model_learner(data_lm).load(
    Path('/mnt/data/group07/johannes/ynacc_proc/proper_threads/exp/lm/'+ args.exp  + "/models/" + best_lm_exp_id, device="cpu"))
learn_lm.save_encoder('encoder_' + best_lm_exp_id)
shutil.move('/mnt/data/group07/johannes/ynacc_proc/proper_threads/exp/lm/'+ args.exp  + "/models/"  + 'encoder_' + best_lm_exp_id + '.pth', '/mnt/data/group07/johannes/ynacc_proc/proper_threads/exp/cl/'+ args.exp  + "/models/" + 'encoder_' + best_lm_exp_id + '.pth')
learn_lm_vocab = data_lm.train_ds.vocab
del data_lm
del learn_lm

print('saved enconder, best model id:', best_lm_exp_id)


# In[2]:


def setup_data():
    UT = Path('~/data/ynacc_proc/proper_threads/data/cls/' + args.exp)
    
    data_clas_train = pd.read_csv(UT/'train.csv')
    data_clas_val = pd.read_csv(UT/'val.csv')

#     data_clas_train = data_clas_train[[clas, 'text_proc']]
#     data_clas_val = data_clas_val[[clas, 'text_proc']]

    data_clas_train = data_clas_train.dropna(subset=all_classes)
    data_clas_val = data_clas_val.dropna(subset=all_classes)
    
    for clas in all_classes:
        data_clas_train[clas] = data_clas_train[clas].astype(int)
        data_clas_val[clas] = data_clas_val[clas].astype(int)

    data_clas = TextClasDataBunch.from_df(EX_PA, data_clas_train, data_clas_val,
                                          vocab=learn_lm_vocab, bs=50, text_cols=['text_proc'], label_cols=all_classes,tokenizer=Tokenizer(cut_n_from_behind=1398))
    return data_clas


# In[3]:


data_clas = setup_data()


# In[4]:


data_clas.one_batch()


# In[5]:


encoder_name = 'encoder_' + best_lm_exp_id
drop_mult = 1

learn = text_classifier_learner(data_clas, drop_mult=drop_mult, embed_prevent_first=6)
learn.load_encoder(encoder_name)

optim_lr = news_utils.fastai.get_optimal_lr(learn, runs=3)


# In[6]:


def accuracy(input:Tensor, targs:Tensor)->Rank0Tensor:
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    print(input.tolist())
    print(targs.tolist())

    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
#     targs = targs.view(n,-1)
    targs = targs.view(-1).long()
    print(targs.tolist())
    print(input.tolist())
    return (input==targs).float().mean()


# In[7]:


# optim_lr = 0.0042854852039743915

#     @ex.config
#     def my_config():
exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")
factor = 2.6
wd = 1e-7
moms = (0.8, 0.7)
full_epochs = 20
bs = 50
embed_prevent=6
# lm_model_type='trained_0_embed_prevent'

#     @ex.main
#     def run_exp(exp_id, drop_mult, lr, moms, wd, factor, full_epochs):

lr = optim_lr

lrs = [lr / (factor ** (4 - x)) for x in range(4)] + [lr]

learn = text_classifier_learner(data_clas, drop_mult=drop_mult, embed_prevent_first=6)
learn.load_encoder(encoder_name)
learn.metrics = [accuracy]
learn.metrics =[fbeta]

# learn.metrics += [news_utils.fastai.F1Macro(),
#                   news_utils.fastai.F1Weighted(), news_utils.fastai.PrecisionMacro(), news_utils.fastai.RecallMacro()]

learn.callbacks += [
    SaveModelCallback(learn, name=exp_id),
#     news_utils.fastai.SacredLogger(learn, ex),
]


# In[8]:


# learn.fit_one_cycle(1, np.array(lrs), wd=wd, moms=moms)


# In[9]:


for i in range(1, 4):
    epochs = 1
    if i in [1, 2]:
        learn.freeze_to(-i)
    else:
        learn.unfreeze()
        epochs = full_epochs
    learn.fit_one_cycle(epochs, np.array(lrs), wd=wd, moms=moms)

#     for _ in range(it):
#         ex.run(config_updates={"lr": optim_lr, "drop_mult": drop_mult})


# run_for_class(all_classes[args.cl])


# In[10]:


data_clas.valid_dl


# In[21]:


learn.predict('that is cool')


# In[14]:


b1, b2 = learn.get_preds()


# In[24]:


for i in range(8):
    preds = [round(x.item()) for x in b1[i]]
    targs = [round(x.item()) for x in b2[i]]
    print(all_classes[i])
    print(metrics.f1_score(targs, preds, average="micro"))
    print(metrics.f1_score(targs, preds, average="macro"))
    print()


# In[ ]:




