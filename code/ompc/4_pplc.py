#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python


# In[10]:


import argparse
import datetime
from pathlib import Path
import shutil

import sklearn

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

from pathlib import Path

import pandas as pd
from bpemb import BPEmb

from fastai.callbacks import *
from fastai.datasets import *
from fastai.imports import nn, torch
from fastai.metrics import *
from fastai.text import *
from fastai.text.data import TextLMDataBunch
from fastai.text.transform import BaseTokenizer, Tokenizer, Vocab
from fastai.train import *
import fastai

import news_utils.fastai
import news_utils.clean.german


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument("--cl", type=int)
parser.add_argument("--fold", type=int)
args = parser.parse_args()

# data_lm_ft = TextLMDataBunch.load(path='/mnt/data/group07/johannes/ompc/pplmexp')

all_cats = ['ArgumentsUsed', 'Discriminating', 'Inappropriate', 'OffTopic', 'PersonalStories', 'PossiblyFeedback', 'SentimentNegative', 'SentimentNeutral',  'SentimentPositive']

fold = str(args.fold)
cat = all_cats[args.cl]
model_id = '2019_ 4_04_20_02_39_766228' 
model_id = '2019_ 4_05_02_05_04_069084'
model_id = '2019_ 4_05_11_54_23_327097'
model_id = '2019_ 4_06_18_53_42_028549'
model_id = '2019_ 4_07_20_36_23_662822'
model_id = '2019_ 4_07_20_35_19_533996'
model_id = '2019_ 4_07_23_55_08_571313'
exp_path = '/mnt/data/group07/johannes/ompc/ppexp_short4/' + cat + '_' + fold

data_lm_ft = TextLMDataBunch.load(Path('/mnt/data/group07/johannes/ompc/pplmexp_short4'))

if True or not Path('/mnt/data/group07/johannes/ompc/ppexp_short4/' + cat + '_' + fold +'/models/enc5.pth').is_file():
    if True or not Path('/mnt/data/group07/johannes/ompc/pplmexp_short/models/enc5.pth').is_file():
        print('need to save enc')
        learn_lm = language_model_learner(data_lm_ft)
        learn_lm.load(model_id)
        learn_lm.save_encoder('enc5')
        del learn_lm
    print('need to copy enc')
    os.makedirs(os.path.dirname('/mnt/data/group07/johannes/ompc/ppexp_short4/' + cat + '_' + fold +'/models/enc5.pth'), exist_ok=True)
    shutil.copy('/mnt/data/group07/johannes/ompc/pplmexp_short4/models/enc5.pth', '/mnt/data/group07/johannes/ompc/ppexp_short4/' + cat + '_' + fold +'/models/enc5.pth')


def run_for_class(it=1):
    train_df = pd.read_pickle(Path('/mnt/data/group07/johannes/ompc/data_ann_pp_short4')/cat/fold/'train.pkl')
    test_df = pd.read_pickle(Path('/mnt/data/group07/johannes/ompc/data_ann_pp_short4')/cat/fold/'test.pkl')
    print(train_df.shape, test_df.shape)
    
    if cat == 'SentimentNeutral':
        train_df['Value'] = train_df['Value'].apply(lambda x: 1 if x == 0 else 0)
        test_df['Value'] = test_df['Value'].apply(lambda x: 1 if x == 0 else 0)
        print('fixing sentimal neutral')

    data = TextClasDataBunch.from_ids(pad_idx=25000, bs=64,path=exp_path, vocab=data_lm_ft.vocab, classes=[0, 1], train_lbls=train_df['Value'], valid_lbls=test_df['Value'], train_ids=train_df['res'], valid_ids=test_df['res'])

    drop_mult = 1
    exp_name = '4pp_' + cat + '_' + fold + '_' + str(drop_mult).replace('.', '_')
                                                                           
    ex = Experiment(exp_name)
    ex.observers.append(MongoObserver.create(db_name=exp_name))

    @ex.config
    def my_config():
        exp_id = datetime.datetime.now().strftime("%Y_%_m_%d_%H_%M_%S_%f")
        factor = 2.6
        full_epochs = 10
        bs = 64
        mod='simple_fit'
        lr = 0.001

    @ex.main
    def run_exp(exp_id, drop_mult, lr, factor, full_epochs):

        lrs = [lr / (factor ** (4 - x)) for x in range(4)] + [lr]

        learn = text_classifier_learner(data, drop_mult=drop_mult)
        
        learn.load_encoder('enc5')
        learn.metrics += [KappaScore(), news_utils.fastai.F1Bin(), news_utils.fastai.PrecBin(), news_utils.fastai.RecaBin()]
        learn.loss_func = torch.nn.CrossEntropyLoss(torch.FloatTensor(sklearn.utils.class_weight.compute_class_weight('balanced', [0, 1], train_df['Value'])).cuda())

        learn.callbacks += [
            SaveModelCallback(learn, name=exp_id, monitor='F1_bin'),
            EarlyStoppingCallback(learn, monitor='F1_bin', patience=80, mode='max'),
            news_utils.fastai.SacredLogger(learn, ex),
        ]

        for i in range(1, 0):
            epochs = 1
            if i in [1, 2, 3]:
                learn.freeze_to(-i)
                #if i == 2:
                    #epochs = full_epochs
            else:
                learn.unfreeze()
                epochs = full_epochs
            learn.fit_one_cycle(epochs, np.array(lrs), moms=(0.8,0.7))
        ep_fac = 3
        learn.fit_one_cycle(ep_fac, 2e-2, moms=(0.8,0.7))
        learn.freeze_to(-2)
        learn.fit_one_cycle(ep_fac, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
        learn.freeze_to(-3)
        learn.fit_one_cycle(ep_fac, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
        learn.unfreeze()
        learn.fit_one_cycle(2*ep_fac, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

    for _ in range(it):
        ex.run(config_updates={ "drop_mult": drop_mult})

run_for_class()
