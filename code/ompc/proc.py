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

import dask.dataframe as ddf
import dask
from dask.diagnostics import ProgressBar

from numba import jit

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--i', type=int)
args = parser.parse_args()

df_all = pd.read_pickle(f'/mnt/data/group07/johannes/ompc/res_split_{args.i}.pkl')


bpemb_de = BPEmb(lang="de", vs=25000, dim=300)

def cut(res):
    last_id = res.rfind('xxe')
    res = res[:last_id] + ' xxe'
    txt = bpemb_de.encode_ids_with_bos_eos(news_utils.clean.german.clean_german(res[-3000:]))
    return txt


# df_all = df_all[(df_all['ID_Article'] >= args.i * step_size) & (df_all['ID_Article'] < (args.i + 1) * step_size)]

df_all['res'] = df_all['res'].apply(cut)

df_all.to_pickle('/mnt/data/group07/johannes/ompc/res_proc7_' + str(args.i) + '.pkl')
