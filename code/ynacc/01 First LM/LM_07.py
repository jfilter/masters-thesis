
# coding: utf-8

# In[1]:


from fastai.text import *
from fastai.datasets import *
from pathlib import Path
import pandas as pd
from fastai.metrics import *
from fastai.train import *
from fastai.vision import *
from fastai.imports import nn, torch
from sklearn import metrics
import sklearn.metrics
torch.cuda.set_device(0)


# In[2]:


EX_PA = Path('/mnt/data/group07/johannes/ynacc_proc/clean_split/')
LM_DATA_PATH = Path('/mnt/data/group07/johannes/ynacc_proc/clean_split/lm_data/')


# In[4]:


tokenizer = Tokenizer(special_cases = ['xxbos','xxfld','xxunk','xxpad', 'xxsep'])
# tokenizer = None


# In[5]:


# Language model data
data_lm = TextLMDataBunch.from_csv(LM_DATA_PATH, valid='val', tokenizer=tokenizer)

data_clas = TextClasDataBunch.from_csv(EX_PA, vocab=data_lm.train_ds.vocab, bs=33, train="train", valid='val', txt_cols=['text'], label_cols=['class'])


# In[ ]:



learn = RNNLearner.language_model(data_lm, pretrained_fnames=['lstm_wt103', 'itos_wt103'], drop_mult=0.5)
learn.fit_one_cycle(20, 1e-2)
learn.fit_one_cycle(20, 1e-2)


print(learn.recorder.losses)
print(learn.recorder.val_losses)
print(learn.recorder.metrics)
# In[ ]:

learn.save_encoder(EX_PA/'ft_enc_05_20_20')

