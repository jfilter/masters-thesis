#!/usr/bin/env python
# coding: utf-8

# In[16]:


import fastText
import pandas as pd
from pathlib import Path

import argparse


import json
from cleantext import clean
import random

import sklearn.metrics


parser = argparse.ArgumentParser()
parser.add_argument("--cl")
parser.add_argument("--path")

args = parser.parse_args()

dct = json.loads(Path(args.path).read_text())

lr = dct['lr']
wordNgrams = dct['wordNgrams']
minCount = dct['minCount']
epoch = dct['epoch']
cls = args.cl

# In[17]:


all_classes = ['claudience', 'clpersuasive', 'clagreement', 'cldisagreement', 'clinformative', 'clmean', 'clcontroversial', 'cltopic', 'clsentiment']


# In[18]:


def each_row(row):
    text = clean(row['text'], lower=True, no_line_breaks=True, zero_digits=True, fix_unicode=True, to_ascii=True)
    text = f"__label__{int(row[cls])} {text}"
    return text


# In[19]:


def to_file(df, cls, kind):
    df = df.dropna(subset=[cls])
    df['out'] = df.apply(each_row, axis=1)
    out_p = f"/mnt/data/group07/johannes/ynacc_proc/fasttext_baseline/cls/{cls}/"
    Path(out_p).mkdir(parents=True, exist_ok=True) 
    out_text = '\n'.join(df['out'].values.tolist())
    Path(out_p + kind).write_text(out_text)
    return str(Path(out_p + kind))


# In[20]:


def run(ps, i):
#     lr = random.uniform(0, 1)
#     epoch=round(random.uniform(5, 50))
#     wordNgrams=round(random.uniform(1, 5))
#     minCount=round(random.uniform(1, 10))
    model = fastText.train_supervised(input=ps[0], lr=lr, epoch=epoch, wordNgrams=wordNgrams, minCount=minCount)

#     had to do it like this because I want to get the prediction and not just a metric from the model
    preds = Path(ps[1]).read_text().split('\n')
    
    truth = []
    output = []
    for p in preds:
        label = p[:10]
        text =  p[11:]
        truth.append(label)
        output.append(model.predict(text)[0][0])

    rpt = sklearn.metrics.classification_report(truth, output, output_dict=True)

    rpt['lr'] = lr
    rpt['epoch'] = epoch
    rpt['wordNgrams'] = wordNgrams
    rpt['minCount'] = minCount
    rpt['kappa'] = sklearn.metrics.cohen_kappa_score(truth, output)
    Path(str(args.path) + 'test.json').write_text(json.dumps(rpt))


# In[21]:


# for cls in all_classes:
ps = []
for x in ['train.csv', 'test.csv']:
    df = pd.read_csv('/mnt/data/group07/johannes/ynacc_proc/replicate/split/' + x, usecols=['text', cls])
    ps.append(to_file(df, cls, x))
for i in range(1):
    run(ps, i)


# In[ ]:





# In[ ]:





# In[ ]:




