from functools import reduce
import math

import sklearn 
import sklearn.metrics
import pandas as pd
import altair as alt
from ipysankeywidget import SankeyWidget
from altair.expr import datum
import numpy as np
from IPython.display import display

def plot_values(df, col_name):
    alt.renderers.enable('notebook')
    alt.Chart(pd.DataFrame(df[col_name].value_counts(dropna=False)).reset_index()).mark_bar().encode(
        x='index:N',
        y=col_name).display()

def class_report(ln):
    preds, y_true = ln.get_preds()
    _, y_pred = preds.max(dim=1)
    return sklearn.metrics.classification_report(y_true, y_pred, output_dict=True)

def plot_class_report(learn):
    alt.renderers.enable('notebook')
    clsrpt = class_report(learn)
    charts = []
    for key in clsrpt.keys():
        fildata = clsrpt[key]
        fildatax = []
        fildatay = []
        supp = key + ', Support: '
        for xxx in fildata:
            if xxx == 'support':
                supp += str(fildata[xxx])
                continue
            fildatax.append(xxx)
            fildatay.append(fildata[xxx])
        df = pd.DataFrame({'x': fildatax, 'y': fildatay})
        bars = alt.Chart(df, width=200).mark_bar(size=30).encode(
            x=alt.X("x", axis = alt.Axis(labelAngle=0, title='')),
            y=alt.Y('y', axis=alt.Axis(title=''), scale=alt.Scale(domain=(0, 1))),
        )

        text = alt.Chart(df).mark_text(baseline='bottom', dy=-1).encode(
            x='x',
            y='y',
            text=alt.Text('y',format='.2f')
        )

        chart = bars + text

        charts.append(chart.properties(
            title=supp
        ))
        
    return reduce((lambda x, y: alt.hconcat(x, y)), charts)

def plot_learning(learn, truncate_y=True):
    if len(learn.recorder.val_losses) >= 5000:
        return
    losses = [t.item() for t in learn.recorder.losses]
    left_for_loss = 5000 - len(learn.recorder.val_losses)
    if left_for_loss <= 0:
        left_for_loss = 1 / (abs(left_for_loss) + 1)
    if len(losses) > left_for_loss:
        # Split `x` up in chunks of 3
        chunk_size = len(learn.recorder.losses) / left_for_loss
        chunk_size = math.ceil(chunk_size)
        chunks = zip(*[iter(losses)]*chunk_size)
        losses = [sum(c)/len(c) for c in chunks]
    factor = int(len(losses) / len(learn.recorder.val_losses))
    
    print(losses)

    data1 = pd.DataFrame({'Type': 'val', 'Loss': learn.recorder.val_losses, 'Epoch': range(1, len(learn.recorder.val_losses) + 1)})
    data2 = pd.DataFrame({'Type': 'train', 'Loss': losses, 'Epoch': [x / factor for x in range(1, len(losses) + 1)]})


    data = pd.concat([data1, data2])
    
    charts = []
    
    ch1 = alt.Chart(data).mark_line().encode(
        x='Epoch:Q',
        y=alt.Y('Loss', scale=alt.Scale(zero=not truncate_y)),
        color='Type',
        tooltip=['Epoch', 'Loss', 'Type']
    ).interactive()
    
    charts.append(ch1)
    metrics = None
    if len(learn.recorder.metrics) > 0:
        metrics = [x[0] for x in learn.recorder.metrics] 
    if not metrics is None:
        ch2 = alt.Chart(pd.DataFrame({'Epoch': range(1, len(metrics) + 1), 'Metric': metrics})).mark_line().encode(
            x='Epoch:Q', 
            y=alt.Y('Metric', scale=alt.Scale(zero=not truncate_y)),
            tooltip=['Epoch', 'Metric']
        ).interactive()
        
        charts.append(ch2)
        
    return reduce((lambda x, y: alt.hconcat(x, y)), charts)

def get_confusions(ln):
    preds, y_true = ln.get_preds()
    _, y_pred = preds.max(dim=1)
    return sklearn.metrics.confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(ln, labels=None):
    conf = get_confusions(ln)
    if labels is None:
        x, y = np.meshgrid(range(0, conf.shape[0]), range(0, conf.shape[0]))
    else:
        x, y = np.meshgrid(labels, labels)


    # Convert this grid to columnar data expected by Altair
    data = pd.DataFrame({'true': x.ravel(),
                         'pred': y.ravel(),
                         'num': conf.ravel()})

    ch1=alt.Chart(data, height=200, width=200).mark_rect().encode(
        x='true:N',
        y='pred:N',
        tooltip=['num'],
        color=alt.Color('num:Q', scale=alt.Scale(scheme='blues')))

    ch2=alt.Chart(data, height=200, width=200).mark_text().encode(
        x='true:N',
        y='pred:N',
        text='num:Q',
        color=alt.condition(datum['num'] < (conf.max() + conf.min()) / 2,
                            alt.value('black'),
                            alt.value('white')))

    comb = ch1 + ch2
    comb.configure_axis(labels=True, ticks=False)
    return comb

def plot_conf_sankey(ln, labels=None):
    conf = get_confusions(ln)
    df = pd.DataFrame(columns=['source', 'target', 'type', 'value'])
    for (i, j), val in np.ndenumerate(conf):
        sou_label = 'true ' + (str(i) if labels is None else labels[i])
        pred_label = 'pred ' + (str(j) if labels is None else labels[j])

        df = df.append({'source': sou_label, 'target': pred_label, 'value': val, 'type': 'cor' if i == j else 'other'}, ignore_index=True)

    return SankeyWidget(links=df.to_dict('records'))

def all(ln, tr=False,labels=None, truncate_y=True):
    cp = plot_class_report(ln)
    if tr:
        train = plot_learning(ln, truncate_y=truncate_y)
    conf = plot_confusion_matrix(ln, labels)
    sank = plot_conf_sankey(ln, labels)
    if tr:
        display(cp, train, conf, sank)
    else:
        display(cp, conf, sank)
