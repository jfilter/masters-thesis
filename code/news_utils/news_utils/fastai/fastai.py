from fastai.imports import torch, np
from fastai.callback import Callback
from fastai.basic_train import LearnerCallback
import sklearn.metrics

from dataclasses import dataclass
from typing import Any
from numbers import Number

# check out the content of a batch
def one_batch_text(data_lm, data_clas):
    a = data_clas.train_dl.one_batch()[0].transpose(1, 0)
    for x in a:
        print(data_lm.train_ds.vocab.textify(x))
        print()

def predict_next_word(learn, data, input_txt, n=100):
    u = data.train_ds.vocab.numericalize(input_txt)
    # make example a torch tensor
    value = torch.from_numpy(np.array(u))

    # then put it on the GPU, make it float and insert a fake batch dimension
    if torch.cuda.is_available():
        test_value = torch.autograd.Variable(value.cuda())
    else:
        test_value = torch.autograd.Variable(value.cpu())

    test_value = test_value.long()
    test_value = test_value.unsqueeze(0)
    res = learn.model(test_value)
    val, ind = torch.topk(res[0][0], n, largest=True)
    return data.train_ds.vocab.textify(ind), val

@dataclass
class F1Macro(Callback):
    name:str='F1_macro'
    
    def on_epoch_begin(self, **kwargs):
        self.y_pred, self.y_true = [], []
        
    def on_batch_end(self, last_output, last_target, **kwargs):
        _, idxs = torch.max(last_output, 1)
        self.y_pred += idxs.tolist()
        self.y_true += last_target.tolist()
    
    def on_epoch_end(self, **kwargs):
        f1_macro = sklearn.metrics.f1_score(self.y_pred, self.y_true, average='macro')
        self.metric = f1_macro

@dataclass
class F1Micro(Callback):
    name:str='F1_micro'
    
    def on_epoch_begin(self, **kwargs):
        self.y_pred, self.y_true = [], []
        
    def on_batch_end(self, last_output, last_target, **kwargs):
        _, idxs = torch.max(last_output, 1)
        self.y_pred += idxs.tolist()
        self.y_true += last_target.tolist()
    
    def on_epoch_end(self, **kwargs):
        f1_micro = sklearn.metrics.f1_score(self.y_pred, self.y_true, average='micro')
        self.metric = f1_micro


@dataclass
class RecallMicro(Callback):
    name:str='recal_micro'
    
    def on_epoch_begin(self, **kwargs):
        self.y_pred, self.y_true = [], []
        
    def on_batch_end(self, last_output, last_target, **kwargs):
        _, idxs = torch.max(last_output, 1)
        self.y_pred += idxs.tolist()
        self.y_true += last_target.tolist()
    
    def on_epoch_end(self, **kwargs):
        f1_micro = sklearn.metrics.recall_score(self.y_pred, self.y_true, average='micro')
        self.metric = f1_micro


@dataclass
class PrecisionMicro(Callback):
    name:str='precision_micro'
    
    def on_epoch_begin(self, **kwargs):
        self.y_pred, self.y_true = [], []
        
    def on_batch_end(self, last_output, last_target, **kwargs):
        _, idxs = torch.max(last_output, 1)
        self.y_pred += idxs.tolist()
        self.y_true += last_target.tolist()
    
    def on_epoch_end(self, **kwargs):
        f1_micro = sklearn.metrics.precision_score(self.y_pred, self.y_true, average='micro')
        self.metric = f1_micro


@dataclass
class RecallMacro(Callback):
    name:str='recal_macro'
    
    def on_epoch_begin(self, **kwargs):
        self.y_pred, self.y_true = [], []
        
    def on_batch_end(self, last_output, last_target, **kwargs):
        _, idxs = torch.max(last_output, 1)
        self.y_pred += idxs.tolist()
        self.y_true += last_target.tolist()
    
    def on_epoch_end(self, **kwargs):
        f1_micro = sklearn.metrics.recall_score(self.y_pred, self.y_true, average='macro')
        self.metric = f1_micro


@dataclass
class PrecisionMacro(Callback):
    name:str='precision_macro'
    
    def on_epoch_begin(self, **kwargs):
        self.y_pred, self.y_true = [], []
        
    def on_batch_end(self, last_output, last_target, **kwargs):
        _, idxs = torch.max(last_output, 1)
        self.y_pred += idxs.tolist()
        self.y_true += last_target.tolist()
    
    def on_epoch_end(self, **kwargs):
        f1_micro = sklearn.metrics.precision_score(self.y_pred, self.y_true, average='macro')
        self.metric = f1_micro

@dataclass
class F1Weighted(Callback):
    name:str='F1_weighted'
    
    def on_epoch_begin(self, **kwargs):
        self.y_pred, self.y_true = [], []
        
    def on_batch_end(self, last_output, last_target, **kwargs):
        _, idxs = torch.max(last_output, 1)
        self.y_pred += idxs.tolist()
        self.y_true += last_target.tolist()
    
    def on_epoch_end(self, **kwargs):
        f1_weighted = sklearn.metrics.f1_score(self.y_pred, self.y_true, average='weighted')
        self.metric = f1_weighted 


@dataclass
class F1Bin(Callback):
    name:str='F1_bin'
    
    def on_epoch_begin(self, **kwargs):
        self.y_pred, self.y_true = [], []
        
    def on_batch_end(self, last_output, last_target, **kwargs):
        _, idxs = torch.max(last_output, 1)
        self.y_pred += idxs.tolist()
        self.y_true += last_target.tolist()
    
    def on_epoch_end(self, **kwargs):
        f1_micro = sklearn.metrics.f1_score(self.y_pred, self.y_true, average='binary')
        self.metric = f1_micro


@dataclass
class PrecBin(Callback):
    name:str='prec_bin'
    
    def on_epoch_begin(self, **kwargs):
        self.y_pred, self.y_true = [], []
        
    def on_batch_end(self, last_output, last_target, **kwargs):
        _, idxs = torch.max(last_output, 1)
        self.y_pred += idxs.tolist()
        self.y_true += last_target.tolist()
    
    def on_epoch_end(self, **kwargs):
        f1_micro = sklearn.metrics.precision_score(self.y_pred, self.y_true, average='binary')
        self.metric = f1_micro


@dataclass
class RecaBin(Callback):
    name:str='reca_bin'
    
    def on_epoch_begin(self, **kwargs):
        self.y_pred, self.y_true = [], []
        
    def on_batch_end(self, last_output, last_target, **kwargs):
        _, idxs = torch.max(last_output, 1)
        self.y_pred += idxs.tolist()
        self.y_true += last_target.tolist()
    
    def on_epoch_end(self, **kwargs):
        f1_micro = sklearn.metrics.recall_score(self.y_pred, self.y_true, average='binary')
        self.metric = f1_micro




@dataclass
class SacredLogger(LearnerCallback):
    learn:Any=None
    experiment:Any=None

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        last_metrics = [] if last_metrics == None else last_metrics
        for name, stat in zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics):
            stat = stat if isinstance(stat, Number) else stat.item()
            self.experiment.log_scalar(name, stat)

def get_optimal_lr(learn, runs=7, **kwargs):
    run_lr = []
    for _ in range(runs):
        learn.lr_find(**{'num_it': 1000, **kwargs})
        best_lr_idx = np.argmin([l.item() for l in learn.recorder.losses])
        assert len(learn.recorder.losses) == len(learn.recorder.lrs)
        best_lr = learn.recorder.lrs[best_lr_idx]
        run_lr.append(best_lr)
    final_lr = np.median(run_lr) / 10 # deduct by one magnitude
    print('best lr: ' + str(final_lr))
    return final_lr


@dataclass
class MyTrackerCallback(LearnerCallback):
    "A `LearnerCallback` that keeps track of the best value in `monitor`."
    monitor1:str='trn_loss'
    monitor2:str='val_loss'
    mode:str='auto'

    def __post_init__(self):
        if self.mode not in ['auto', 'min', 'max']:
            warn(f'{self.__class__} mode {self.mode} is invalid, falling back to "auto" mode.')
            self.mode = 'auto'
        mode_dict = {'min': np.less, 'max':np.greater}
        mode_dict['auto'] = np.less if 'loss' in self.monitor1 else np.greater
        self.operator = mode_dict[self.mode]

    def on_train_begin(self, **kwargs:Any)->None:
        "Initializes the best value."
        self.best = float('inf') if self.operator == np.less else -float('inf')

    def get_monitor1_value(self):
        "Pick the monitored value."
        if self.monitor1=='trn_loss' and len(self.learn.recorder.losses) == 0: return None
        elif len(self.learn.recorder.val_losses) == 0: return None
        values = {'trn_loss':self.learn.recorder.losses[-1:][0].cpu().numpy(),
                  'val_loss':self.learn.recorder.val_losses[-1:][0]}
        for i, name in enumerate(self.learn.recorder.names[3:]):
            values[name]=self.learn.recorder.metrics[-1:][0][i]
        if values.get(self.monitor1) is None:
            warn(f'{self.__class__} conditioned on metric `{self.monitor}` which is not available. Available metrics are: {", ".join(map(str, self.learn.recorder.names[1:]))}')
        return values.get(self.monitor1)

@dataclass
class EarlyStoppingCallback(MyTrackerCallback):
    "A `TrackerCallback` that terminates training when monitored quantity stops improving."
    min_delta:int=0
    patience:int=0

    def __post_init__(self):
        super().__post_init__()
        if self.operator == np.less:  self.min_delta *= -1

    def on_train_begin(self, **kwargs:Any)->None:
        "Initialize inner arguments."
        self.wait = 0
        super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe stop training."
        current = self.get_monitor_value()
        if current is None: return
        if self.operator(current - self.min_delta, self.best):
            self.best,self.wait = current,0
        else:
            self.wait += 1
            if self.wait > self.patience:
                print(f'Epoch {epoch}: early stopping')
                return True

@dataclass
class NoOverfittingSaveModelCallback(MyTrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    every:str='improvement'
    name:str='bestmodel'
    
    def __post_init__(self):
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.every=="epoch": self.learn.save(f'{self.name}_{epoch}')
        else: #every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                self.best = current
                self.learn.save(f'{self.name}')

    def on_train_end(self, **kwargs):
        "Load the best model."
        if self.every=="improvement" and (self.learn.path/f'{self.learn.model_dir}/{self.name}.pth').is_file():
            self.learn.load(f'{self.name}')
