from fastai.torch_core import *
from fastai.text.transform import *
from fastai.basic_data import *
from fastai.data_block import *
from fastai.imports import torch
from fastai.text.data import TextDataBunch
from collections import Counter

DatasetType = Enum('DatasetType', 'Train Valid Test')

class SortSampler(Sampler):
    "Go through the text data by order of length."

    def __init__(self, data_source:NPArrayList, key:KeyFunc): self.data_source,self.key = data_source,key
    def __len__(self) -> int: return len(self.data_source)
    def __iter__(self):
        return iter(sorted(range_of(self.data_source), key=self.key, reverse=True))

class SortishSamplerRandom(Sampler):
    "Go through the text data by order of length with a bit of randomness."

    def __init__(self, data_source:NPArrayList, y:NPArrayList, key:KeyFunc, bs:int, num_samples):
        self.data_source,self.key,self.bs = data_source,key,bs
        
        # modified
        self.num_samples = num_samples
        # distribution of classes in the dataset 
        label_to_count = Counter(y)
        # weight for each sample
        weights = [1.0 / label_to_count[label] for label in y]
        self.weights = torch.DoubleTensor(weights)


    def __len__(self) -> int: return len(self.data_source)

    def __iter__(self):
        idxs = torch.multinomial(
            self.weights, self.num_samples, replacement=True)
#         idxs = np.random.permutation(len(self.data_source))
        sz = self.bs*50
        ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0],ck_idx[max_ck] = ck_idx[max_ck],ck_idx[0]     # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)

def pad_collate(samples:BatchSamples, pad_idx:int=1, pad_first:bool=True) -> Tuple[LongTensor, LongTensor]:
    "Function that collect samples and adds padding."
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(max_len, len(samples)).long() + pad_idx
    for i,s in enumerate(samples):
        if pad_first: res[-len(s[0]):,i] = LongTensor(s[0])
        else:         res[:len(s[0]):,i] = LongTensor(s[0])
    return res, tensor([s[1] for s in samples])

class TextClasDataBunchRandom(TextDataBunch):
    "Create a `TextDataBunch` suitable for training an RNN classifier."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs=64, pad_idx=1, pad_first=True, num_samples=2000,
               **kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification."
        datasets = [train_ds,valid_ds]
        if test_ds is not None: datasets.append(test_ds)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first)
        train_sampler = SortishSamplerRandom(datasets[0].x, y=datasets[0].y, key=lambda t: len(datasets[0].x[t]), bs=bs//2, num_samples=num_samples)
        train_dl = DataLoader(datasets[0], batch_size=bs//2, sampler=train_sampler, **kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            sampler = SortSampler(ds.x, key=lambda t: len(ds.x[t]))
            dataloaders.append(DataLoader(ds, batch_size=bs,  sampler=sampler, **kwargs))
        return cls(*dataloaders, path=path, collate_fn=collate_fn)
    
    def show_batch(self, sep=' ', ds_type:DatasetType=DatasetType.Train, rows:int=10, max_len:int=100):
        "Show `rows` texts from a batch of `ds_type`, tokens are joined with `sep`, truncated at `max_len`."
        from IPython.display import clear_output, display, HTML
        dl = self.dl(ds_type)
        b_idx = next(iter(dl.batch_sampler))
        items = [['text', 'label']]
        for i in b_idx[:rows]:
            items.append(list(dl.get_text_item(i, sep, max_len)))
        display(HTML(_text2html_table(items, [90,10])))
