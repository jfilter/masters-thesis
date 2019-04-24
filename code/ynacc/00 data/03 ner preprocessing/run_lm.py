import news_utils

news_utils.flair.peprocess_text('~/data/ynacc_proc/replicate/lmdata/val.csv', '~/data/ynacc_proc/replicate/lmdata/val_proc.csv', header=None, names=['class', 'text'])
news_utils.flair.peprocess_text('~/data/ynacc_proc/replicate/lmdata/train.csv', '~/data/ynacc_proc/replicate/lmdata/train_proc.csv', header=None, names=['class', 'text'])
