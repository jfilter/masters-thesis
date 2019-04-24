from pathlib import Path
import re
import json
from collections import defaultdict

import news_utils.paper

best = {}
best_macro = {}
classes = ['claudience', 'clpersuasive', 'clagreement', 'cldisagreement', 'clinformative', 'clmean', 'clcontroversial', 'cltopic', 'clsentiment']

all  = defaultdict(lambda: []) 

for fn in Path('/mnt/data/group07/johannes/ynacc_proc/fasttext_baseline').glob('**/*result*.json'):
    cls_name = fn.parts[-2]
    print(fn)
    text = fn.read_text()
    all[cls_name].append(text)

news_utils.paper.print_table(all)
