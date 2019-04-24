from pathlib import Path
import subprocess
import re
import json
from collections import defaultdict

import news_utils.paper

best = {}
best_macro = {}
classes = ['claudience', 'clpersuasive', 'clagreement', 'cldisagreement', 'clinformative', 'clmean', 'clcontroversial', 'cltopic', 'clsentiment']

all  = defaultdict(lambda: []) 

for fn in Path.cwd().glob('reply_output_*/*.json'):
    nums = re.findall(r'\d+', str(fn))
    dir = str(fn.parts[-2])
#    def get_labels(self):
    print(nums)
    cls_idx = int(nums[2])

    text = fn.read_text()
    all[classes[cls_idx]].append([dir, text])

#news_utils.paper.print_table(all)
res = news_utils.paper.get_best_kappa(all)
final = []
for r in res:
    print(r)
    cl = classes.index('cl' + r[1].lower())
    print(r[0], cl)
    subprocess.run("./test.sh " + str(cl) + ' "' +  r[0] + '"', shell=True, check=True)
    text = (Path(r[0])/'test_report.json').read_text()
    js = json.loads(text)
    final.append(r + [js['micro avg']['f1-score'], js['macro avg']['f1-score'], js['kappa']])

for f in final:
    f = [str(x) for x in f[1:]]
    print(' & '.join(f) + ' \\\\')
