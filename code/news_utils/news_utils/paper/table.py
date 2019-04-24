from collections import defaultdict
import json

cls = ['clpersuasive','claudience','clagreement','clinformative','clmean','clcontroversial', 'cldisagreement','cltopic', 'clsentiment']

def print_table(data):
    for k, v in sorted(data.items(), key=lambda kv: cls.index(kv[0])): 
        name = k[2:].title()
        #if 'Topic' in name:
           # name = 'Off-topic'
        best = defaultdict(lambda: 0)
        for string in v:
            dct = json.loads(string)
            for x in ['micro avg', 'macro avg']:
                for xx in ['precision', 'recall', 'f1-score']:    
                    best[x[:2] + xx[:2]] = max(best[x[:2] + xx[:2]], dct[x][xx])
        # micro pre reca and f1 are accuracy
        arr = [name, best['mif1'], best['mapr'], best['mare'], best['maf1']]
        print(' & '.join([str(x) for x in arr]), ' \\\\')

def get_best_kappa(data):
    res = []
    for k, v in data.items(): 
        name = k[2:].title()
        #if 'Topic' in name:
            # name = 'Off-topic'
        best_kappa = 0
        micro = 0
        macro = 0
        best_dir = None 
        for dir, string in v:
            dct = json.loads(string)
            if dct['kappa'] > best_kappa:
                print(dir)
                print(dct['kappa'])
                best_dir = dir
                best_kappa = dct['kappa']
                micro = dct['micro avg']['f1-score']
                macro = dct['macro avg']['f1-score']
        res.append([best_dir, name,  micro, macro, best_kappa])
    return res
        # micro pre reca and f1 are accuracy
        #arr = [name, best['mif1'], best['mapr'], best['mare'], best['maf1']]
        #print(' & '.join([str(x) for x in arr]), ' \\\\')

