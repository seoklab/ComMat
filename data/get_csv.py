#!/usr/bin/env python3

import pandas as pd
import json
from collections import defaultdict

a = json.load(open('cluster_train.json'))

info_dict = defaultdict(list)

for k, v in a.items():
    cluster = v
    for k1, v1 in v.items():
        for pdbname in v1:
            pdb = pdbname.split('_')[0]
            if pdb in info_dict['pdbname']: continue
            else:
                info_dict['pdbname'].append(pdb)
                info_dict['cluster'].append(k)
                info_dict['subcluster'].append(k1)

print(info_dict)
print(len(info_dict['pdbname']))
print(len(info_dict['cluster']))
print(len(info_dict['subcluster']))
df = pd.DataFrame.from_dict(data=info_dict, orient='columns')
df.to_csv('train.csv', index=False)
