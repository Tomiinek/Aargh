#!/usr/bin/env python3

import glob
import sys
import numpy
import os
import pickle
import torch
import io
from tqdm import tqdm
from sklearn.metrics import silhouette_samples
# from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score


method = sys.argv[1]
fold   = sys.argv[2]
out_f  = sys.argv[3]

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

results = {
        'sillhouette' : [], 
        #'calinski' : [], 
        #'davies' : [],
    }
seeds = []

for file_path in glob.iglob(method + f'/*/{fold}_ctxt_embeddings.pkl'):
    seeds.append(file_path.split('/')[-2])
    with open(file_path, 'rb') as f:
        data = CPU_Unpickler(f).load()

    def normalize(s):
        s = s.split('-')[1]
        if s == "offerbooked":
            return "book"
        elif s == "offerbook":
            return "book"
        return s

    clusters = {}
    vecs = []
    total = 0
    silhouette = []

    for idx, (i, e) in enumerate(zip(data['items'], data['ctxt_encodings']['embeddings'])):
        vecs.append(e.numpy())
        item_actions = set(normalize(a) + '-' + n for a, sa in i.actions.items() for n, v in sa)
        for a in item_actions:
            if a not in clusters:
                clusters[a] = {
                    "count" : 0,
                    "pos" : []
                }
            total += 1
            clusters[a]["count"] += 1
            clusters[a]["pos"].append(idx)

    vecs = numpy.stack(vecs)
    for c in tqdm(clusters.values()):

        l = numpy.zeros(len(data['items']))
        l[c["pos"]] = 1

        sample_silhouette_values = silhouette_samples(vecs, l)
        score = sample_silhouette_values[l == 1].mean()
        silhouette.append(score * c["count"])

    results['sillhouette'].append(sum(silhouette) / total)

    # c = 0
    # x = []
    # l = []
    # last_domain = None
    # for i, e in zip(data['items'], data['ctxt_encodings']['embeddings']):
        
    #     domain = None
    #     for a in i.actions:
    #         d = a.split('-')[0]
    #         if d == "booking":
    #             if domain is None:
    #                 domain = last_domain
    #         else:             
    #             if d == "general":
    #                 if domain is None:
    #                     domain = d
    #             else:
    #                 domain = d
    #     if domain is not None:
    #         l.append(domain)
    #         x.append(e.numpy())
    #         last_domain = domain

    # l = numpy.array(l)
    # x = numpy.stack(x)
    
    # results['sillhouette'].append(silhouette_score(x, l))
    # results['calinski'].append(calinski_harabasz_score(x, l))
    # results['davies'].append(davies_bouldin_score(x, l))

with open(out_f, '+w') as f:
    print(f"Results for '{method} on {fold}':", file=f)
    for k in results:
        print(f"Mean {k}: {numpy.mean(results[k])} +- {round(numpy.std(results[k]), 3)}", file=f)
    print(f"All: {results}", file=f)
