import glob
import sys
import json
import pickle
import sys
import numpy
import os
from mwzeval import Evaluator


method = sys.argv[1]
fold   = sys.argv[2]
out_f  = sys.argv[3]


def format_predictions(predictions, output_file):
    formatted = {}
    for p, g in zip(predictions['item'], predictions['gt_item']):
        conv_id = g.conv_id.split('.')[0].lower()
        if conv_id not in formatted:
            formatted[conv_id] = []
        formatted[conv_id].append({
            'response' : p.response
        })
    with open(output_file, 'w+') as f:
        json.dump(formatted, f, indent=2)
    return formatted


total = {
        'bleus' : [], 
        'infos' : [], 
        'succs' : [], 
        'unis'  : [], 
        'tris'  : [], 
        'conds' : []
    }
seeds = []

for file_path in glob.iglob(method + f'/*/{fold}_preds.pkl'):
    with open(file_path, 'rb') as f:
        predictions = pickle.load(f)
    output_file_path = os.path.join(file_path.rsplit('/', 1)[0], f'{fold}_responses.json')
    
    input_data = format_predictions(predictions, output_file_path)
    
    e = Evaluator(True, True, True)
    r = e.evaluate(input_data)

    total['bleus'].append(r['bleu']['mwz22']) 
    total['infos'].append(r['success']['inform']['total']) 
    total['succs'].append(r['success']['success']['total']) 
    total['unis'].append(r['richness']['num_unigrams']) 
    total['tris'].append(r['richness']['num_trigrams']) 
    total['conds'].append(r['richness']['cond_entropy']) 
    seeds.append(file_path.rsplit('/', 2)[-2])

with open(out_f, '+w') as f:

    print(f"Results for '{method} on {fold}':", file=f)
    for k in total:
        print(f"Mean {k}: {numpy.mean(total[k])} +- {round(numpy.std(total[k]), 1)}", file=f)

    best = (0, 0)
    best_seed = -1
    for idx, (i, s) in enumerate(zip(total['infos'], total['succs'])):
        if i + s > best[0] + best[1]:
            best = (i, s)
            best_seed = seeds[idx]

    print(f"Best seed is {best_seed} with Inf: {best[0]}, Succ: {best[1]} \n", file=f)


