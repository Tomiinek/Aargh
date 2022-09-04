import glob
import sys
import json
import numpy


method = sys.argv[1]
fold   = sys.argv[2]
out_f  = sys.argv[3]

results = []
seeds = []
counts = []
full_matches = []
no_matches = []
for file_path in glob.iglob(method + f'/*/{fold}_action_accuracy.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
        if data['map'] < 0.1:
            print(f"something bad has happend! seed {file_path.split('/')[-2]}")
        results.append(data['map'])
        seeds.append(file_path.split('/')[-2])
        counts.append(data['unique'])
        full_matches.append(data['full'])
        no_matches.append(data['no'])

with open(out_f, "+w") as f:
    print(f"Results for '{method}':", file=f)
    print(f"All: {results}", file=f)
    print(f"Best seed {seeds[numpy.argmax(results)]} with {numpy.max(results)}", file=f)
    print(f"Mean: {numpy.mean(results)}", file=f)
    print(f"Std: {numpy.std(results)}", file=f)
    print(f"Unique used: {numpy.mean(counts)}", file=f)
    print(f"Full matches: {numpy.mean(full_matches)}", file=f)
    print(f"No matches: {numpy.mean(no_matches)}", file=f)