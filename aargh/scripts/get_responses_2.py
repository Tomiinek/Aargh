import glob
import sys
import json
import sys
import numpy
import re
from functools import partial
from sacremoses import MosesTokenizer, MosesDetokenizer
from sacrebleu import corpus_bleu
from mwzeval import Evaluator
from mwzeval.normalization import normalize_slot_name


def normalize_data(input_data):

    mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')
    slot_name_re = re.compile(r'\[([\w\s\d]+)\](es|s|-s|-es|)')
    slot_name_normalizer = partial(slot_name_re.sub, lambda x: normalize_slot_name(x.group(1)))

    normalized = []
    for h, r in input_data:
        r = slot_name_normalizer(r.lower())
        r = md.detokenize(mt.tokenize(r.replace('-s', '').replace('-ly', '')))
        h = slot_name_normalizer(h.lower())
        h = md.detokenize(mt.tokenize(h.replace('-s', '').replace('-ly', '')))
        normalized.append((h, r))

    return normalized


def get_bleu(input_data):
    bleus = []
    for h, r in input_data:
        bleus.append(corpus_bleu([h], [[r]]).score)
    return bleus


def get_match(input_data):
    m = 0
    for h, r in input_data:
        if h == r:
            m += 1
    m /= len(input_data)
    return m


def format_predictions(predictions):
    
    formatted = {}
    for k in predictions:
        active_domains = []
        for turn in predictions[k]:
            #turn.pop('state')
            turn.pop('api_call') 
            #continue
            turn.pop('api_results')
            # if 'api_call' in turn:
            #     if turn['api_call'][0]:
            #         #active_domains = [x.split('_')[1] for x in turn['api_call'][0].keys()]
            #         active_domains = turn['api_call']
            #     turn.pop('api_call')	
            # turn['active_domains'] = active_domains
            turn['state'] = turn['state'][0]
            for d in turn['state']:
                for s in turn['state'][d]:
                    turn['state'][d][s] = turn['state'][d][s][0] if len(turn['state'][d][s]) > 0 else ''
            #turn.pop('state')
            turn.pop('response')
            turn['response'] = turn['response_raw'][0]
        conv_id = k.split('.')[0].lower()
        formatted[conv_id] = predictions[k]

    return formatted


def get_hint_response_pairs(predictions):
    pairs = []
    for k in predictions:
        for turn in predictions[k]:
            pairs.append((turn['hint'][0], turn['response_raw'][0]))
    return pairs


if __name__ == "__main__":

    method = sys.argv[1]
    fold   = sys.argv[2]
    identifier = sys.argv[3]
    out_f  = sys.argv[4]

    total = {
            'hmatch' : [],
            'hbleus' : [],
            'bleus' : [], 
            'infos' : [], 
            'succs' : [], 
            'unis'  : [], 
            'tris'  : [],
        }
    seeds = []

    for file_path in glob.iglob(method + f'/*/{fold}_{identifier}_outputs.json'):
        with open(file_path, 'rb') as f:
            predictions = json.load(f)

        pairs = get_hint_response_pairs(predictions)
        pairs = normalize_data(pairs)
        hbleu = get_bleu(pairs)
        match = get_match(pairs)

        input_data = format_predictions(predictions)
        
        e = Evaluator(True, True, True)
        r = e.evaluate(input_data)

        total['hmatch'].append(match) 
        total['hbleus'].extend(hbleu) 
        total['bleus'].append(r['bleu']['mwz22']) 
        total['infos'].append(r['success']['inform']['total']) 
        total['succs'].append(r['success']['success']['total']) 
        total['unis'].append(r['richness']['num_unigrams']) 
        total['tris'].append(r['richness']['num_trigrams']) 
        total['conds'].append(r['richness']['cond_entropy']) 

        seeds.append(file_path.rsplit('/', 2)[-2])

    with open(out_f, '+w') as f:

        print(f"Results for '{method} ({identifier}) on {fold}':", file=f)
        for k in total:
            print(f"Mean {k}: {numpy.mean(total[k])} +- {round(numpy.std(total[k]), 1)}", file=f)

        best = (0, 0)
        best_seed = -1
        for idx, (i, s) in enumerate(zip(total['infos'], total['succs'])):
            if i + s > best[0] + best[1]:
                best = (i, s)
                best_seed = seeds[idx]

        print(f"Best seed is {best_seed} with Inf: {best[0]}, Succ: {best[1]} \n", file=f)


