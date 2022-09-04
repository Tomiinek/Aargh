#!/usr/bin/env python3

import os
import sys
import pytorch_lightning as pl
import pickle
import random
import json
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm
from aargh.config import Params
from aargh.data.abstract import AutoTask, AutoTokenizer
from aargh.agents import AutoAgent


def get_embeddings(batch_size, dataset, output_file_path):
    resp_embeddings = []
    ctxt_embeddings = []
    dataset_items = []

    for batch in get_batch(batch_size, dataset):
        prepared_batch = test_task.collate(batch)
        resp_encodings = model.encode_response(prepared_batch)
        ctxt_encodings = model.encode_context(prepared_batch)
        resp_embeddings.append(resp_encodings)    
        ctxt_embeddings.append(ctxt_encodings)
        dataset_items.extend(batch)

    output_resp_embeddings = {}
    for e in resp_embeddings:
        if e is None:
            continue
        for k in e:
            if k not in output_resp_embeddings:
                output_resp_embeddings[k] = []
            output_resp_embeddings[k].append(e[k])
    for k in output_resp_embeddings:
        output_resp_embeddings[k] = torch.cat(output_resp_embeddings[k])

    if not output_resp_embeddings:
        output_resp_embeddings = None

    output_ctxt_embeddings = {}
    for e in ctxt_embeddings:
        if e is None:
            continue
        for k in e:
            if k not in output_ctxt_embeddings:
                output_ctxt_embeddings[k] = []
            output_ctxt_embeddings[k].append(e[k])
    for k in output_ctxt_embeddings:
        output_ctxt_embeddings[k] = torch.cat(output_ctxt_embeddings[k])

    if not output_ctxt_embeddings:
        output_ctxt_embeddings = None

    model.save_support_cache(output_ctxt_embeddings, output_resp_embeddings, dataset_items, output_file_path)


def eval_recall(n, dataset):
    labels = []
    predictions = []

    for batch in get_batch(n, dataset, randomize=True):
        if len(batch) != n:
            break
        prepared_batch = test_task.collate(batch)
        outputs = model.respond(prepared_batch)
        predictions.append(outputs['label'].cpu().numpy())
        labels.append(np.arange(n))
   
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    def _recall(p, l, k):
        p = p[:, :k]
        acc = 0
        for i, label in enumerate(l):
            if label in p[i]: acc += 1
        acc = acc / len(l)       
        return acc
   
    top_1 = _recall(predictions, labels, 1)
    #print(f"Top-1 recall: {top_1}")
    
    return {
        "top-1": top_1, 
        "top-3": _recall(predictions, labels, 3), 
        "top-5": _recall(predictions, labels, 5), 
        "top-10": _recall(predictions, labels, 10)
    }


def eval_action_map(outputs):

    def normalize(s):
        if s == "offerbooked":
            return "book"
        elif s == "offerbook":
            return "book"
        return s

    def get_actions(item):
        return set(normalize(a.split('-')[1]) + '-' + n for a, sa in item.actions.items() for n, v in sa)

    def jaccard(a, b):
        total = len(a.union(b))
        if total == 0:
            return 1.0
        return float(len(a.intersection(b)) / total)

    full_match = 0
    no_match = 0
    num_unique = Counter()
    action_map, c = 0.0, 0
    for pi, gi in zip(outputs['item'], outputs['gt_item']):
        p_actions = get_actions(pi)
        g_actions = get_actions(gi)
        overlap = jaccard(p_actions, g_actions)
        action_map += overlap
        c += 1
        num_unique[pi.idx] += 1
        if overlap == 1.0:
            full_match += 1
        if overlap == 0.0:
            no_match += 1

    #print(f"Action accuracy: {action_map / c}")
    #print(f"Unique retrieved items: {len(num_unique)}")

    return {
        "map" : action_map / c,
        "unique" : len(num_unique),
        "full" : full_match,
        "no" : no_match
    }


def predict(batch_size, dataset):
    outputs = {
        'score' : [],
        #'label' : [],
        'item'  : [],
        'gt_item' : [],
        #'resp'  : [],
        #'ctxt'  : []
    }

    for batch in get_batch(batch_size, dataset):
        prepared_batch = test_task.collate(batch)
        output = model.respond(prepared_batch, retrieval=True)
        outputs['gt_item'].extend(batch)
        outputs['item'].extend(output['item'])
        #outputs['resp'].append(output['response_embedding'].cpu().numpy())
        #outputs['ctxt'].append(output['context_embedding'].cpu().numpy()) 
        #outputs['label'].append(output['label'].cpu().numpy())
        a = output['score']
        if type(a) is not np.ndarray:
            a = a.cpu().numpy()
        outputs['score'].append(a)

    #outputs['resp'] = np.concatenate(outputs['resp'])
    #outputs['ctxt'] = np.concatenate(outputs['ctxt'])
    #outputs['label'] = np.concatenate(outputs['label'])
    outputs['score'] = np.concatenate(outputs['score'])
    
    #for idx, i in enumerate(batch):
    #    print("-----")
    #    print(i.context)
    #    print(i.response)
    #    print(output['item'][idx].response)
    
    return outputs


def get_hints(batch_size, train, val, test):
    hints = {}
    folds = [train, val, test]
    for fold in folds:
        for batch in get_batch(batch_size, fold):
            prepared_batch = test_task.collate(batch)
            _, items = model.respond(prepared_batch, return_responses=True, return_items=True, top_k=4, retrieval=True)            
            for i, r in zip(batch, items):
                valid_responses = []
                for response in r:
                    if response.conv_id == i.conv_id:
                        continue
                    valid_responses.append(response.response)
                cid = i.conv_id.lower().split('.')[0]
                if cid not in hints:
                    hints[cid] = []
                hints[cid].append(valid_responses)
    return hints


def get_batch(size, dataset, randomize=False):
  batch = []
  sampler = range(len(dataset))
  if randomize:
    sampler = list(sampler)
    random.shuffle(sampler)
  for idx in tqdm(sampler):
      i = dataset[idx]
      batch.append(i)
      if len(batch) == size:
          yield batch
          batch = []
  if len(batch) > 0:
      yield batch


if __name__ == "__main__":

    pl.utilities.seed.seed_everything(42)

    checkpoint = sys.argv[1]
    output_directory = sys.argv[2]
    batch_size = int(sys.argv[3])

    params = Params.from_checkpoint(checkpoint)
    #if params.response_prefix is not None:
    #    print(f"Unsetting response prefix: {params.response_prefix} (for the purpose of hint generation)!")
    params.response_prefix = None
    #print(params.state_dict())
    task_class = AutoTask.from_config(params)
    tokenizer = AutoTokenizer.from_config(params)

    task_class.prepare_data()
    task_class.setup()

    dev_task = task_class(params, tokenizer=tokenizer, is_testing=False)
    test_task = task_class(params, tokenizer=tokenizer, is_testing=True)
    
    model_class = AutoAgent.from_config(params, return_instance=False)

    #ccc = torch.load(checkpoint)
    #print(ccc)

    model = model_class.load_from_checkpoint(checkpoint)
    model.cuda()
    model.eval()
    model.freeze()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    #print("Generating context embeddings of the train set!")
    get_embeddings(batch_size, dev_task.train, os.path.join(output_directory, 'train_encodings.pkl')),
    #print("Generating context embeddings of the validation set!")
    get_embeddings(batch_size, dev_task.val, os.path.join(output_directory, 'val_ctxt_embeddings.pkl')),
    get_embeddings(batch_size, test_task, os.path.join(output_directory, 'test_ctxt_embeddings.pkl')),

    #print("Evaluating top-k recall of the validation set!")
    #val_results = eval_recall(100, dev_task.val)
    #print("Evaluating top-k recall of the test set!")
    #test_results = eval_recall(100, test_task)
    # 
    #with open(os.path.join(output_directory, 'val_recall.json'), '+w') as f1: #, \ 
    #    open(os.path.join(output_directory, 'test_recall.json'), 'w+') as f2:
    #    json.dump(val_results, f1, indent=2)
    #    json.dump(test_results, f2, indent=2)

    if model.try_load_support_cache(os.path.join(output_directory, 'train_encodings.pkl')):
        #print("Predicting best training items for the validation set!")
        val_preds = predict(batch_size, dev_task.val)
        #print("Predicting best training items for the test set!")
        test_preds = predict(batch_size, test_task)
        
        with open(os.path.join(output_directory, 'val_preds.pkl'), 'wb') as f1, \
            open(os.path.join(output_directory, 'test_preds.pkl'), 'wb') as f2:
            pickle.dump(val_preds, f1)
            pickle.dump(test_preds, f2)

        #print("Generating hints for whole dataset!")
        hints = get_hints(batch_size, dev_task.train, dev_task.val, test_task)

        with open(os.path.join(output_directory, 'hints.json'), 'w') as f:
            json.dump(hints, f, indent=2)

        #print("Evaluating action accuracy for the validation set!")

        val_map = eval_action_map(val_preds)
        test_map = eval_action_map(test_preds)
        with open(os.path.join(output_directory, 'val_action_accuracy.json'), '+w') as f1, \
            open(os.path.join(output_directory, 'test_action_accuracy.json'), '+w') as f2:
            json.dump(val_map, f1, indent=2)
            json.dump(test_map, f2, indent=2)
        
        #print("Evaluating action accuracy for the test set!")
        #test_map = eval_action_map(test_preds)  


