#!/usr/bin/env python3

import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import deque
from pytorch_lightning import seed_everything
from aargh.utils.logging import get_logger
from aargh.utils.scripting import parse_extra_args, load_config, load_components, load_dataset_wrapper


def generate_corpus(task, model, dataset, ground_truth_keys, verbose=True):

    all_predictions = {}
    current_conv = None
    conv_items, conv_predicted = [], []
    predicted = {}

    for item in tqdm(dataset.items, disable=(not args.verbose)):
        
        if current_conv is None or item.conv_id != current_conv:
            
            if current_conv is not None: 
                all_predictions[current_conv] = conv_predicted
            
            conv_items = []
            conv_predicted = []
            predicted = {} 
            current_conv = item.conv_id

        predicted = task.get_responses(model, [item], ground_truth_keys, **predicted)
        conv_items.append(item)
        conv_predicted.append(predicted)

    all_predictions[current_conv] = conv_predicted
    json.dump(all_predictions, f, indent=2)


def generate_interactive(task, model, output_file=None):

    context = deque(maxlen=params.try_get('context_length', 100))
    responses = {}

    while True:
        user = input("user: ") 
        if user == "exit":
            break

        if output_file is not None:
            print(f"user: {user}", file=output_file)

        context.append({'speaker': 'user', 'utterance': user})

        item = task.DatasetItem()
        item.context = context
        responses = task.get_responses(model, [item], {'context'}, **responses)
        
        context.append({'speaker': 'system', 'utterance': responses['response'][0]})

        if args.debug or output_file is not None:
            for k, v in responses.items():
                
                if output_file is not None:
                    print(f"{k}: {v[0]}", file=output_file)
                
                if args.debug:
                    if k == 'response':
                        continue
                    print(f"{k}: {v[0]}")

        print("system: " + responses['response'][0])


def parse_arguments():
 
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Checkpoint of the pretrained agent.")
    parser.add_argument("-cfg", "--config", type=str, default=None, help="Path to config file, used if not checkpoint provided.")
    parser.add_argument("-d", "--debug", dest='debug', default=False, action='store_true', help="Set debug mode.")
    parser.add_argument("-q", "--quiet", dest='verbose', default=True, action='store_false', help="Set quiet mode.")
    parser.add_argument("-i", "--interactive", dest='interactive', default=False, action='store_true', help="Set interactive mode.")
    parser.add_argument("-o", "--output-file", type=str, default=None, help="File for logging the conversations.")
    parser.add_argument("-f", "--fold", type=str, default=None, help="Task fold to be evaluated if not in interactive mode, one of `val`, `test`")
    parser.add_argument("-w", "--num-workers", type=int, default=1, help="Number threads used.")
    parser.add_argument("-g", "--use-gpu", dest="use_gpu", action="store_true", default=False, help="If set, GPU is used for the inference.")   
    parser.add_argument("-t", "--ground-truths", default=['context', 'api_call'], nargs='+', type=str, help="List of dataset item members which should be kept as ground truth.")
    parser.add_argument("--set", metavar="KEY=VALUE", nargs='+', help="Custom configuration options.")
    parser.set_defaults(full=False)
    args = parser.parse_args()

    extra_args = parse_extra_args(args.set) 

    return args, extra_args


if __name__ == '__main__':
    
    seed_everything(42)

    # Parse arguments
    logger = get_logger(__name__)
    args, extra_args = parse_arguments()

    # Load config file
    params = load_config(logger, args.config, args.checkpoint)

    # If some hyper parameters were provided directly via a command line option, use it
    if extra_args is not None:
        params.load_state_dict(extra_args)
        logger.info(f"Experiment configuration updated with: {extra_args}")

    params.load_state_dict({"experiment" : "retrieval"})

    # Load task, data transforms, tokenizer, update tokenizer with transforms, model ... all from config 
    task_class, transforms, tokenizer, model = load_components(
        logger if args.verbose else None, params, task=True, transforms=False, tokenizer=True, model=True
    )

    task = task_class(params, is_testing=True)
    task.tokenizer = tokenizer

    if args.checkpoint:
        additional_params = params.state_dict() if args.config is not None or args.set is not None else {}
        model = model.load_from_checkpoint(args.checkpoint, **additional_params)
    
    # Move model to the correct device and prepare it for inference
    torch.set_num_threads(args.num_workers)
    model.to('cuda' if args.use_gpu else 'cpu')
    model.eval()
    model.freeze()

    # Prepare the output file for wirting if it exists
    if args.output_file is None:
        f = None
    else:
        f = open(args.output_file, "w+")

    # Run an interactive session between the system and the user
    if args.interactive:   
        generate_interactive(task, model, output_file=f)

    # Generate responses based on ground truth context from a corpus
    else:      
        data = load_dataset_wrapper(args, params, task_class, tokenizer, transforms, model)
        data.prepare_data()

        assert args.fold in ['val', 'test'], logger.error(f"The task fold must be either `val` or `test`, given: {args.fold}")

        data.setup(args.fold)
        dataset = data.dataset.val if args.fold == 'val' else data.dataset
        sorted(dataset.items, key=lambda x: (x.conv_id, x.idx))

        generate_corpus(task, model, dataset, set(args.ground_truths), verbose=args.verbose)

    if f is not None:
        f.close()
