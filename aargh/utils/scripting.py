import os
from aargh.config import Params
from aargh.utils.logging import highlight
from aargh.utils.data import tokens_from_transforms
from aargh.data.abstract import AutoTokenizer, AutoTransform, AutoTask
from aargh.agents.abstract import AutoAgent
from aargh.utils.file import try_import_module


def prepare_paths(args, use_wandb=False):
    
    # Experiment root folder in "aargh/experiments/current_experiment_name"
    # root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", args.experiment)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(root): 
        os.makedirs(root)
    paths = {'root' : root}

    outputs = os.path.join(root, 'outputs')
    if args.root_suffix is not None:
       outputs = os.path.join(outputs, args.root_suffix)

    if not os.path.exists(outputs): 
        os.makedirs(outputs)
    paths['outputs'] = outputs
    
    if use_wandb:

        wandb = os.path.join(root, 'wandb')
        if args.root_suffix is not None:
            wandb = os.path.join(wandb, args.root_suffix)

        if not os.path.exists(wandb): 
            os.makedirs(wandb)
        paths['wandb'] = wandb
    
    paths['checkpoint_path'] = None
    if args.save_checkpoints or args.checkpoint is not None:
        
        checkpoints = os.path.join(root, 'checkpoints')
        if args.root_suffix is not None:
            checkpoints = os.path.join(checkpoints, args.root_suffix)
        
        if not os.path.exists(checkpoints): 
            os.makedirs(checkpoints)
        paths['checkpoints'] = checkpoints

        if args.checkpoint is not None:
            paths['checkpoint_path'] = os.path.join(paths["checkpoints"], args.checkpoint)
    
    return paths


def parse_extra_args(items):

    def to_int(s):
        try:
            return int(s)
        except ValueError:
            return s

    def parse_var(s):
        items = s.split('=')
        key = items[0].strip() 
        if len(items) > 1:
            value = '='.join(items[1:])
        if value == "false":
            value = False
        elif value == "true":
            value = True
        else:
            value = to_int(value)
        return (key, value)

    if items is None:
        return None

    d = {}
    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value

    return d


def load_config(logger, config_file_path=None, checkpoint_path=None):

    assert config_file_path is not None or checkpoint_path is not None, (
        "You have to provide a path to the config file or to the checkpoint"
        f", given config: {config_file_path}, checkpoint: {checkpoint_path}"
    )

    if checkpoint_path is None:
        params = Params.from_file(config_file_path)
        logger.info(f"Experiment configuration loaded from: {highlight(os.path.abspath(config_file_path))}")
    
    else:
        params = Params.from_checkpoint(checkpoint_path)
        logger.info(f"Experiment configuration loaded from the checkpoint file: {highlight(checkpoint_path)}")
        
        if config_file_path is not None:
            params.load(config_file_path)
            logger.info(f"Experiment configuration updated with: {highlight(os.path.abspath(config_file_path))}")

    return params


def load_components(logger, params, task=False, transforms=False, tokenizer=False, model=False):

    tsk = AutoTask.from_config(params) if task else None
    trf = AutoTransform.from_config(params) if transforms else None
    tok = AutoTokenizer.from_config(params) if tokenizer else None
    
    if trf is not None and logger is not None:
        logger.info(f"Initialized dataset transforms: {highlight(', '.join([t.NAME for t in trf.transforms]), c='y')}")

    if tok is not None:
        if logger is not None:
            logger.info(f"Initialized the text tokenizer: {highlight(tok.NAME, c='y')} with vocabulary size {highlight(tok.get_vocabulary_size())}")

        if tsk is not None:
            new_tokens = tsk.get_new_tokens() + tokens_from_transforms(tsk.get_task_transforms(params))
            tok.add_tokens(new_tokens)
            if logger is not None:
                logger.info(f"Tokenizer extended with new tokens (from task): {new_tokens}")

        if trf is not None:
            new_tokens = tokens_from_transforms(trf)
            tok.add_tokens(new_tokens)
            if logger is not None:
                logger.info(f"Tokenizer extended with new tokens (from augmentation transforms): {new_tokens}")
    
        params.vocabulary_size = tok.get_vocabulary_size()
        params.padding_idx = tok.get_pad_token_id()
    
    mdl = AutoAgent.from_config(params) if model else None

    if mdl is not None:
        if logger is not None:
            logger.info(f"Initialized the model: {highlight(mdl.NAME, c='y')}")
        if tok is not None:
            mdl.set_tokenizer_reference(tok)

    return tsk, trf, tok, mdl


def load_dataset_wrapper(args, params, task_class, tokenizer, transforms, model):

    experiment = os.getenv("EXPERIMENT")
    if experiment is not None:
        module = try_import_module(f"aargh.experiments.{experiment}.data.dataset_wrapper")
        if module is not None:
            DatasetWrapper = getattr(module, 'CustomDatasetWrapper')
        else:
            from aargh.data.dataset_wrapper import DatasetWrapper 
    else:
        from aargh.data.dataset_wrapper import DatasetWrapper 

    return DatasetWrapper(
        args, params, task_class, tokenizer, transforms, 
        batch_sampler=model.get_batch_sampler(), sampler=model.get_sampler()
    ) 