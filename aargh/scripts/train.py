#!/usr/bin/env python3
import os
import random
import string
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from aargh.utils.logging import CustomProgressBar, get_logger, highlight
from aargh.utils.scripting import parse_extra_args, load_config, prepare_paths, load_components, load_dataset_wrapper


def parse_arguments():
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="default", type=str, help="Name of the run.")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=False, help="Use W&B for logging.")
    parser.add_argument("--wandb-checkpoints", dest="wandb_checkpoints", action="store_true", default=False, help="Set this flag to save checkpoint to W&B.")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=False,  help="Make training deterministic again.")
    parser.add_argument("--single-pass", dest="single_pass", action="store_true", default=False,  help="If set, one batch of train, val, test are run.")
    parser.add_argument("--overfit", dest="overfit", type=int, default=0,  help="Number of batches to overfit.")
    parser.add_argument('--gpus', type=int, default=None, help="Number of GPUs to use.")
    parser.add_argument('--num-workers', type=int, default=4, help="Number threads per GPU used for data loading.")
    parser.add_argument('--nodes', type=int, default=1, help="Number of nodes to use.")
    parser.add_argument('--accelerator', type=str, default="ddp", help="The accelerator backend to use, one of 'ddp' (preffered), 'dp', 'ddp2', or 'ddp_cpu'.")
    parser.add_argument('--num-processes', type=int, default=1, help="Num. of processes, used only if accelerator == 'ddp_cpu'.")
    parser.add_argument('--precision', type=int, default=32, help="Floating point precision should be 16 or 32.")
    parser.add_argument('--cuda-benchmark', dest="cudnn_benchmark", action="store_true", default=False, help="Set if inputs are of fixed size (enables optimization).")
    parser.add_argument("--root-suffix", type=str, default=None, help="More detailed experiment root if needed.")
    parser.add_argument("--no-checkpoints", dest="save_checkpoints", action="store_false", default=True, help="If set, checkpoints are not saved.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to restore tranining from.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    parser.add_argument('--evaluate-steps', type=(lambda x: (int(x) if float(x) > 1 else float(x))), default=1.0, help="Every this steps (backward passes) run evaluation.")
    parser.add_argument('--log-steps', type=int, default=50, help="Every this steps (backward passes) log training progress.")
    parser.add_argument('--limit-evaluation', type=int, default=None, help="Number of batches (without gradient accumulation) to be used for validation.")
    parser.add_argument('--multiple_trainloader_mode', type=str, default="max_size_cycle", help="Mode of iterating through multiple datasets, if provided.")
    parser.add_argument("--set", metavar="KEY=VALUE", nargs='+', help="Custom configuration options.")
    args = parser.parse_args()

    extra_args = parse_extra_args(args.set) 

    return args, extra_args


def print_paths(paths):
    if paths.get('root', None):
        console_logger.info(f"Initialized the root directory: {highlight(paths['root'])}")
    if paths.get('outputs', None):
        console_logger.info(f"Initialized the output directory for predictions: {highlight(paths['outputs'])}")
    if paths.get('wandb', None):
        console_logger.info(f"Initialized the W&B logging directory: {highlight(paths['wandb'])}")
    if paths.get('checkpoints', None):
        console_logger.info(f"Initialized the checkpoint save directory: {highlight(paths['checkpoints'])}")
    if paths.get('checkpoint_path', None):
        console_logger.info(f"Training will continue with the checkpoint: {highlight(paths['checkpoint_path'], c='y')}")


if __name__ == '__main__':

    console_logger = get_logger("lightning")

    # Parse arguments and initialize paths
    args, extra_args = parse_arguments()
    paths = prepare_paths(args, use_wandb=args.wandb)

    # Load config file
    params = load_config(console_logger, args.config, paths["checkpoint_path"] if args.checkpoint is not None else None)

    # If some hyper parameters were provided directly via a command line option, use it
    if extra_args is not None:
        params.load_state_dict(extra_args)
        console_logger.info(f"Experiment configuration updated with: {extra_args}")
    params.load_state_dict({"experiment" : "retrieval"})

    # Set the seed of everything, i.e., torch, numpy, ... 
    if args.deterministic:
        seed_everything(params.try_get("seed", 42))

    # Start W&B logger and change the checkpoint directory to the W&B experiment directory
    if args.wandb:
        version_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        logger = WandbLogger(name=args.name, project='aargh', save_dir=paths["root"], 
                             log_model=args.wandb_checkpoints, version=version_id)
        _ = logger.experiment # this is done for an earlier initialization of the experiment
        console_logger.info(f"Initialized W&B experiment with id: {highlight(version_id, c='y')}, named: {highlight(args.name, c='y')}")
        if args.save_checkpoints: 
            paths["checkpoints"] = os.path.join(paths["checkpoints"], logger.version)
    else:
        logger = None

    # Pring used path to console
    print_paths(paths)

    # Load task, data transforms, tokenizer, update tokenizer with transforms, model ... all from config 
    task_class, transforms, tokenizer, model = load_components(
        console_logger, params, task=True, transforms=True, tokenizer=True, model=True
    )

    # Prepare the dataset (instantiate task_class, download, ...), setup workers etc., connect task and tokenizer / transforms 
    data = load_dataset_wrapper(args, params, task_class, tokenizer, transforms, model)

    # Setup progress bar and checkpoint callbacks
    callbacks = [CustomProgressBar()]
    if args.wandb:
        callbacks.append(LearningRateMonitor())
    if args.save_checkpoints:
        callbacks.append(ModelCheckpoint(dirpath=paths["checkpoints"], save_last=True, monitor='val_total_loss', filename='{epoch}-{step}'))

    class SeedCallback(Callback):
        def on_fit_start(self, trainer, pl_module):
            seed_everything(params.try_get("seed", 42))

    callbacks.append(SeedCallback())

    console_logger.info(f"Train setup:")
    console_logger.info(f"   Number of GPUs:        {highlight(args.gpus, c='y')}")
    console_logger.info(f"   Threads per device:    {highlight(args.num_workers, c='y')}")
    console_logger.info(f"   Number of nodes:       {highlight(args.nodes, c='y')}")
    console_logger.info(f"   Accelerator:           {highlight('default' if not args.accelerator else args.accelerator, c='y')}")
    ebs = params.batch_size * params.try_get('accumulation_steps', 1) * \
         (args.gpus if args.accelerator in ['ddp', 'horovod'] else 1)
    console_logger.info(f"   Effective batch size:  {highlight(ebs, c='y')}")

    if args.accelerator not in ['dp', 'ddp', 'ddp2', 'horovod'] and args.gpus:
        console_logger.warning(f"Running with accelerator {highlight(args.accelerator, c='r')} while {args.gpus} GPU available!")

    train_plugins = []
    if args.accelerator == "ddp":
        train_plugins.append(DDPPlugin(find_unused_parameters=True))

    trainer = Trainer(accelerator=args.accelerator,
                      accumulate_grad_batches=params.accumulation_steps, 
                      benchmark=args.cudnn_benchmark,
                      callbacks=callbacks,
                      checkpoint_callback=args.save_checkpoints,
                      default_root_dir=paths["root"],
                      deterministic=args.deterministic,
                      fast_dev_run=args.single_pass,
                      gpus=args.gpus, 
                      gradient_clip_val=params.gradient_clipping,
                      limit_val_batches=(1.0 if args.limit_evaluation is None else args.limit_evaluation),
                      log_every_n_steps=args.log_steps * params.accumulation_steps,
                      logger=logger,
                      max_epochs=params.epochs,
                      multiple_trainloader_mode=args.multiple_trainloader_mode,
                      num_nodes=args.nodes,
                      num_processes=args.num_processes,
                      overfit_batches=args.overfit,
                      precision=args.precision,
                      # profiler="simple",
                      resume_from_checkpoint=paths['checkpoint_path'],
                      plugins=train_plugins,
                      # track_grad_norm=2,
                      val_check_interval=args.evaluate_steps * (params.accumulation_steps if args.evaluate_steps > 1 else 1.0),
                      replace_sampler_ddp=(not data.has_custom_sampler())
                    )

    trainer.fit(model, datamodule=data)

    # data.setup('test')
    # test_data = data.test_dataloader()
    # if test_data is not None and len(test_data) > 0:
    #     trainer.test(model, test_data)