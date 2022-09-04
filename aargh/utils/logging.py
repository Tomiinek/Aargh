import importlib
import sys
import logging as python_logging
from colorlog import ColoredFormatter 

if importlib.util.find_spec('ipywidgets') is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.utilities import rank_zero_only


python_logging.addLevelName(python_logging.INFO, 'info')
python_logging.addLevelName(python_logging.ERROR, 'error')
python_logging.addLevelName(python_logging.WARNING, 'warning')
python_logging.addLevelName(python_logging.CRITICAL, 'critical')
python_logging.addLevelName(python_logging.DEBUG, 'debug')


def highlight(string, c="b"):
    colors = {
        'p' : '\033[1;35m',
        'c' : '\033[1;36m',
        'b' : '\033[1;34m',
        'g' : '\033[1;32m',
        'y' : '\033[1;33m',
        'r' : '\033[1;31m',
        'b' : '\033[1m'
    }    
    return colors[c] + str(string) + '\033[0m' 


def get_logger(name):
    logger = python_logging.getLogger(name)
    if not logger.hasHandlers():
        logger.addHandler(python_logging.StreamHandler())
    # for adding time:  %(cyan)s%(asctime)s%(reset)s
    formatter = ColoredFormatter(
        '%(bold)s%(log_color)s%(levelname)s%(reset)s:  %(message)s', 
        datefmt='%H:%M:%S',
        log_colors={
            'debug':    'cyan',
            'info':     'green',
            'warning':  'yellow',
            'error':    'red',
            'critical': 'red,bg_white'
        }
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)     
    logger.setLevel(python_logging.ERROR)
    set_logger_level(logger)
    return logger


@rank_zero_only
def set_logger_level(logger):
    logger.setLevel(python_logging.INFO)


class CustomProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()
    
    @staticmethod
    def convert_inf(x):
        return None if x == float('inf') else x

    def init_validation_tqdm(self) -> tqdm:
        bar = tqdm(
            desc='Validating',
            position=(2 * self.process_position + 1),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout
        )
        return bar
    
    def on_epoch_start(self, trainer, pl_module):
        super(ProgressBar, self).on_epoch_start(trainer, pl_module)
        acc = self.trainer.accumulate_grad_batches
        total_batches = (self.total_train_batches + acc // 2) // acc
        if not self.main_progress_bar.disable:
            self.main_progress_bar.reset(total_batches)
        self.main_progress_bar.set_description(f'Epoch {trainer.current_epoch}')
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super(ProgressBar, self).on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.train_batch_idx, self.total_train_batches, self.refresh_rate * self.trainer.accumulate_grad_batches):
            self._update_bar(self.main_progress_bar)
            train_dict = { k:v for k,v in trainer.progress_bar_dict.items() if not k.startswith("val") }
            self.main_progress_bar.set_postfix(train_dict)
    
    def on_validation_start(self, trainer, pl_module):
        super(ProgressBar, self).on_validation_start(trainer, pl_module)
        if not trainer.running_sanity_check:
            self.val_progress_bar = self.init_validation_tqdm()
            if not self.val_progress_bar.disable:
                self.val_progress_bar.reset(self.total_val_batches)
            
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super(ProgressBar, self).on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.val_batch_idx, self.total_val_batches):  
            self._update_bar(self.val_progress_bar)     
            
    def _should_update(self, current, total, rate=None):
        rate = self.refresh_rate if rate is None else rate
        return self.is_enabled and (current % rate == 0 or current == total)
    
    def on_validation_end(self, trainer, pl_module):
        super(ProgressBar, self).on_validation_end(trainer, pl_module)
        val_dict = { k:v for k,v in trainer.progress_bar_dict.items() if k.startswith("val") }
        self.val_progress_bar.set_postfix(val_dict)
        self.val_progress_bar.close()
        
    def _update_bar(self, bar):
        """ Updates the bar by the refresh rate without overshooting. """
        if bar.total is not None:
            delta = min(self.refresh_rate, bar.total - bar.n)
        else:
            # infinite / unknown size
            delta = self.refresh_rate
        if delta > 0:
            bar.update(delta)
