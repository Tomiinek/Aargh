from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from aargh.utils.logging import get_logger, highlight


class DatasetWrapper(LightningDataModule):

    def __init__(self, args, params, task, tokenizer=None, augment_transforms=None, batch_sampler=None, sampler=None):
        """
        Dataset wrapper providing DataLoaders and Samplers (ditributed versions are employed automatically).
         - To use a custom sampler, Trainer must have set replace_sampler_ddp = False.
         - To reload DataLoaders every epoch, set reload_dataloaders_every_epoch = True in the Trainer.

        Arguments:
            args (dictionary): Dictionary technical hyperparameters (number of workers, GPUs, ...).
            params (dictionary): Dictionary with necessary hyperparameters.
            task (BaseTask): Task to be wrapped to create a dataset.
            tokenizer (Tokenizer, optional): HF Tokenizer to be used for data tokenization.
            augment_transforms (collable, optional): Transform to be applied on items of the underlying torch Dataset.
        """
        super().__init__()
        self.args = args
        self.params = params
        self.task_class = task
        self.tokenizer = tokenizer
        self.augment_transforms = augment_transforms
        self.logger = get_logger("lightning")
        self.batch_sampler = batch_sampler
        self.sampler = sampler

    def prepare_data(self):
        """ Dataset setup called only once on the main node, use for downloading datasets. """
        self.task_class.prepare_data()
        dataset_str = ', '.join([highlight(n) + ('' if v is None else f'({v})') for n, v in self.task_class.DATASETS])
        self.logger.info(f"Prepared task: {highlight(self.task_class.NAME, c='y')} with datasets: {dataset_str}")

    def setup(self, stage=None):
        """
        Dataset setup called on every GPU.

        Arguments:
            stage (string): Either 'fit' or 'test'. If it is 'fit', it is called at the beginning of training fit. 
        """     
        self.task_class.setup()
        if stage == 'fit' or stage is None:
            self.dataset = self.task_class(self.params, is_testing=False, tokenizer=self.tokenizer, augment_transforms=self.augment_transforms)
            self.logger.info("Initialized train dataset (" + highlight("train: ") + highlight(len(self.dataset.train), c='y') + \
                             ", " + highlight("validation: ") + highlight(len(self.dataset.val), c='y') + " examples)")
        if stage == 'test' or stage is None:
            self.dataset = self.task_class(self.params, is_testing=True, tokenizer=self.tokenizer)
            self.logger.info(f"Initialized test dataset (with {highlight(len(self.dataset), c='y')} examples)")  

    def train_dataloader(self):
        if self.batch_sampler is not None:
            batch_sampler = self.batch_sampler(self.dataset.train, self.params)
            return {
               'custom'  : DataLoader(self.dataset.train, collate_fn=self.dataset.collate, num_workers=self.args.num_workers,
                                      pin_memory=(self.args.gpus), batch_sampler=batch_sampler),
               'default' : DataLoader(self.dataset.train, batch_size=8, collate_fn=self.dataset.collate, 
                                      num_workers=self.args.num_workers, pin_memory=(self.args.gpus))
            }
        else:
            if self.sampler is not None:
                sampler = self.sampler(self.dataset.train, self.params)
            else:
                sampler = None
            return DataLoader(self.dataset.train, batch_size=self.params.batch_size, collate_fn=self.dataset.collate, 
                              num_workers=self.args.num_workers, pin_memory=(self.args.gpus), sampler=sampler),

    def val_dataloader(self):
        if self.batch_sampler is not None:
            batch_sampler = self.batch_sampler(self.dataset.val, self.params, shuffle=False)
            return [
               DataLoader(self.dataset.val, collate_fn=self.dataset.collate, num_workers=self.args.num_workers,
                                      pin_memory=(self.args.gpus), batch_sampler=batch_sampler),
               DataLoader(self.dataset.val, batch_size=8, collate_fn=self.dataset.collate, 
                                      num_workers=self.args.num_workers, pin_memory=(self.args.gpus))
            ]
        else:
            if self.sampler is not None:
                sampler = self.sampler(self.dataset.val, self.params, shuffle=False)
            else:
                sampler = None
            return DataLoader(self.dataset.val, batch_size=self.params.batch_size, collate_fn=self.dataset.collate,
                        num_workers=self.args.num_workers, pin_memory=(self.args.gpus), sampler=sampler)

    def test_dataloader(self):
        if self.dataset is not None:
            if self.batch_sampler is not None:
                batch_sampler = self.batch_sampler(self.dataset.test, self.params, shuffle=False)
                return DataLoader(self.dataset.test, collate_fn=self.dataset.collate, num_workers=self.args.num_workers, 
                                  pin_memory=(self.args.gpus), batch_sampler=batch_sampler)
            else:
                if self.sampler is not None:
                    sampler = self.sampler(self.dataset.test, self.params, shuffle=False)
                else :
                    sampler = None
                return DataLoader(self.dataset, batch_size=self.params.batch_size, collate_fn=self.dataset.collate,
                            num_workers=self.args.num_workers, pin_memory=(self.args.gpus), shuffle=False, drop_last=False, sampler=sampler)
        else:
            return None

    def has_custom_sampler(self):
        return self.batch_sampler is not None or self.sampler is not None
