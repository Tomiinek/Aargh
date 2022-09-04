import wandb
import torch
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup as CosineWarmup
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.functional import cross_entropy, softmax
from torch.nn import ModuleDict
from einops import rearrange
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy
from aargh.agents.abstract import BaseTrainableAgent
from aargh.utils.metrics import AverageLoss #, Perplexity


class HFBaseAgent(BaseTrainableAgent):

    def __init__(self, params=None, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.model = self.instantiate_model()
        self.model.resize_token_embeddings(self.hparams.vocabulary_size)
        
        if self.hparams.get('differential_lr', False):
            self.param_groups = self.param_splitter(self.model)

        self.val_metrics = ModuleDict(sum(([
            [x + '_loss',        AverageLoss()],
            #[x + '_perplexity',  Perplexity()],
            [x + '_acc',         Accuracy()]
        ] for x in self.get_target_label_keys()), []))
        
        self.train_metrics = ModuleDict(sum(([
            [x + '_loss',        AverageLoss()],
            #[x + '_perplexity',  Perplexity()]
        ] for x in self.get_target_label_keys()), []))

    def get_target_label_keys(self):
        """ Get a list of label keys that are used for computing loss, logging etc. (e.g. belief, response, ...). """
        raise NotImplementedError()

    def instantiate_model(self):
        """ Build underlying model using something such as ModelClass.from_pretrained(model_name) and return it. """
        raise NotImplementedError()

    def param_splitter(self, model):
        """ Split omdel parameters into groups, used only when training with differential learning rate. """
        raise NotImplementedError()

    def forward(self, batch):
        """ Split omdel parameters into groups, used only when training with differential learning rate. """
        raise NotImplementedError()
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(parameters, lr=self.hparams.learning_rate)
        warmup = self.hparams.get('num_warmup_steps', None)
        if warmup is not None:
            scheduler = {
                    'scheduler': CosineWarmup(optimizer, warmup, self.hparams.num_training_steps),
                    'interval': 'step'
            }
            return [optimizer], [scheduler]
        else:
            return optimizer

    def on_train_start(self):
        
        if not self.hparams.get('differential_lr', False):
            return

        self.freeze()
        self.train()    
        self.stage = 0

    def differential_lr_update(self):
        
        def unfreeze_module(module):
            for param in module.parameters():
                param.requires_grad = True
            module.train()

        if not self.hparams.get('differential_lr', False):
            return

        next_milestone = sum([self.hparams.differential_lr[i]['steps'] for i in range(self.stage)], 0)

        if self.global_step != next_milestone:
            return

        unfreeze_module(self.param_groups[self.stage])    
        self.trainer.lr_schedulers[0]['scheduler'] = OneCycleLR(
            self.trainer.optimizers[0], 
            self.hparams.differential_lr[self.stage]['learning_rate'], 
            total_steps=self.hparams.differential_lr[self.stage]['steps'])
        self.stage += 1   

    def process_step(self, outputs, labels, metrics, validation=False, log_distribution=False, ignore_index=-100):

        shifted_logits = rearrange(outputs.logits[..., :-1, :], 'b d c -> b c d')
        predicted = shifted_logits.argmax(1)

        pointwise_losses = {}
        for k in self.get_target_label_keys():
            
            if k not in labels:
                continue

            shifted_labels = labels[k][..., 1:]

            ce = cross_entropy(shifted_logits, shifted_labels, reduction='none')
            loss = torch.masked_select(ce, shifted_labels != ignore_index)
           
            pointwise_losses[k] = loss
            metrics[k + '_loss'](loss)

            if validation:

                predicted_labels = predicted[shifted_labels != ignore_index]
                target_labels = shifted_labels[shifted_labels != ignore_index]
                metrics[k + '_acc'](predicted_labels, target_labels)

            if log_distribution:
            
                self.log_prediction(k + '_prediction', predicted, shifted_labels, self.global_step, ignore_index=ignore_index)
                self.log_next_token_ditribution(k + '_logits', shifted_logits[0][:, shifted_labels[0] != ignore_index], self.global_step)

        total_loss = torch.mean(torch.cat(list(pointwise_losses.values())))

        return total_loss

    @rank_zero_only
    def log_prediction(self, log_name, predicted, target, step, ignore_index=-100):
        if not isinstance(self.logger, WandbLogger):
            return

        mask = target != ignore_index
        mask = mask.tolist()
        target = target.tolist()
        predicted = predicted.tolist()
        
        for i in range(len(mask)):
            target[i] = [target[i][j] for j, m in enumerate(mask[i]) if m]
            predicted[i] = [predicted[i][j] for j, m in enumerate(mask[i]) if m]

        if self.tokenizer is not None:
            target_tokens = self.tokenizer.decode_batch(target)
            predicted_tokens = self.tokenizer.decode_batch(predicted)
        else:
            target_tokens = [' '.join(map(str, x)) for x in target]
            predicted_tokens = [' '.join(map(str, x)) for x in predicted]
        
        data = []
        for ti, t, pi, p in zip(target, target_tokens, predicted, predicted_tokens):
            data.append([' '.join(map(str, pi)), p, ' '.join(map(str, ti)), t])

        table = wandb.Table(data=data, columns=["Predction_ids", "Prediction_text", "Target_ids", "Target_text"])
        self.logger.experiment.log({ log_name : table }, step=step)

    @rank_zero_only
    def log_next_token_ditribution(self, log_name, predicted_logits, step, show_first=32, log_threshold=0.01):
        if not isinstance(self.logger, WandbLogger):
            return

        predicted_logits = rearrange(predicted_logits, 'c d -> d c')

        predicted_distributions = softmax(predicted_logits, dim=-1)
        probabilities, tokens = torch.sort(predicted_distributions, descending=True)
        probabilities, tokens = probabilities[...,:show_first].tolist(), tokens[...,:show_first].tolist()

        for i, (label, prob) in enumerate(zip(tokens, probabilities)):
            data = [[p, (self.tokenizer.get_id_token(l) if self.tokenizer is not None else l)] for l, p in zip(label, prob) if p > log_threshold] 
            table = wandb.Table(data=data, columns = ["probability", "token"])
            self.logger.experiment.log({ log_name + str(i) : wandb.plot.bar(table, "token", "probability", title=f"Token ({str(i).zfill(3)}) distribution")}, step=step)
