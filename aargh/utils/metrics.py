import torch
from torchmetrics import Metric


class Perplexity(Metric):

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("perplexity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, cross_entropy):
        self.perplexity += torch.sum(torch.exp(cross_entropy))
        self.total += cross_entropy.numel()

    def compute(self):
        return self.perplexity.float() / self.total


class AverageLoss(Metric):

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss):
        self.loss += torch.sum(loss)
        self.total += loss.numel()

    def compute(self):
        return self.loss.float() / self.total
