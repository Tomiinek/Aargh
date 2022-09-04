import os
import inspect
from pytorch_lightning import LightningModule
from pytorch_lightning.core.decorators import auto_move_data
import aargh.agents as agents
from aargh.agents import *
from aargh.utils.file import try_import_module


class BaseTrainableAgent(LightningModule):

    NAME = None

    def __init__(self, params=None, *args, **kwargs):
        super().__init__()

        # This happens when initializing from a checkpoint
        if params is None:
            self.save_hyperparameters(kwargs)
            return 
        
        # This does not save the Params object, but a dictionary of contained 
        # parameters, they are then accessible in derived classes via self.hparams
        self.save_hyperparameters(params.state_dict()) 
        self.tokenizer = None

    def get_sampler(self):
        return None

    def get_batch_sampler(self):
        return None

    def get_params_dict(self):
        return self.hparams

    def set_tokenizer_reference(self, tokenizer):
        self.tokenizer = tokenizer

    @auto_move_data
    def respond(self, batch, *args, **kwarg):
        """ Take a batch produced by a task and generate and return response. """
        raise NotImplementedError

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict


class AutoAgent:

    NAME = None

    def __init__(self):
        raise EnvironmentError(
            "AutoAgent is designed to be instantiated using the `from_pretrained(pretrained_model_path)` or `from_config(params)` methods."
        )

    @staticmethod
    def from_config(config, return_instance=True):
        """
        Go through module classes and find a suitable agent.

        Arguments:
            config (Params): Configuration object, must contain attribute `agent`.
        """

        if not config.contains("agent"):
            return None

        t = AutoAgent.from_name(config.agent)
        return t(config) if return_instance else t

    @staticmethod
    def from_name(name, return_instance=False):
        """
        Go through module classes and find a suitable agent.

        Arguments:
            name (str): Name of the agent.
            return_instance (bool, optional) If True, returns an instance of the agent, otherwise class type.
        """

        modules = [agents]
        experiment = os.getenv("EXPERIMENT")

        if experiment is not None:
            modules.append(try_import_module(f"aargh.experiments.{experiment}.agents"))

        available_names = []
        for m in modules:
            if m is None:
                continue

            for c in inspect.getmembers(m, inspect.isclass): 
                valid_class = c[1].__module__.startswith(m.__name__) and getattr(c[1], 'NAME', None) is not None
                if valid_class:
                    available_names.append(c[1].NAME)
                    if c[1].NAME == name:
                        return c[1]

        raise ValueError(
            f"Unrecognized agent for AutoAgent, given: {name}.\n"
            f"Should be one of {[n for n in available_names]}."
        )