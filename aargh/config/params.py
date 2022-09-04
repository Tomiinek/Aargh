import yaml
import copy
import torch


class Params:
    def __init__(self):
        
        self.accumulation_steps = 1
        self.weight_decay = 1e-6
        self.epochs = 100
        self.gradient_clipping = 0
        
    @staticmethod
    def from_dict(config_dict):
        p = Params()
        p.load_state_dict(config_dict)
        return p    

    @staticmethod
    def from_file(config_file):
        p = Params()
        p.load(config_file)
        return p

    @staticmethod
    def from_checkpoint(checkpoint):
        ch = torch.load(checkpoint, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return Params.from_dict(ch['hyper_parameters'])

    def load_state_dict(self, d):
        if d is not None:
            for k, v in d.items():
                setattr(self, k, v)
        return self

    def state_dict(self):
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return {k: self.__dict__[k] for k in members}

    def load(self, config_file):
        if config_file is None:
            return
        with open(config_file, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
            self.load_state_dict(params)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            d = self.state_dict()
            yaml.dump(d, f)

    def contains(self, parameter):
        return hasattr(self, parameter)

    def try_get(self, key, default_value):
        if self.contains(key):
            return getattr(self, key)
        return default_value

    def get_hyperparameters(self):
        return copy.copy(self)
