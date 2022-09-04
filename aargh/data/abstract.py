import os
import inspect
from torchvision.transforms import Compose 
import aargh.data as data
from aargh.utils.file import try_import_module
from aargh.data.loaders import *
from aargh.data.tasks import *
from aargh.data.tokenizers import *
from aargh.data.transforms import *


class AutoTransform:

    def __init__(self):
        raise EnvironmentError("AutoTransform is designed to be instantiated using the `from_config(params)()` method.")

    @staticmethod
    def from_config(config):
        """
        Go through module classes and find a suitable tokenizer.

        Arguments:
            config (Params): Configuration object.
        """

        if not config.contains("transforms"):
            return None

        available_transforms = {}

        modules = [data.transforms]
        experiment = os.getenv("EXPERIMENT")

        if experiment is not None:
            modules.append(try_import_module(f"aargh.experiments.{experiment}.data.transforms"))

        for m in modules:
            if m is None:
                continue
            
            for c in inspect.getmembers(m, inspect.isclass): 
                if  c[1].__module__.startswith(m.__name__) and c[1].NAME is not None:
                    available_transforms[c[1].NAME] = c[1]

        transforms = []

        for t in config.transforms:
            if t["name"] not in available_transforms:
                raise ValueError(f'Unrecognized transform for AutoTransform, given: {t["name"]}.')
            
            kwargs = {k : w for k, w in t.items() if k != "name"}
            new_transform = available_transforms[t["name"]](**kwargs)
            transforms.append(new_transform)    

        return Compose(transforms)


class AutoTokenizer:

    def __init__(self):
        raise EnvironmentError("AutoTokenizer is designed to be instantiated using the `from_config(params)()` method.")

    @staticmethod
    def from_config(config, return_instance=True):
        """
        Go through module classes and find a suitable tokenizer. Return tokenizer instance.

        Arguments:
            config (Params): Configuration object, must contain attribute `tokenizer`.
            return_instance (bool, optional) If True, returns an instance of the tokenizer, otherwise class type.
        """

        if not config.contains("tokenizer"):
            return None

        tokenizer_class = AutoTokenizer.from_name(config.tokenizer)
        return tokenizer_class if not return_instance else tokenizer_class(config.try_get("sequence_tokens", None))

    @staticmethod
    def from_name(name, return_instance=False):
        """
        Go through module classes and find a suitable tokenizer. Return tokenizer instance.

        Arguments:
            name (str): Name of the tokenizer.
        """

        modules = [data.tokenizers]
        experiment = os.getenv("EXPERIMENT")

        if experiment is not None:
            modules.append(try_import_module(f"aargh.experiments.{experiment}.data.tokenizers"))

        available_names = []
        for m in modules:
            if m is None:
                continue

            for c in inspect.getmembers(m, inspect.isclass): 
                valid_class = c[1].__module__.startswith(m.__name__) and c[1].NAME is not None
                if valid_class:
                    available_names.append(c[1].NAME)
                    if c[1].NAME == name:
                        return c[1]() if return_instance else c[1]

        raise ValueError(
            f"Unrecognized tokenizer for AutoTokenizer, given: {name}.\n"
            f"Should be one of {[n for n in available_names]}."
        )


class AutoTask:

    def __init__(self):
        raise EnvironmentError("AutoTask is designed to be instantiated using the `from_config(params)()` method.")

    @staticmethod
    def from_config(config, return_instance=False):
        """
        Go through module classes and find a suitable task. Return class type.

        Arguments:
            config (Params): Configuration object, must contain attribute `task`.
            return_instance (bool, optional) If True, returns an instance of the task, otherwise class type.
        """

        if not config.contains("task"):
            return None

        t = AutoTask.from_name(config.task)
        return t(config, is_testing=True) if return_instance else t

    @staticmethod
    def from_name(name):
        """
        Go through module classes and find a suitable task. Return class type.

        Arguments:
            name (str): Name of the loader.
            return_instance (bool, optional) If True, returns an instance of the task, otherwise class type.
        """
        modules = [data.tasks]
        experiment = os.getenv("EXPERIMENT")

        if experiment is not None:
            modules.append(try_import_module(f"aargh.experiments.{experiment}.data.tasks"))

        available_names = []
        for m in modules:
            if m is None:
                continue

            for c in inspect.getmembers(m, inspect.isclass): 
                valid_class = c[1].__module__.startswith(m.__name__) and c[1].NAME is not None
                if valid_class:
                    available_names.append(c[1].NAME)
                    if c[1].NAME == name:
                        return c[1]

        raise ValueError(
            f"Unrecognized task for AutoTask, given: {name}.\n"
            f"Should be one of {[n for n in available_names]}."
        )


class AutoLoader:

    def __init__(self):
        raise EnvironmentError("AutoLoader is designed to be instantiated using the `from_config(params)()` method.")

    @staticmethod
    def from_name(name, version=None, return_instance=True):
        """
        Go through module classes and find a suitable loader. Return loader instance.

        Arguments:
            name (str): Name of the loader.
            version (str): Version of the loader or None if not specified.
            return_instance (bool, optional) If True, returns an instance of the loader, otherwise class type.
        """
        modules = [data.loaders]
        experiment = os.getenv("EXPERIMENT")

        if experiment is not None:
            modules.append(try_import_module(f"aargh.experiments.{experiment}.data.loaders"))

        available_names = []
        for m in modules:
            if m is None:
                continue

            for c in inspect.getmembers(m, inspect.isclass): 
                valid_class = c[1].__module__.startswith(m.__name__) and c[1].NAME is not None and c[1].VERSION is not None
                if valid_class: 
                    available_names.append((c[1].NAME, c[1].VERSION))
                    if c[1].NAME == name and c[1].VERSION == version:
                        return c[1]() if return_instance else c[1]

        raise ValueError(
            f"Unrecognized task for AutoLoader, given name: {name}, given version: {version}.\n"
            f"Should be one of {[n for n in available_names]}."
        )