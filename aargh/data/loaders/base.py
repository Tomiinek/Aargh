import os
import yaml
from tarfile import is_tarfile
from zipfile import is_zipfile
from aargh.utils.file import download_file
from aargh.utils.file import untar
from aargh.utils.file import unzip


class Loader:

    NAME = None
    VERSION = None
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".datasets"))
    META_FILE = os.path.join(BASE_DIR, "meta-data.yaml")

    def __init__(self):
        version = self.get_version()
        self.root = os.path.join(self.BASE_DIR, self.get_name() + ("" if version is None else "_" + version))

    def get_root(self):
        return self.root

    def get_resources(self):
        return self.RESOURCES

    def get_version(self):
        return self.VERSION

    def get_name(self):
        return self.NAME

    def log_info(self, message):
        from aargh.utils.logging import get_logger
        logger = get_logger(self.get_name())
        logger.info(message)

    def get_items(self, verbose=False):
        """
        (1) Download remote data if not present locally.
        (2) Read all associated files and return them as a python object.

        Arguments:
            verbose (bool): Level of verbosity. 
        """
        self.download(verbose)
        if verbose:
            self.log_info(f"Parsing dataset files in {self.BASE_DIR}")
        return self.read(verbose)

    def read(self, verbose=False):
        raise NotImplementedError()

    def download(self, verbose=False):
        """
        Download remote resources if not present locally.

        Arguments:
            verbose (bool): Level of verbosity. 
        """

        name = self.get_name()
        version = self.get_version()

        if not os.path.exists(self.BASE_DIR):
            if verbose:
                self.log_info(f"Creating directory for datasets: {self.BASE_DIR}")
            os.makedirs(self.BASE_DIR)
      
        present, present_datasets = False, {}
        
        if os.path.exists(self.META_FILE):
            with open(self.META_FILE, 'r+') as f:
                y = yaml.load(f, Loader=yaml.FullLoader)
                if y is not None:
                    present_datasets.update(y)
                    if verbose:
                        self.log_info("Acquiring information about already downloaded datasets")
                if name in present_datasets and version in present_datasets[name]:
                    present = True
                    if verbose:
                        self.log_info(f"Dataset {name} of version {version} is present: skipping download")
                else:
                    if verbose:
                        self.log_info(f"Dataset {name} of version {version} is not present")

        if not present:
            if not os.path.isdir(self.get_root()):
                self.log_info(f"Creating directory for dataset: {self.get_root()}")
                os.makedirs(self.get_root())

            for url in self.get_resources():
                target_dir = self.get_root()
                if isinstance(url, dict):
                    target_dir = os.path.join(self.get_root(), url["target_dir"])
                    url = url["file"]
                    if not os.path.isdir(target_dir):
                        self.log_info(f"Creating directory for dataset: {target_dir}")
                        os.makedirs(target_dir)
                if verbose:
                    self.log_info(f"Getting resources: {url}")
                downloaded_file = download_file(url, target_dir)
                if is_tarfile(downloaded_file):
                    if verbose:
                        self.log_info(f"Extracting compressed or tar file: {downloaded_file}")
                    untar(downloaded_file)
                elif is_zipfile(downloaded_file):
                    if verbose:
                        self.log_info(f"Extracting zip file: {downloaded_file}")
                    unzip(downloaded_file)
            
            if name in present_datasets:
                present_datasets[name].append(version)
            else:
                present_datasets[name] = [version]
            with open(self.META_FILE, 'w') as f:
                yaml.dump(present_datasets, f)