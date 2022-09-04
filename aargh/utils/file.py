import os
import shutil
import gzip
import tarfile
import zipfile
import importlib
import urllib.request
from tqdm import tqdm


def download_file(url, directory):
    """
    Downlad file from given URL into directory.

    Arguments:
        url (string): URL of the file.
        directory (string): Destination directory.
    """
    def tqdm_hook(t):
        last_b = [0]
        def update_to(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return update_to

    filename = url.split('/')[-1]
    target = os.path.join(directory, filename)
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=target, reporthook=tqdm_hook(t), data=None)
    
    return target


def untar(path_to_file, delete=True):
    """
    Unpack or decompress the given tar or gzip file.

    Arguments:
        path_to_file (string): Full path to the tar file.
        delete (bool, default True): If True, the archive is deleted after extraction.
    """

    resolved = lambda x: os.path.realpath(os.path.abspath(x))

    def good_path(path, base):
        return resolved(os.path.join(base, path)).startswith(base)

    def good_link(info, base):
        tip = resolved(os.path.join(base, os.path.dirname(info.name)))
        return good_path(info.linkname, base=tip)

    def safe_members(members, base):
        for m in members:
            if good_path(m.name, base) and \
                (not m.issym() or good_path(m, base)) and \
                (not m.islnk() or good_link(m, base)):
                yield m

    base = os.path.dirname(path_to_file)
    path_to_file_split = path_to_file.rsplit('.', 1)
    ext = path_to_file_split[-1] # tgz, gz, txz, xz, tbz, bz2

    if ext == "gz" and path_to_file_split[0].rsplit('.', 1)[-1] != "tar":
        new_path_to_file = os.path.splitext(path_to_file)[0]
        with gzip.open(path_to_file, 'r') as gf:     
            with open(new_path_to_file, 'wb+') as tf:
                shutil.copyfileobj(gf, tf)
    else:
        ext = ext[1:] if ext.startswith('t') else ext
        with tarfile.open(path_to_file, "r:" + ext) as tf:   
            members = list(safe_members(tf, base))
            files_num = len(members)
            tf.extractall(path=base, members=tqdm(members, total=files_num, desc="Extracting"))

    if delete:
        os.remove(path_to_file)


def unzip(path_to_file, delete=True):
    """
    Unpack the given zip file.

    Arguments:
        path_to_file (string): Full path to the zip file.
        delete (bool, default True): If True, the archive is deleted after extraction.
    """

    with zipfile.ZipFile(path_to_file) as zf:   
        for f in tqdm(iterable=zf.namelist(), total=len(zf.namelist()), desc="Extracting"):
            zf.extract(member=f, path=os.path.dirname(path_to_file))

    if delete:
        os.remove(path_to_file)


def try_import_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None