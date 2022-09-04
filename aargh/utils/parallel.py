from joblib import Parallel, delayed
import psutil


def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def preprocess_parallel(item_list, process_chunk, n_jobs=8, chunksize=1000, prefer="processes", flatten_func=None):
    executor = Parallel(n_jobs=n_jobs, prefer=prefer)
    do = delayed(process_chunk)
    tasks = (do(chunk) for chunk in chunker(item_list, len(item_list), chunksize=chunksize))
    result = executor(tasks)
    if flatten_func is None:
        flatten_func = flatten
    return flatten_func(result)


def get_num_cpus():
    return len(psutil.Process().cpu_affinity())
