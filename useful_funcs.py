import os
import pickle
import sys
import time
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    """
    Use to not print the following
    example:
    with suppress_stdout:
        print('Not printed')

    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def timeit(func):
    """
    Use as decorator
    """
    def inner(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        print('duration of "{}": {}'.format(func.__name__, time.time() - t0))
        return result

    return inner


def save_data_to_pickle(func, arguments, file_path, force_run=False):
    """

    wrap around the function to call to save the returned value. Arguments should be in tuple or list
    If path already exists, it will load the data and not run the func
    example: save_data_to_pickle(np.array, (some_list,), array.p)

    If path contains 'None' no saving and loading will be performed

    """

    if 'None' in file_path:
        result = func(*arguments)
        return result

    if os.path.exists(file_path) and not force_run:

        with open(file_path, 'rb') as f:
            print('Data already exists, loading data...')
            result = pickle.load(f)


    else:
        if not os.path.exists(os.path.dirname(file_path)):
            print(os.path.dirname(file_path))
            os.makedirs(os.path.dirname(file_path))
        result = func(*arguments)
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)

    return result
