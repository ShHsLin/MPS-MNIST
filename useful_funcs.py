import os
import pickle
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd


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
    @param func:
    @param arguments:
    @param file_path:
    @param force_run:
    @return:
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


def get_dims(iterable_, index=0):
    """
    get the dimensions of a nested lists or np.ndarrays. Also gives the type of each level
    @param iterable_:
    @param index: Delve into nested list on this index. If the index is larger than the length of one of the list. Take the
    last index in the list
    @return:
    """
    dims = []
    types = []
    while True:
        try:
            type_ = type(iterable_)
            len_ = len(iterable_)
            if len_ < index + 1:
                index_ = 0
            else:
                index_ = index
            types.append(type_)
            dims.append(len_)
            iterable_ = iterable_[index_]

        except TypeError:
            print(dims)
            print(types)
            break
    return dims, types


def transpose_wf_compressed(wf_compressed):
    """
    Transposed first dims of wf_compressed:
    (list(N),list(D),L,P,R) -> (list(D),list(n),L,P,R)
    @param wf_compressed:
    @return:
    """
    return [np.array(x) for x in zip(*wf_compressed)]


def open_csv(path):
    if not os.path.exists(path):
        f = open(path, 'x')
        f.close()
        return pd.DataFrame()
    else:
        try:
            df = pd.read_csv(path, delimiter=',', index_col=0)
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


def save_to_csv(df, calculations_dict, path):
    """
    Appends data calculated from functions provided in calculations dict.
    It calculates and saves only one row
    @param df: dataframe to add data to
    @param calculations_dict: dictionary structured as follows:
    {name_of_column: (func, (args))}

    the first entry in the dict must be a column that does not calculate anything -> its like the index and the entry in the dict must look like:
    {name_of_column: value}

    all entries after that have the given structure above.
    args must be given in the proper order as we just unpack --> TODO look into kwargs

    If the data already exists in the df/csv, it wont run it again.


    @param path: path to save data/df to
    @return:
    """
    new_idx = len(df)

    empty = len(calculations_dict.keys()) * [True]
    column_names = list(calculations_dict.keys())
    index_name = column_names[0]
    index_value = calculations_dict[index_name]
    if not df.empty:
        if calculations_dict[index_name] in df[index_name].astype(type(index_value)).values:
            empty[0] = False
        for column_idx, key in enumerate(column_names[1:]):
            if not df[df[index_name] == index_value][key].empty:
                empty[column_idx + 1] = False
    if empty[0]:
        df.loc[new_idx, index_name] = index_value
    else:
        new_idx = df[df[index_name] == index_value].index[0]
    for column_idx, key in enumerate(column_names[1:]):
        if empty[column_idx + 1]:
            args = calculations_dict[key][1]
            value = calculations_dict[key][0](*args)
            df.loc[new_idx, key] = value
        else:
            print(f'Already calculated {key} for {index_name}: {index_value}. Skipping...')
    df.to_csv(path)
