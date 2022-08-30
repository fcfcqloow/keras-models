import os, pickle
from typing import Callable

def _save_pickle(path, result):
    print('save pickle', path)
    with open(path, 'wb') as f:
        pickle.dump(result, f)
def _load_pickle(path):
    print('load pickle', path)
    with open(path, 'rb') as f:
        return pickle.load(f)

def loadCacheFunction(path, fn: Callable):
    if os.path.exists(path):
        return _load_pickle(path)
    result = fn()
    _save_pickle(path, result)
    return result

def loadCacheVariable(path):
    def save(var):
        _save_pickle(path, var)
        return var
    if os.path.exists(path):
        return _load_pickle(path), save
    return None, save
