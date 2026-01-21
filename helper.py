import math
from typing import Tuple, List, Any, Optional, Union, Callable, Dict  # type: ignore
import struct
import os
import numpy as np
import json
import time
from functools import wraps


#========================================================
#                    Useful functions
#========================================================

def sum_all(*args):
    result = 0
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for elem in arg:
                result += elem
        elif isinstance(arg, np.ndarray):
            result += np.sum(arg)
        elif isinstance(arg, (int, float, np.number)):
            result += arg
        else:
            raise TypeError(f"Invalid type: {type(arg)}")
    return result


#========================================================
#                       DATA HANDLING
#========================================================

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data.json")

def save(data: dict, path=MODEL_PATH):
    serializable_data = {}
    for key, value in data.items():
        serializable_data[key] = [v.tolist() for v in value]  # convert np.ndarray -> list
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable_data, f, indent=4)
    return True


def load(path=MODEL_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:
            data = json.load(f)
            # convert lists back to np.ndarray
            for key in data:
                data[key] = [np.array(v) for v in data[key]]
        except json.decoder.JSONDecodeError:
            data = {}
    return data


#========================================================
#                      Decorators
#========================================================


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper