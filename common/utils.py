# Licensed under the MIT license.

import json
import re
import os
import random
import numpy as np
import torch
import multiprocessing
from typing import Tuple, Iterable, Dict, List
from statistics import mean
from torch.utils.data import Dataset
import jsonlines


def fix_seeds(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_jsonl(filename: str) -> List[Dict]:
    data = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            data.append(obj)
    return data


def load_dataset(data: List[Dict]):
    # 只保留 state 为 true 的 item
    return [item for item in data if item["state"]]


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = True):
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    with open(filename, mode) as fp:
        for x in data:
            fp.write((json.dumps(x) + "\n").encode("utf-8"))


# https://review-of-my-life.blogspot.com/2017/11/python-dict-shuffle.html
def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    # keys = d(keys)
    return dict(keys)
