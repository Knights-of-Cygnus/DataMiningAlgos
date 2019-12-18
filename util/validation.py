import numpy as np
import pandas as pd
from operator import add, is_not
from functools import reduce, partial
from typing import Iterable

def make_slices(lengths: Iterable[int]):
    current_index = 0
    for length in lengths:
        yield slice(current_index, current_index + length)
        current_index += length

def fold_cross_validation(k: int, data: pd.DataFrame):
    shuffled = data.sample(frac=1)
    chunk_length, remain = divmod(len(data), k)
    lengths = [chunk_length for _ in range(1, k)] + [chunk_length + remain]
    slices = list(make_slices(lengths))
    chunks = [shuffled[s] for s in slices]
    for chunk in chunks:
        yield reduce(pd.DataFrame.append, filter(partial(is_not, chunk), chunks)), chunk
