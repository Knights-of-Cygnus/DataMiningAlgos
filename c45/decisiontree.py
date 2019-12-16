import math
from typing import Union, List, Iterable, Optional, Callable, TypeVar
import pandas as pd
from operator import itemgetter
import itertools
from collections import Counter

Numeric = Union[int, float]
T = TypeVar('T')


class DTree:
    def __init__(self, label: T, threshold: Optional[Numeric] = None, children: Optional[list] = None):
        self.label = label
        self.threshold = threshold
        self.isLeaf = threshold is None
        if children:
            l, r = children
            if l.is_leaf() and r.is_leaf() and l.label == r.label:
                self.children = []
                self.label = l.label
                self.threshold = None
            else:
                self.children = children
        else:
            self.children = []

    def is_leaf(self):
        return not self.children

    def to_string(self) -> str:
        return f'Tree({self.label}, {self.threshold}, [{"".join(map(DTree.to_string, self.children))}])'
    
    def format_tree(self, level: int = 0):
        s = '  ' * level
        if self.is_leaf():
            return f'{s}{self.label}'
        return f'if [{self.label}] < {self.threshold}:\n{s} {self.children[0].format_tree(level + 1)}\n{s}else:\n{s} {self.children[1].format_tree(level + 1)}'

    def predict(self, data: list):
        if self.is_leaf():
            return self.label
        if data[self.label] < self.threshold:
            return self.children[0].predict(data)
        return self.children[1].predict(data)

    def __str__(self):
        return self.to_string()


class C45Algo:
    @staticmethod
    def log(e: Numeric):
        if e == 0:
            return 0
        return math.log(e, 2)

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def tree(self):
        data = self.dataset.values
        return self.generate_tree(data, list(range(len(data[0]) - 1)))

    def generate_tree(self, current_data: List[list], current_attribute: List[int]):
        if not len(current_data):
            return DTree("Fail")

        if not len(current_attribute):
            counter = Counter(map(itemgetter(-1), current_data))
            most_label = counter.most_common(1)[0][0]
            print(f'Attributes exhaust, {len(current_data)} data -> {most_label}')
            return DTree(most_label)

        all_same_class = self.all_same_class(current_data)
        if all_same_class:
            return DTree(current_data[0][-1])

        _gain, attribute, threshold, split = max(self.try_split(current_data, current_attribute), key=itemgetter(0))
        remaining = [att for att in current_attribute if att != attribute]
        node = DTree(attribute, threshold, [self.generate_tree(subset, remaining) for subset in split])
        return node

    @classmethod
    def all_same_class(cls, data: List[list]) -> bool:
        first_class = data[0][-1]
        return all(row[-1] == first_class for row in data)

    @classmethod
    def partition(cls, pred: Callable[[T], bool], ls: Iterable[T]) -> (List[T], List[T]):
        left = []
        right = []
        for it in ls:
            (left if pred(it) else right).append(it)
        return left, right

    @classmethod
    def try_split(cls, current_data: List[list], current_attribute: List[int]):
        for attribute in current_attribute:
            data = sorted(current_data, key=itemgetter(attribute))
            for d1, d2 in zip(data, data[1:]):
                threshold = (d1[attribute] + d2[attribute]) / 2

                less, greater = cls.partition(lambda x: x[attribute] < threshold, data)
                split = [less, greater]
                gain = cls.gain(data, split)
                yield gain, attribute, threshold, split

    @classmethod
    def gain(cls, data, subsets: Iterable[list]):
        length = len(data)
        impurity_before = cls.entropy(data)

        weights = [len(s) / length for s in subsets]

        impurity_after = sum(map(lambda p: (lambda x, w: w * cls.entropy(x))(*p), zip(subsets, weights)))
        return impurity_before - impurity_after

    @classmethod
    def entropy(cls, data):
        length = len(data)
        if not length:
            return 0

        counter = Counter(map(itemgetter(-1), data))
        possibilities = map(lambda x: x / length, counter.values())
        return - sum(map(lambda x: x * cls.log(x), possibilities))
