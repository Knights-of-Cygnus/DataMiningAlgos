import pandas as pd
import numpy as np
import math
from typing import Union, TypeVar, List, Iterable, Optional, Dict, Tuple
from operator import itemgetter, mul
from collections import Counter
import itertools
import functools

Numerable = Union[int, float]


def product(xs: Iterable[Numerable], one: Optional[Numerable] = None):
    return functools.reduce(mul, xs, one)


def normalize(xs: Iterable[Numerable]):
    res = list(xs)
    s = sum(res)
    return [x / s for x in res]


class NaiveBayes:
    @classmethod
    def gauss_of(cls, x: Numerable, average: Numerable, standard: Numerable) -> float:
        variance = standard ** 2
        return math.exp(-(x - average) ** 2 / 2 / variance) / standard / math.sqrt(2 * math.pi)

    @classmethod
    def get_average_and_standard(cls, nums: Iterable[Numerable]):
        variance = np.var(nums)
        average = np.average(nums)
        return average, math.sqrt(variance)

    def __init__(self, data: pd.DataFrame):
        self.data = data
        # self.gauss_parameters = list(zip(data.mean(), map(math.sqrt, data.var())))
        self.values = data.values
        counter = Counter(map(itemgetter(-1), self.values))
        length = len(data)
        self.possibility_labels = {k: v / length for k, v in counter.items()}
        self.labels = list(counter.keys())
        self.gauss_parameters: Dict[int, Dict[str, Tuple[Numerable, Numerable]]] = {}
        self.attributes = list(range(len(self.values[0]) - 1))

    def group_by_class(self):
        yield from self.data.groupby(len(self.values[0]) - 1)

    def train(self):
        p = {x: {} for x in self.attributes}
        for label, data in self.group_by_class():
            for attribute in self.attributes:
                d = data[attribute]
                mean, std = self.get_average_and_standard(d)
                p[attribute][label] = (mean, std)

        self.gauss_parameters = p

    def yield_possibilities_for_label(self, label, features: List[Numerable]):
        for attribute in self.attributes:
            mean, std = self.gauss_parameters[attribute][label]
            yield self.gauss_of(features[attribute], mean, std)

    def predict_possibility(self, features: List[Numerable]):
        ls = normalize(map(lambda label: sum(self.yield_possibilities_for_label(label, features)), self.labels))

        return dict(zip(self.labels, ls))

    def predict(self, features: List[Numerable]):
        return max(self.predict_possibility(features).items(), key=itemgetter(1))[0]


if __name__ == '__main__':
    data = pd.read_csv('iris.data', header=None)
    shuffled_data = data.sample(frac=1)
    split_at = int(len(data) * 0.9)
    train, predict = shuffled_data[:split_at], shuffled_data[split_at:]
    guass_nb = NaiveBayes(train)
    guass_nb.train()
    error = 0
    for row in predict.values:
        expect = row[-1]
        prediction = guass_nb.predict(row)
        if expect != prediction:
            print(f'Predict failure: {expect} => {prediction}')
            print(f'All possibilities: {guass_nb.predict_possibility(row)}')
            error += 1

    print(f'Total: {error} / {len(predict)}, rate: {error / len(predict)}')
