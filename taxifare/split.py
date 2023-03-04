import argparse
from typing import Optional

import polars as pl

from taxifare import data


class Namespace:
    input_path: str
    valid_split: float = 0.1
    test_split: float = 0.1
    train_split_path: str
    valid_split_path: str
    test_split_path: str
    shuffle: bool = False
    seed: Optional[int] = None


def split(namespace: Namespace):
    df = pl.read_parquet(namespace.input_path)

    train, valid, test = data.split(df, namespace.valid_split,
                                    namespace.test_split, namespace.seed,
                                    namespace.shuffle)

    train.write_parquet(namespace.train_split_path)
    valid.write_parquet(namespace.valid_split_path)
    test.write_parquet(namespace.test_split_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', type=str)
    parser.add_argument('train_split_path', type=str)
    parser.add_argument('valid_split_path', type=str)
    parser.add_argument('test_split_path', type=str)
    parser.add_argument('-r', '--random-shuffle', action='store_true',
                        dest='shuffle')
    parser.add_argument('-s', '--seed', type=int, dest='seed')

    namespace = parser.parse_args(namespace=Namespace())
    split(namespace)
