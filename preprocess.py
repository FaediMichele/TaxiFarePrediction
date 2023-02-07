"""Preprocess the given dataset and dump it to a parquet file."""
import argparse

DESCRIPTION = __doc__


class Namespace:
    """Custom namespace for CLI parameters."""
    input_path: str
    output_path: str


def main(namespace: Namespace):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    args = parser.parse_args(namespace=Namespace())
    main(args)
