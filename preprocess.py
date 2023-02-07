"""Preprocess the given dataset and dump it to a parquet file."""
import argparse
import util

DESCRIPTION = __doc__


class Namespace:
    """Custom namespace for CLI parameters."""
    input_path: str
    output_path: str


def main(namespace: Namespace):
    """Process the given arguments."""
    df = util.load_data(namespace.input_path)
    df.sink_parquet(namespace.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('input_path', type=str, metavar='INPUT_FILE')
    parser.add_argument('output_path', type=str, metavar='OUTPUT_FILE')

    args = parser.parse_args(namespace=Namespace())
    main(args)
