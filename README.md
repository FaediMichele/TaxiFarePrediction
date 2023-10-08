# TaxiFarePrediction
Machine learning applied to taxi rides fare prediction in New York city. Data comes from the Kaggle challenge: [New York City Taxi Fare Prediction
](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction).

## Installation
In order to execute the notebooks or run the `taxifare`'s package code it is necessary to install a set or dependencies. The dependencies are listed by `requirements.txt`. As of now, no setup script is provided for the library, so that you shall download the library and install the dependencies manually. Note that results can be directly inspected through the notebooks here on GitHub, without downloading or running them.

To download the code:
```
git clone https://github.com/FaediMichele/TaxiFarePrediction
```

To install the dependencies:
```
pip install -r requirements.txt
```
Once downloaded and installed, the notebooks can be run and/or the library can be manually used. `taxifare` is a proper Python package, so that it can be accessed by importing it and its submodules. E.g.
```py
import taxifare
```

## Project structure
The project is composed by a centralized Python package, containing all the necessary tools and logic, and a collection of notebooks which represent the different phases of our work. Results can be inspected by simply opening such notebooks here on GitHub, without downloading or running them. In particular:

* `taxifare` is the main Python package (see *Installation*)
* `analysis.ipynb` contains a preliminary analysis of the data. In particular of the spatial locations and temporal trends.
* `outlier_detection.ipynb` uses the information obtained during analysis in order to identify and eventually remove anomalies from the data.
* `neural_models.ipynb` contains prediction experiments through neural models. This notebook is designed to work on Google Colab.
* `osmnx.ipynb` contains the code used to approximate a graph of the New York city. Such graph is used to compute the approximate travel time of each taxi ride. This is a feature used by neural models.

Note that in order to reproduce these experiments, it is necessary to download data from the [proper Kaggle challenge](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction). Sometimes, some other preprocessing shall be applied a priori through some of the tools contained in `taxifare` (see *Command Line Interface*).

## Main experiments and results
See the notebooks (*Project structure*) for the complete analysis and methodology. Here are reported the main simple neural networks that were employed and the final results.

The used neural networks are simple fully connected ones, with ReLU non-linearities. The baseline is a degenerate network, which is just a linear layer having access to the features:

* Pickup coordinates
* Dropoff coordinates
* Passenger count
* Pickup datetime (month, week, hour)

The main non-linear experiments make use of two similar networks having this shape (fully connected, ReLU non-linearities):
![model1](https://github.com/FaediMichele/TaxiFarePrediction/assets/12080380/6512c830-7ba1-4fd5-9fa7-c3dff7214acb)

More features are provided in this case, some of them are engineered based on the analysis (see `analysis.ipynb`):

* Pickup coordinates
* Dropoff coordinates
* Passenger count
* Pickup datetime (year, month, day of week, hour)
* Boolean: is the pickup datetime after Setpember 2012? (see analysis for the reason)
* Boolean: is the pickup datetime during the weekend?

The only difference between the two models is an extra feature: the approximate travel time of the ride. The travel time is approximated through the NYC graph, obtained by running `osmnx.ipynb`. However, this graph only covers the urban area of the city. Since some of the rides in the dataset are outside the urban area, it was not possible to obtain such feature for them. For this reason, two models are trained and combined. One having access to the urban feature of travel time, predicting all the urban rides, and one without it, predicting all the suburban rides.

### Results
For the train-time results and evaluations see `neural_models.ipynb`. Here are reported the results of the submission of our experiments to the Kaggle leaderboard of the competition:
![kaggle_results](https://github.com/FaediMichele/TaxiFarePrediction/assets/12080380/a5bb4800-7636-4c0f-b5c2-77c8ac1d34b9)

Even though the competition is long finished, our best score would place us around the first 30% of the leaderboard. The best score (MLP2048 + MLP2048) refers to the double neural network setup described above.

## Command Line Interface
The `taxifare` package contains a few CLI entry points that makes it possible to preprocess the data using the pipelines preprogrammed in the package.

### `taxifare.preprocess`
Can be invoked through `python -m taxifare.preprocess ...`. Here the output of the help command: `python -m taxifare.preprocess -h`.
```
usage: preprocess.py [-h] [-p OPERATION] [-n SAMPLES] [--for-autoencoder] [--ree-model REE_MODEL] [--ree-threshold REE_THRESHOLD] [--for-train] [--city-graph CITY_GRAPH]
                     [--city-distance-matrix CITY_DISTANCE_MATRIX]
                     INPUT_FILE OUTPUT_FILE

Preprocess the given dataset and dump it to a parquet file. Entrypoint, run with ``python -m texifare.preprocess [...]``

positional arguments:
  INPUT_FILE
  OUTPUT_FILE

options:
  -h, --help            show this help message and exit
  -p OPERATION, --preprocess OPERATION
  -n SAMPLES, --num-samples SAMPLES
  --for-autoencoder
  --ree-model REE_MODEL
  --ree-threshold REE_THRESHOLD
  --for-train
  --city-graph CITY_GRAPH
  --city-distance-matrix CITY_DISTANCE_MATRIX
```
The script expects as input the unprocessed csv data from [the Kaggle challenge](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) and dumps to a parquet file the processed dataset.

Main expected usage is to preprocess the data for neural models to train on. This is achieved by:
```
python -m taxifare.preprocess --for-train --city-graph ny-graph.graphml --city-distance-matrix distance-matrix.pkl data_from_kaggle.csv processed_data.parquet
```
In this example a file `processed_data.parquet` will be generated, containing a processed version of `data_from_kaggle.csv`. The distance NYC distance matrix `distance-matrix.pkl` and `ny-graph.graphml` are also required. These can be obtained by running `osmnx.ipynb`.

Note that *autoencoder* and *ree* options are related to anomaly detection through reconstruction error, but were not used in the final experiment.

### `taxifare.split`
Can be invoked through `python -m taxifare.split ...`. Here the output of the help command: `python -m taxifare.split -h`.
```
usage: split.py [-h] [-r] [-s SEED] input_path train_split_path valid_split_path test_split_path

positional arguments:
  input_path
  train_split_path
  valid_split_path
  test_split_path

options:
  -h, --help            show this help message and exit
  -r, --random-shuffle
  -s SEED, --seed SEED
```
The script expects as input a preprocessed dataset (probably through `taxifare.preprocess`) splitting it into train, validation and test splits. These tree datasets are dumped into parquet files. The ratio is currently fixed at 80%/10%/10%. Shuffling and seeding are optional. E.g.
```
python -m taxifare.split -r -s 1 processed_data.parquet train.parquet valid.parquet test.parquet
```
This example reads the preprocessed dataset `processed_data.parquet`, shuffles it with seed `1` and outputs the splits to `train.parquet` (80% of the samples), `valid.parquet` (10% of the samples) and `test.parquet` (10% of the samples).

## Code style
Most code is documented though docstrings and inline comments, and styled as described by Python's [PEP8](https://peps.python.org/pep-0008/).
