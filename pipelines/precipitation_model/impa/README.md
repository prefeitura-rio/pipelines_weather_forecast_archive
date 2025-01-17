# Rio Rain

This project aims to provide a pipeline for real time prediction in Rio de Janeiro. It contains a script that automatically downloads the most recent data to then process it and, finally, make predictions for the next 3 hours.

## Table of Contents

- [Rio Rain](#rio-rain)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Main script](#main-script)
    - [Evaluation](#evaluation)
  - [File structure](#file-structure)

## Introduction

This project contains the necessary code for nowcasting with 5 different models. It also includes a script that plots the predictions.

## Features

- Real-time prediction
- Nowcasting visualization
- Performance of each model in the last 3 hours

## Installation

To use the scripts contained in the provided zip file, the necessary libraries must first be correctly installed. There are different ways to do it:

- Using `poetry`:
  To install the necessary requirements using `poetry`, you may open the terminal on the root directory and call `poetry install`. Unfortunately, there is a necessary library that cannot be installed through `poetry`, so you can instead use `pip` for this particular library calling `pip install --no-use-pep517 mamba-ssm`.

- Using `pip`:
  To install the dependencies using `pip`, you can use the provided file `requirements.txt` by calling `pip install -r requirements.txt`.

- Using other project managers:
  You may also use other project managers like `conda` directly installing the libraries contained in `requirements.txt`.

## Usage

### Main script

To produce predictions in real time, it is enough to call `python src/eval/update-real_time.py` on the root directory in the correct project environment. Once this command is called, the script will run the following tasks, in order:

- Download GOES-16 data from AWS
- Process the data locally
- Build a dataset appropriate for making predictions
- Make predictions for each of the provided models

Once the script stops running, you can find the output files as described in the [file structure](#file-structure) section.

Note that there are optional arguments for this command:

`python src/eval/update-real_time_[dataset].py [--cuda] [--num_workers] [--datetime]`,

`[dataset]` represents the type of data to be used and may be one of `SAT` (GOES-16 RRQPE product) and `MDN` (Mendanha radar).

Here `--cuda` may be passed if you want to make predictions though GPU computing. If you want to do all calculations in CPU, you should not pass the optional argument `--cuda`. This will be slower and some models may not work in this mode.

On the other hand `[--num_workers]` is the number of processes that may be run in parallel. It must be an integer value greater than zero. Generally, the larger this number is, the faster the script reaches its conclusion.

Finally, `[--datetime]` may be passed to make predictions from the time passed in UTC. The format passed must be '%Y-%m-%d %H:%M:%S', so if we want predictions from 13/01/2024 14:00:00 BRT, we must call `python src/eval/update-real_time_SAT.py --datetime '2024-01-13 17:00:00'`.

If it is desired to make predictions for just some of the models, it is possible to edit the file `src/eval/real_time_config.json` and delete the dictionary entries associated to the model that is to be excluded. Be mindful that the model `EVONET` is necessary for predicting with `NowcastNet`.

### Evaluation

Scripts for evaluating model predictions for the last three hours are also made available. Once the predictions are made through the main script, you may call `python src/eval/viz/plot-real_time.py [dataset] [--num_workers]` to produce plots or `python src/eval/metrics/calc-metrics.py [--num_workers]` to calculate metrics.


## File structure

```
rio-rain
│   README.md
│   poetry.lock
│   pyproject.toml
│
└───data
│   │
│   └───dataframe_grids
│       │   ...
|       dataframes
|       |   ...
|       processed
|       |   ...
|       raw
|       └   ...
└───eval
|   └  ...
|
└───models
|   └  ...
|
└───predictions
|   └  ...
|
└───src
    └  ...
```

In the `data` folder, the coordinate grids associated to the points in Earth's surface, raw data downloaded from AWS, locally processed data and dataframes ready for prediction are found in their respective subfolder.

In the `eval` folder, the output metrics and prediction graphs are found in their respective subfolder.

In the `models_[dataset]` folder, the parameters and necessary information associated to each model are found.

In the `predictions_[dataset]` folder, the output predictions are saved in `.hdf` files which contain predictions for each model. In these files, the data is organized as follows:

```
root
│
└───date1
│   │
│   └───time1
|       |
│       └───datetime1
|       |
│       └───datetime2
|       |
│       └───   ...
|       |
|       time2
|       |   ...
|
└───date2
|   └  ...
|   ...
```

In this hierarchical structure, `date1`, `date2`, etc. refer to the date of the last observation informed to the model. `time1`, `time2`, etc. refer to the time of the last observation informed to the model. `datetime1`, `datetime2`, etc. represent the date and time of the prediction being made. All of these dates and times are in accordance with UTC timezone.

In the leaf nodes given by `datetime1`, `datetime2`, etc. 256x256 matrices which represent the predictions are found. The coordinates of these pixels is in accordance with the `.npy` file found in `data/dataframe_grids/rio_de_janeiro-res=2km-256x256.npy`.

Finally, we also have the `src` folder, in which all source code is found, relating to the downloading and processing of data, the scripts that load the saved models and make predictions, as well as the codes that evaluate the predictions.
