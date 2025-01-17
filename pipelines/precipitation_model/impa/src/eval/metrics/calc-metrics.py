# -*- coding: utf-8 -*-
import datetime
import json
import pathlib
from argparse import ArgumentParser
from multiprocessing.pool import Pool

import h5py
import numpy as np
import pandas as pd
import tqdm

from pipelines.precipitation_model.impa.src.eval.metrics.metrics import metrics_dict
from pipelines.precipitation_model.impa.src.utils.general_utils import print_warning
from pipelines.precipitation_model.impa.src.utils.hdf_utils import get_dataset_keys

dataset_dict = {
    "SAT": {
        "ground_truth_file": "pipelines/precipitation_model/data/dataframes/SAT-CORRECTED-ABI-L2-RRQPEF-real_time-rio_de_janeiro/test.hdf",
        "grid_file": "pipelines/precipitation_model/data/dataframe_grids/rio_de_janeiro-res=2km-256x256.npy",
    },
    "MDN": {
        "ground_truth_file": "pipelines/precipitation_model/data/dataframes/MDN-d2CMAX-DBZH-real_time/test.hdf",
        "grid_file": "pipelines/precipitation_model/data/dataframe_grids/rio_de_janeiro-res=700m-256x256.npy",
    },
}

parser = ArgumentParser()
parser.add_argument("dataset", type=str, choices=["SAT", "MDN"])
parser.add_argument("--num_workers", type=int, default=16)
args = parser.parse_args()

NLAGS = 18
METRICS_NAMES = ["log-MAE", "log-MSE", "CSI1", "CSI8"]
order = np.array([[1, 1, -1, -1]]).reshape(1, -1)

config = pathlib.Path(f"pipelines/precipitation_model/impa/src/eval/real_time_config_{args.dataset}.json")
with open(config, "r") as json_file:
    specs_dict = json.load(json_file)

ground_truth_df = h5py.File(dataset_dict[args.dataset]["ground_truth_file"])
latlons = np.load(dataset_dict[args.dataset]["grid_file"])
feature = ground_truth_df["what"].attrs["feature"]
timestep = int(ground_truth_df["what"].attrs["timestep"])

keys = get_dataset_keys(ground_truth_df)
past_obs = keys[-(NLAGS + 1)]
past_obs_dt = pd.to_datetime(past_obs)

preds = [ground_truth_df]
model_names = ["Ground truth"]
for model_name in specs_dict["models"].keys():
    predictions = f"predictions_{args.dataset}/{model_name}.hdf"  # aqui
    try:
        pred_hdf = h5py.File(predictions)
    except FileNotFoundError:
        print_warning(f"File {predictions} not found. Skipping model {model_name}...")
        continue
    preds.append(pred_hdf)
    model_names.append(model_name)


# task to return dict with metrics
# flake8: noqa: C901
def task_lag(lag: int):
    past_dt = past_obs_dt + datetime.timedelta(minutes=lag * timestep)
    metrics_array = np.zeros((len(model_names) - 1, len(METRICS_NAMES)))

    # calculate metrics for each model
    for i in range(len(model_names)):
        pred = preds[i]
        if i == 0:
            future_key = past_dt.strftime("%Y%m%d/%H%M")
        else:
            future_key = past_dt.strftime("%Y%m%d-%H%M")
            future_key = f"{past_obs}/{future_key}"
        try:
            if i == 0:
                values = np.array(pred[future_key]).reshape((*latlons.shape[:2], -1))[:, :, 0]
            else:
                values = np.array(pred[future_key])
            if i == 0:
                ground_truth = values
            else:
                for j, metric_name in enumerate(METRICS_NAMES):
                    metric = metrics_dict[metric_name]
                    try:
                        metrics_array[i - 1, j] = metric(values, ground_truth)
                    except ValueError:
                        metrics_array[i - 1, j] = np.nan
        except KeyError:
            pass

    full_nan_column = np.all(np.isnan(metrics_array), axis=0)
    best_metrics = np.ones(len(METRICS_NAMES)) * np.nan
    worst_metrics = np.ones(len(METRICS_NAMES)) * np.nan

    best_metrics[~full_nan_column] = np.nanargmin(
        metrics_array[:, ~full_nan_column] * order[:, ~full_nan_column], axis=0
    )
    worst_metrics[~full_nan_column] = np.nanargmax(
        metrics_array[:, ~full_nan_column] * order[:, ~full_nan_column], axis=0
    )

    # make dataframe entry
    dicts = []
    for i, model_name in enumerate(model_names[1:]):
        d = dict()
        d["lag"] = lag
        d["datetime(UTC)"] = past_dt.strftime("%Y-%m-%d %H:%M")
        d["model"] = model_name
        for j, metric_name in enumerate(METRICS_NAMES):
            d[metric_name] = metrics_array[i - 1, j]
        for j, metric_name in enumerate(METRICS_NAMES):
            d[f"worst_{metric_name}"] = (
                worst_metrics[j] == i if not np.isnan(worst_metrics[j]) else np.nan
            )
            d[f"best_{metric_name}"] = (
                best_metrics[j] == i if not np.isnan(best_metrics[j]) else np.nan
            )
        dicts.append(d)

    df = pd.DataFrame(dicts)
    return df


with Pool(min(NLAGS, args.num_workers)) as pool:
    dfs = list(tqdm.tqdm(pool.imap(task_lag, list(range(1, NLAGS + 1))), total=NLAGS))

# save dataframe
df = pd.concat(dfs)
metrics_filepath = pathlib.Path(f"pipelines/precipitation_model/eval/metrics/metrics-{args.dataset}.csv")
metrics_filepath.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(metrics_filepath, index=False, na_rep="nan")
