import datetime
import json
import pathlib
from argparse import ArgumentParser
from multiprocessing.pool import Pool

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from src.eval.metrics.metrics import metrics_dict
from src.utils.eval_utils import get_img
from src.utils.general_utils import print_warning
from src.utils.hdf_utils import get_dataset_keys

dataset_dict = {
    "SAT": {
        "ground_truth_file": "data/dataframes/SAT-CORRECTED-ABI-L2-RRQPEF-real_time-rio_de_janeiro/test.hdf",
        "grid_file": "data/dataframe_grids/rio_de_janeiro-res=2km-256x256.npy",
    },
    "MDN": {
        "ground_truth_file": "data/dataframes/MDN-d2CMAX-DBZH-real_time/test.hdf",
        "grid_file": "data/dataframe_grids/rio_de_janeiro-res=700m-256x256.npy",
    },
}

parser = ArgumentParser()
parser.add_argument("dataset", type=str, choices=["SAT", "MDN"])
parser.add_argument("--num_workers", type=int, default=16)
args = parser.parse_args()

NLAGS = 18
BG_COLOR = "white"
METRICS_NAMES = ["log-MAE", "log-MSE", "CSI1", "CSI8"]
order = np.array([[1, 1, -1, -1]]).reshape(1, -1)

HEIGHT = 500
WIDTH = 500

config = pathlib.Path(f"src/eval/real_time_config_{args.dataset}.json")
with open(config, "r") as json_file:
    specs_dict = json.load(json_file)

ground_truth_df = h5py.File(dataset_dict[args.dataset]["ground_truth_file"])
latlons = np.load(dataset_dict[args.dataset]["grid_file"])
feature = ground_truth_df["what"].attrs["feature"]
timestep = int(ground_truth_df["what"].attrs["timestep"])

keys = get_dataset_keys(ground_truth_df)
last_obs = keys[-1]
past_obs = keys[-(NLAGS + 1)]
last_obs_dt = pd.to_datetime(last_obs)
past_obs_dt = pd.to_datetime(past_obs)

preds = [ground_truth_df]
model_names = ["Ground truth"]
for model_name in specs_dict["models"].keys():
    predictions = f"predictions_{args.dataset}/{model_name}.hdf"
    if (
        "plot" in specs_dict["models"][model_name].keys()
        and specs_dict["models"][model_name]["plot"] == False
    ):
        continue
    try:
        pred_hdf = h5py.File(predictions)
    except FileNotFoundError:
        print_warning(f"File {predictions} not found. Skipping model {model_name}...")
        continue
    preds.append(pred_hdf)
    model_names.append(model_name)


# flake8: noqa: C901
def task_lag(lag: int):
    future_dt = last_obs_dt + datetime.timedelta(minutes=lag * timestep)
    past_dt = past_obs_dt + datetime.timedelta(minutes=lag * timestep)
    output_filepath = pathlib.Path(f"eval/viz/plot-real_time-{args.dataset}/lag={lag}.png")

    future_time = (future_dt - datetime.timedelta(hours=3)).strftime("%H:%M")
    past_time = (past_dt - datetime.timedelta(hours=3)).strftime("%H:%M")
    present_time = (last_obs_dt - datetime.timedelta(hours=3)).strftime("%H:%M")
    future_imgs = len(preds) * [np.zeros((HEIGHT, WIDTH, 4), dtype=int)]
    past_imgs = len(preds) * [np.zeros((HEIGHT, WIDTH, 4), dtype=int)]
    metrics_array = np.zeros((len(model_names) - 1, len(METRICS_NAMES)))
    # Predict future
    for i, model_name in enumerate(model_names):
        pred = preds[i]
        if i == 0:
            future_key = last_obs
        else:
            future_key = future_dt.strftime("%Y%m%d-%H%M")
            future_key = f"{last_obs}/{future_key}"
        try:
            if i == 0:
                values = np.array(pred[future_key]).reshape((*latlons.shape[:2], -1))[:, :, 0]
                future_imgs[i] = get_img(
                    values,
                    latlons,
                    "Last observation",
                    feature,
                    None,
                    bg_color=BG_COLOR,
                    height=HEIGHT,
                    width=WIDTH,
                    no_colorbar=True,
                )
            else:
                values = np.array(pred[future_key])
                future_imgs[i] = get_img(
                    values,
                    latlons,
                    model_name,
                    feature,
                    None,
                    bg_color=BG_COLOR,
                    height=HEIGHT,
                    width=WIDTH,
                    no_colorbar=True,
                )
        except KeyError:
            pass

    # Predict past
    for i, model_name in enumerate(model_names):
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
            past_imgs[i] = get_img(
                values,
                latlons,
                model_name,
                feature,
                None,
                bg_color=BG_COLOR,
                height=HEIGHT,
                width=WIDTH,
                no_colorbar=True,
            )
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
    best_metrics = np.argmin(metrics_array * order, axis=0)
    worst_metrics = np.argmax(metrics_array * order, axis=0)
    pathlib.Path(output_filepath).parents[0].mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(figsize=(5 * len(preds), 10), ncols=len(preds), nrows=2)
    imgs = future_imgs + past_imgs
    present_dt = last_obs_dt - datetime.timedelta(hours=3)
    textstr_up = f"Predictions based on data up to {present_time}"
    textstr_down = f'Predictions based on data up to {(past_obs_dt - datetime.timedelta(hours=3)).strftime("%H:%M")}'
    for i in range(2):
        for j in range(len(preds)):
            ax = axs[i, j]
            ax.imshow(imgs[len(preds) * i + j])
            if j == 0:
                ax.set_facecolor((0.1, 0.2, 0.5))
                ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
            else:
                ax.axis("off")
            ax.axis("tight")
            if i == 0 and j == 0:
                time = present_time
            elif i == 0 and j != 0:
                time = future_time
            else:
                time = past_time
            props = dict(boxstyle="round", facecolor="black", alpha=0.9)
            ax.text(
                0.15,
                0.1,
                time,
                backgroundcolor="black",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontdict={"fontsize": 24, "family": "alarm clock", "color": "red"},
                bbox=props,
            )

            if i == 1 and j != 0:
                for k, metric_name in enumerate(METRICS_NAMES):
                    if j - 1 == best_metrics[k]:
                        color = "green"
                    elif j - 1 == worst_metrics[k]:
                        color = "red"
                    else:
                        color = "black"
                    text = f"{metric_name}: {metrics_array[j-1,k]:.2f}"
                    ax.text(
                        0.85,
                        0.21 - k * 0.06,
                        text,
                        backgroundcolor="white",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                        # bbox=props,
                        fontdict={"fontsize": 12, "color": color},
                    )
    fig.tight_layout()
    line = plt.Line2D(
        [0, 1],
        [0.506, 0.506],
        transform=fig.transFigure,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    fig.add_artist(line)
    xpos = 1 / len(preds)
    fig.subplots_adjust(bottom=0.1, top=0.9)
    plt.text(0.5, 0.975, str(present_dt.date()), fontsize=22, transform=fig.transFigure)
    plt.text(xpos, 0.945, textstr_up, fontsize=18, transform=fig.transFigure)
    plt.text(xpos, 0.045, textstr_down, fontsize=18, transform=fig.transFigure)
    fig.savefig(output_filepath, transparent=True)
    fig.clf()
    plt.close()


with Pool(min(NLAGS, args.num_workers)) as pool:
    list(tqdm.tqdm(pool.imap(task_lag, list(range(1, NLAGS + 1))), total=NLAGS))
