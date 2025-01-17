# -*- coding: utf-8 -*-
# flake8: noqa: E501

import pathlib
from functools import partial

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pipelines.precipitation_model.impa.src.data.HDFDataset2 import HDFDataset2
from pipelines.precipitation_model.impa.src.data.HDFDatasetLocations import HDFDatasetLocations
from pipelines.precipitation_model.impa.src.models.mamba.lightning_radar import Vmamba_lightning as Vmamba_lightning_radar
from pipelines.precipitation_model.impa.src.models.mamba.lightning_sat import Vmamba_lightning as Vmamba_lightning_sat
from pipelines.precipitation_model.impa.src.utils.general_utils import print_ok
from pipelines.precipitation_model.impa.src.utils.hdf_utils import array_to_pred_hdf

SAT_MEAN = 0.08
SAT_STD = 0.39


def transform_SAT(X, std_fac):
    X = torch.log1p(torch.permute(X, (2, 0, 1)).nan_to_num(0.0))
    X_norm = (X - SAT_MEAN) / (std_fac * SAT_STD)
    X_norm[X_norm > 1] = 1
    X_norm[X_norm < -1] = -1
    return X_norm


def inv_transform_SAT(X, std_fac):
    return torch.expm1(torch.permute(X, (0, 2, 3, 1)) * std_fac * SAT_STD + SAT_MEAN)


def transform_RADAR(X):
    X = torch.permute(X, (2, 0, 1))
    X = torch.log1p(X).nan_to_num(0.0)
    return X


def inv_transform_RADAR(X):
    return torch.expm1(torch.permute(X, (0, 2, 3, 1)))


dataframe_dict = {
    "SATELLITE": {
        "transform": transform_SAT,
        "inv_transform": inv_transform_SAT,
        "dataset_class": HDFDatasetLocations,
        "lightning_module": Vmamba_lightning_sat,
    },
    "RADAR": {
        "transform": transform_RADAR,
        "inv_transform": inv_transform_RADAR,
        "dataset_class": HDFDataset2,
        "lightning_module": Vmamba_lightning_radar,
    },
}


def make_predictions(
    ds,
    transform,
    transform_kwargs,
    inv_transform,
    inv_transform_kwargs,
    model,
    batch_size,
    n_predict,
    n_after,
    output_predict_filepath,
    num_workers,
):
    ds.x_transform = partial(transform, **transform_kwargs)
    ds.y_transform = partial(transform, **transform_kwargs)
    test_dataloader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    s2 = len(ds.keys)
    ni = ds[0][0].shape[1]
    array_pre = torch.zeros((n_predict, s2, n_after, ni, ni))
    for i in range(n_predict):
        predictions = pl.Trainer(devices=[0], logger=False, enable_checkpointing=False).predict(
            model, test_dataloader
        )
        predictions = torch.cat(predictions, axis=0)
        predictions = predictions.squeeze(1)
        array_pre[i, :, :, :, :] = predictions

    predictions = torch.mean(array_pre, 0)
    predictions = inv_transform(predictions, **inv_transform_kwargs)
    array_to_pred_hdf(predictions, ds.keys, ds.future_keys, output_predict_filepath)


def main(args_dict, parameters_dict):
    dataframe_filepath = args_dict["dataframe_filepath"]
    locations = args_dict["locations"]
    output_predict_filepaths = args_dict["output_predict_filepaths"]
    input_model_filepath = args_dict["input_model_filepath"]
    data_type = args_dict["data_type"]

    batch_size = args_dict["batch_to_predict"]

    n_predict = 1
    n_after = args_dict["n_after"]
    n_after = 18
    n_before = args_dict["n_before"]
    if args_dict["data_modification"] is not None:
        leadtime_conditioning = "Lead_time_cond" in args_dict["data_modification"]
    else:
        leadtime_conditioning = False

    torch.set_float32_matmul_precision("medium")
    parameters_dict_copy = parameters_dict.copy()
    for flag in ["n_epoch", "std_fac", "weight_bool", "model_name"]:
        del parameters_dict_copy[flag]

    if (
        pathlib.Path(input_model_filepath).suffix == ".pt"
        or pathlib.Path(input_model_filepath).suffix == ".joblib"
    ):
        model = dataframe_dict[data_type]["lightning_module"](**parameters_dict_copy)
        model.load_state_dict(torch.load(input_model_filepath, map_location="cuda:0"))
    elif pathlib.Path(input_model_filepath).suffix == ".ckpt":
        model = dataframe_dict[data_type]["lightning_module"].load_from_checkpoint(
            input_model_filepath,
            map_location="cuda:0",
        )  # , **parameters_dict, context=ds[0][0], truth=ds[0][1])
    else:
        raise ValueError("Invalid model file extension.")
    model.eval()

    for output_predict_filepath in output_predict_filepaths:
        output_predict_filepath = pathlib.Path(output_predict_filepath)
        if output_predict_filepath.is_file():
            if args_dict["overwrite"]:
                output_predict_filepath.unlink()
            else:
                raise FileExistsError(
                    "Output file already exists. Pass 'overwrite' as true to overwrite."
                )

    # Load data
    if locations is None:
        ds = dataframe_dict[data_type]["dataset_class"](
            dataframe_filepath,
            n_after=n_after,
            n_before=n_before,
            use_datetime_keys=True,
            get_item_output=["X", "Y"],
        )
        make_predictions(
            ds,
            dataframe_dict[data_type]["transform"],
            dict(),
            dataframe_dict[data_type]["inv_transform"],
            dict(),
            model,
            batch_size,
            n_predict,
            n_after,
            output_predict_filepaths[0],
            args_dict["num_workers"],
        )

        ok_message = "OK: Saved predictions successfully."
        print_ok(ok_message)

        return

    for i, location in enumerate(locations):
        ds = HDFDatasetLocations(
            dataframe_filepath,
            [location],
            n_after=n_after,
            n_before=n_before,
            leadtime_conditioning=leadtime_conditioning,
        )
        make_predictions(
            ds,
            dataframe_dict[data_type]["transform"],
            {"std_fac": args_dict["std_fac"]},
            dataframe_dict[data_type]["inv_transform"],
            {"std_fac": args_dict["std_fac"]},
            model,
            batch_size,
            n_predict,
            n_after,
            output_predict_filepaths[i],
            args_dict["num_workers"],
        )

    ok_message = "OK: Saved predictions successfully."
    print_ok(ok_message)
