# -*- coding: utf-8 -*-
# flake8: noqa: E501

import importlib
import pathlib
from functools import partial

import joblib
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.utils.data import DataLoader

from pipelines.precipitation_model.impa.src.data.HDFDatasetLocations import (
    HDFDatasetLocations,
)
from pipelines.precipitation_model.impa.src.data.PredHDFDatasetLocations import (
    PredHDFDatasetLocations,
)
from pipelines.precipitation_model.impa.src.utils.data_utils import sat_dataframe
from pipelines.precipitation_model.impa.src.utils.general_utils import print_ok
from pipelines.precipitation_model.impa.src.utils.hdf_utils import array_to_pred_hdf
from pipelines.precipitation_model.impa.src.utils.models_utils import (
    get_ds,
)

MEAN_LOG_SAT = 0.08
STD_LOG_SAT = 0.39

MEAN_LOG_RAD = 0.32662693
STD_LOG_RAD = 1.2138965


def transform0(X, mean, std, sat=False):
    # Normalization from -1 to 1
    X = torch.permute(X.nan_to_num(0.0), (2, 0, 1))
    X = torch.log1p(X)
    X_norm = (X - mean) / (3 * std)
    # X_norm[X_norm > 1] = 1
    # X_norm[X_norm < -1] = -1
    return X_norm


def transform1(X, mean, std):
    # Normalization from 0 to 1
    # Use to train the data of satellite with log
    X_norm = torch.log1p(torch.permute(X.nan_to_num(0.0), (2, 0, 1))) / (6 * std)
    X_norm[X_norm > 1] = 1
    return X_norm


def transform2(X, mean, std, sat=False):
    X = torch.permute(X.nan_to_num(0.0), (2, 0, 1))
    if not sat:
        X = torch.log1p(X)
    return X


def transform3(X, mean, std, sat=False):
    X = X.nan_to_num(0.0)
    return X


def transform4(X, mean=0, std=1, sat=False):
    # The metnet transformation
    X = X.nan_to_num(0.0)
    X = torch.tanh(torch.log1p(X) / 4)
    return X


def transform5(X, mean, std, sat=False):
    X = X.nan_to_num(0.0)
    X = (torch.log1p(X) - mean) / (3 * std)
    return X


def transform6(X, mean, std, sat=False):
    X = X.nan_to_num(0.0)
    X = torch.nn.functional.interpolate(
        X.T, scale_factor=0.25, mode="bilinear", align_corners=False
    ).T
    return X


def transform7(X, mean, std, sat=False):
    X = X.nan_to_num(0.0)
    X = (torch.log1p(X) - mean) / (3 * std)
    X = torch.nn.functional.interpolate(
        X.T, scale_factor=0.25, mode="bilinear", align_corners=False
    ).T
    return X


def inv_transform0(X, mean, std):
    return torch.expm1(torch.permute(X, (0, 2, 3, 1)) * 3 * std + mean)


def inv_transform1(X, mean, std):
    return torch.expm1(torch.permute(X, (0, 2, 3, 1)) * 6 * std)


def inv_transform2(X, mean, std, sat=False):
    if not sat:
        X = torch.expm1(X)
    return torch.permute(X, (0, 2, 3, 1))


def inv_transform3(X, mean, std, sat=False):
    return torch.permute(X, (0, 2, 3, 1))


def inv_transform4(X, mean, std, sat=False):
    return torch.expm1(torch.atanh(X) * 4)


def inv_transform5(X, mean, std, sat=False):
    return torch.expm1((X * 3 * std) + mean)


# flake8: noqa: C901
def main(args_dict, parameters_dict):
    predict_sat = args_dict["predict_dataset"] == True
    # if args_dict["predict_dataset"] is None and args_dict["dataframe"] != "RADAR-d2CMAX-DBZH-file=thr=0_split_radar":
    #     del args_dict["predict_dataset"]
    # del args_dict["predict_dataset"]
    model_name = args_dict["model"]
    del args_dict["model"]

    try:
        merge = args_dict["merge"]
        del args_dict["merge"]
    except KeyError:
        merge = False

    model_location = f"src.models.{model_name}.lightning"
    model = importlib.import_module(model_location).model

    dataframe_filepath = args_dict["dataframe_filepath"]
    locations = args_dict["locations"]
    output_predict_filepaths = args_dict["output_predict_filepaths"]
    input_model_filepath = args_dict["input_model_filepath"]

    batch_size = args_dict["batch_to_predict"]
    # n_predict = 1
    n_after = args_dict["n_after"]
    n_before = args_dict["n_before"]
    new_dataset = locations is not None
    needs_prediction = model_name in ["NowcastNet"]

    sat = args_dict["dataframe_filepath"] in sat_dataframe

    # if args_dict["data_modification"] is None:
    #     lead_time = False
    # else:
    #     lead_time = "Lead_time_cond" in args_dict["data_modification"]

    lead_time = False

    try:
        del parameters_dict["n_epochs"]
    except KeyError:
        pass

    torch.set_float32_matmul_precision("medium")

    if args_dict["normalized"] == 0:
        transform = transform0
        inv_transform = inv_transform0
    elif args_dict["normalized"] == 1:
        transform = transform1
        inv_transform = inv_transform1
    elif args_dict["normalized"] == 2:
        transform = transform2
        inv_transform = inv_transform2
    elif args_dict["normalized"] == 3:
        transform = transform3
        inv_transform = inv_transform3
    elif args_dict["normalized"] == 4:
        transform = transform4
        inv_transform = inv_transform4
    elif args_dict["normalized"] == 5:
        transform = transform5
        inv_transform = inv_transform5

    print("Predicting using the model: ", input_model_filepath)

    if pathlib.Path(input_model_filepath).suffix == ".joblib":
        if not needs_prediction:
            pipe = joblib.load(open(input_model_filepath, "rb"))
        else:
            pipe = model(
                **parameters_dict,
                merge=merge,
                satellite=sat,
                map_location="cuda:0" if args_dict["accelerator"] == "cuda" else "cpu",
            )
            pipe.load_state_dict(torch.load(input_model_filepath))

    elif pathlib.Path(input_model_filepath).suffix == ".ckpt":
        pipe = model.load_from_checkpoint(
            input_model_filepath,
            **parameters_dict,
            map_location="cuda:0" if args_dict["accelerator"] == "cuda" else "cpu",
        )
    else:
        raise ValueError("Invalid model file extension.")
    pipe.eval()

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
    if new_dataset:
        for i, location in enumerate(locations):
            if not needs_prediction:
                ds = HDFDatasetLocations(
                    dataframe_filepath,
                    [location],
                    n_after=n_after,
                    n_before=n_before,
                    leadtime_conditioning=lead_time,
                )
            else:
                n_predictions = n_after

                ds = PredHDFDatasetLocations(
                    dataframe_filepath,
                    [location],
                    n_predictions=n_predictions,
                    dataset=args_dict["predict_dataframe"],
                    n_after=n_after,
                    n_before=n_before,
                )
            ds.x_transform = partial(transform, mean=0, std=1, sat=True)
            ds.y_transform = partial(transform, mean=0, std=1, sat=True)
            test_dataloader = DataLoader(
                ds, batch_size=batch_size, num_workers=args_dict["num_workers"]
            )

            # s2 = len(ds.keys)
            # ni = ds[0][0].shape[1]

            predictions = pl.Trainer(
                accelerator=args_dict["accelerator"],
                logger=False,
                enable_checkpointing=False,
            ).predict(pipe, test_dataloader)

            predictions = torch.cat(predictions, axis=0)
            predictions[predictions < 0] = 0
            predictions = predictions.squeeze(1)

            if lead_time:
                # Reshape the predictions:
                predictions = rearrange(predictions, "(b c) w i -> b c w i", c=n_after)

            predictions = inv_transform(predictions, mean=0, std=1, sat=True)
            array_to_pred_hdf(predictions, ds.keys, ds.future_keys, output_predict_filepaths[i])
    else:
        ds, _ = get_ds(
            dataframe_filepath,
            n_before,
            n_after,
            new_dataset,
            merge,
            lead_time=False,
            predict_sat=predict_sat,
            locations=locations,
            needs_prediction=needs_prediction,
            args_dict=args_dict,
        )

        if sat and args_dict["normalized"] != 1:
            try:
                mean_data = ds.train_mean
                std_data = ds.train_std
            except AttributeError:
                mean_data = MEAN_LOG_SAT
                std_data = STD_LOG_SAT
        else:
            try:
                mean_data = ds.train_log_mean
                std_data = ds.train_log_std
            except AttributeError:
                mean_data = MEAN_LOG_RAD
                std_data = STD_LOG_RAD

        ds.x_transform = partial(transform, mean=mean_data, std=std_data, sat=sat)
        ds.y_transform = partial(transform, mean=mean_data, std=std_data, sat=sat)

        test_dataloader = DataLoader(
            ds, batch_size=batch_size, num_workers=args_dict["num_workers"]
        )

        predictions = pl.Trainer(
            accelerator=args_dict["accelerator"],
            logger=False,
            enable_checkpointing=False,
        ).predict(pipe, test_dataloader)
        predictions = torch.cat(predictions, axis=0)

        predictions = inv_transform(predictions, mean=mean_data, std=std_data, sat=sat)

        array_to_pred_hdf(predictions, ds.keys, ds.future_keys, output_predict_filepaths[0])

    ok_message = "OK: Saved predictions successfully."
    print_ok(ok_message)
