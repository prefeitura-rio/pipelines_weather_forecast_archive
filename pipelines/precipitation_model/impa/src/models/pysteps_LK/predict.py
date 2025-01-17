import pathlib
from functools import partial

import numpy as np
import torch
from joblib import Parallel, delayed
from pysteps import motion, nowcasts
from pysteps.utils import conversion, transformation

from src.data.HDFDataset2 import HDFDataset2
from src.data.HDFDatasetLocations import HDFDatasetLocations
from src.utils.general_utils import print_ok
from src.utils.hdf_utils import array_to_pred_hdf


def r_to_z(value: int, a_z: float = 223.0, b_z: float = 1.56):
    """Convert from rain rate to reflectivity using Z-R relation

    Args:
        value: value to be converted
        a_z: linear coefficient of Z-R relation
        b_z: exponent of Z-R relation

    Returns: Float of converted value
    """

    return a_z * (10 ** (value / 10)) ** b_z


def make_pred(it: int, hdf, adv_scheme: str, motion_f: str, data_type: str) -> np.array:
    """Make predictions for reflectivity using pySTEPS

    Args:
        it: int with entry of hdf to be used (set so function can run in parallel)
        hdf: hdf file with data
        adv_scheme: string with advection scheme name accepted by pySTEPS
        motion_f: string with motion field method name accepted by pySTEPS
        data_type: string discerning radar from satellite observations

    Returns:
        Array with predictions
    """
    X, y = hdf[it][:2]

    metadata = {"transform": None, "zerovalue": -15.0, "threshold": 0.1}

    # Define variables for conversion to precipitation
    zero = metadata["zerovalue"]
    thr = metadata["threshold"]
    a_z = 223.0
    b_z = 1.53

    # Transform data

    # Change order so time comes first and put lags in opposite order
    if data_type == "RADAR":
        Xt = np.array(X).transpose((2, 0, 1))[::-1, :, :]
        # Change order of dimensions so time comes first
        yt = np.array(y).transpose((2, 0, 1))

    else:
        Xt = np.array(X).transpose((2, 0, 1))[::2][::-1, :, :]
        # Change order of dimensions so time comes first
        yt = np.array(y).transpose((2, 0, 1))[:36:2]

    # Set motion field estimator
    oflow_method = motion.get_method(motion_f)

    # Set advection scheme
    extrapolate = nowcasts.get_method(adv_scheme)
    n_leadtimes = yt.shape[0]

    if adv_scheme == "steps":
        if data_type == "RADAR":
            metadata["unit"] = "dBZ"
            km = 0.5
            t = 2

        else:
            metadata["unit"] = "mm/h"
            km = 2
            t = 10

        # Transform data to rain rate for correct use of STEPS method

        # Use Z-R relation
        # Note that not much is changed if unit is set to mm/h
        train_precip_pre = conversion.to_rainrate(Xt, metadata, zr_a=a_z, zr_b=b_z)[0]
        train_precip_pre[
            torch.isclose(torch.tensor(train_precip_pre), torch.tensor(0.0), atol=1e-04)
        ] = 0.0

        # Change to dB
        train_precip = transformation.dB_transform(train_precip_pre, metadata)[0]

        # Set zerovalue
        train_precip[~np.isfinite(train_precip)] = zero

        # Predict motion field
        motion_field = oflow_method(train_precip)

        # Apply advection/extrapolation
        # Predict array of nan when there are no previous times to use in computation
        try:
            # needs to be updated to get km and time from data
            precip_forecast_ens = extrapolate(
                train_precip[-3:, :, :],
                motion_field,
                n_leadtimes,
                n_ens_members=16,
                n_cascade_levels=8,
                precip_thr=10 * np.log10(thr),
                kmperpixel=km,
                timestep=t,
            )

            # Compute mean value from ensemble
            precip_forecast_mean = torch.nanmean(torch.from_numpy(precip_forecast_ens), dim=0)

            # Undo transformations made for STEPS model

            if data_type == "RADAR":
                precip_forecast = r_to_z(precip_forecast_mean)

            else:
                precip_forecast = 10 ** (precip_forecast_mean / 10)

        except ValueError:
            precip_forecast = torch.ones(yt.shape) * np.nan
        except np.linalg.LinAlgError:
            precip_forecast = torch.ones(yt.shape) * np.nan
        except RuntimeError:
            precip_forecast = torch.ones(yt.shape) * np.nan

    else:
        # Predict motion field
        motion_field = oflow_method(Xt)

        last_observation = Xt[-1]

        # Extrapolate
        last_observation[~np.isfinite(last_observation)] = 0
        precip_forecast = extrapolate(Xt[-1], motion_field, n_leadtimes)

    # Undo transformations on original data
    precip_forecast = np.array(precip_forecast).transpose((1, 2, 0))

    return precip_forecast


def main(args_dict):
    # Get necessary filepaths
    output_predict_filepath = args_dict["output_predict_filepaths"][0]
    dataframe_filepath = args_dict["dataframe_filepath"]
    locations = args_dict["locations"]

    output_predict_filepath = pathlib.Path(output_predict_filepath)
    if output_predict_filepath.is_file():
        if args_dict["overwrite"]:
            output_predict_filepath.unlink()
        else:
            raise FileExistsError(
                "Output file already exists. Pass 'overwrite' as true to overwrite."
            )

    data_type = args_dict["data_type"]

    # Load data
    if data_type == "RADAR":
        ds = HDFDataset2(
            dataframe_filepath,
            n_before=18,
            n_after=18,
            get_item_output=["X", "Y"],
            use_datetime_keys=True,
        )
    else:
        ds = HDFDatasetLocations(dataframe_filepath, locations, n_before=18, n_after=18)
    keys = ds.keys
    step = args_dict["num_workers"]
    size = len(keys)

    # Get indices to slice ds into chuncks
    chunk_indices = np.arange(0, size, step)
    if chunk_indices[-1] != size:
        chunk_indices = np.append(chunk_indices, [size])

    # Fix second entry of make_pred function to use parallelism
    task = partial(
        make_pred,
        hdf=ds,
        adv_scheme=args_dict["advection"],
        motion_f=args_dict["motion_field"],
        data_type=data_type,
    )

    # Run predictions for each chunk
    for i, j in zip(chunk_indices, chunk_indices[1:]):
        chunk_iterable = range(i, j)
        # print(chunk_iterable)
        predicts_list = Parallel(n_jobs=step)(delayed(task)(it) for it in chunk_iterable)

        # Concatenate predictions
        predict = np.stack(predicts_list, axis=0)

        # Create hdf
        array_to_pred_hdf(predict, ds.keys[i:j], ds.future_keys[i:j], output_predict_filepath)

    ok_message = "OK: Saved predictions successfully."
    print_ok(ok_message)
