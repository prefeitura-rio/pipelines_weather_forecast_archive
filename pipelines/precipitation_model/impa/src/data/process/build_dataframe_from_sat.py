import argparse
import datetime
import glob
import multiprocessing
import pathlib
from functools import partial

import h5py
import numpy as np
from tqdm import tqdm

from src.data.process.SatelliteData import SatelliteData
from src.utils.general_utils import print_warning

HEAVY_RAIN_TRAIN_LOG_MEAN = 0.15839338
HEAVY_RAIN_TRAIN_LOG_STD = 0.5295107

HEAVY_RAIN_TRAIN_MEAN = 0.5742944
HEAVY_RAIN_TRAIN_STD = 2.8702092


def task_dt(
    dt,
    ni: int,
    nj: int,
    sat_df: SatelliteData,
    grid_small: np.ndarray,
    grid_large: np.ndarray,
    value: str,
    band: str,
):
    datetime_key = (dt).strftime("%Y%m%d/%H%M")

    sat_df = sat_df.correct_parallax()
    data_small = sat_df.interp_at_grid(band, dt + datetime.timedelta(minutes=5), grid_small)
    data_large = sat_df.interp_at_grid(band, dt + datetime.timedelta(minutes=5), grid_large)
    assert data_small.shape == (ni, nj)
    assert data_large.shape == (ni, nj)
    data = np.dstack([data_small, data_large])
    return (datetime_key, data)


def build_dataframe_from_sat(
    datetimes,
    verbose=False,
    overwrite=False,
    product="ABI-L2-RRQPEF",
    num_workers=1,
    location="rio_de_janeiro",
    timestep=10,
    value="RRQPE",
    band="RRQPE",
):

    assert location in ["rio_de_janeiro", "curitiba", "porto_alegre", "salvador", "sao_luis"]

    sat_folder = f"data/processed/satellite/{product}"
    sat_df = SatelliteData.load_data(f"{sat_folder}/SAT-real_time.feather", value)

    output_filename = "real_time-" + location

    output_filepath = pathlib.Path(
        f"data/dataframes/SAT-CORRECTED-{product}-{pathlib.Path(output_filename).stem}/test.hdf"
    )

    pathlib.Path(output_filepath).parents[0].mkdir(parents=True, exist_ok=True)

    if output_filepath.is_file():
        if overwrite:
            print_warning(f"Warning: overwriting existing file {output_filepath}", verbose=verbose)
            output_filepath.unlink()
        else:
            print_warning(
                f"Warning: {output_filepath} already exists. Call with -o option to overwrite.",
                verbose=verbose,
            )
            exit(0)

    grid_small = np.load(f"data/dataframe_grids/{location}-res=2km-256x256.npy")
    grid_large = np.load(f"data/dataframe_grids/{location}-res=4km-256x256.npy")
    assert grid_small.shape == grid_large.shape
    ni, nj = grid_small.shape[:2]
    with h5py.File(output_filepath, "w") as f:
        what = f.create_group("what")
        what.attrs["feature"] = product
        what.attrs["process_type"] = "-"
        what.attrs["ni"] = ni
        what.attrs["nj"] = nj
        what.attrs["timestep"] = timestep
        what.attrs["train_log_mean"] = HEAVY_RAIN_TRAIN_LOG_MEAN
        what.attrs["train_log_std"] = HEAVY_RAIN_TRAIN_LOG_STD
        what.attrs["train_mean"] = HEAVY_RAIN_TRAIN_MEAN
        what.attrs["train_std"] = HEAVY_RAIN_TRAIN_STD

        # these are the datetimes which we want to predict from
        relevant_datetimes = [datetimes[18], datetimes[0]]
        what.create_dataset(
            "datetime_keys",
            data=np.asarray([dt.strftime("%Y%m%d/%H%M") for dt in relevant_datetimes], dtype="S"),
        )

        step = num_workers
        size = len(datetimes)

        # Get indices to slice ds into chuncks
        chunk_indices = np.arange(0, size, step)
        if chunk_indices[-1] != size:
            chunk_indices = np.append(chunk_indices, [size])

        task_dt_partial = partial(
            task_dt,
            ni=ni,
            nj=nj,
            sat_df=sat_df,
            grid_small=grid_small,
            grid_large=grid_large,
            value=value,
            band=band,
        )

        for i, j in tqdm(zip(chunk_indices, chunk_indices[1:]), total=len(chunk_indices)):
            chunk_iterable = datetimes[i:j]
            with multiprocessing.Pool(num_workers) as pool:
                datasets = list(pool.map(task_dt_partial, chunk_iterable))
            for datetime_key, data in datasets:
                f.create_dataset(datetime_key, data=data.astype(np.float32), compression="lzf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="If true, prints more information."
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="If true, overwrites output; otherwise, skips existing files.",
    )
    parser.add_argument(
        "--product",
        "-pr",
        default="ABI-L2-RRQPEF",
        help="Satellite product to be processed.",
        type=str,
    )
    parser.add_argument(
        "--num_workers",
        "-n",
        type=int,
        default=1,
        help="Number of processes for parallelization.",
    )
    parser.add_argument(
        "--location",
        "-loc",
        default="rio_de_janeiro",
        type=str,
        help="Location to build dataframe.",
    )
    parser.add_argument("--timestep", "-ts", default=10, type=int, help="Timestep in minutes.")
    parser.add_argument("--value", "-val", default="RRQPE", type=str, help="Satellite value.")
    parser.add_argument("--band", "-b", default="RRQPE", type=str, help="Satellite band.")
    parser.add_argument("--dt", "-dt", default=None, type=str, help="Datetime to predict from.")

    args = parser.parse_args()
    build_dataframe_from_sat(**vars(args))
