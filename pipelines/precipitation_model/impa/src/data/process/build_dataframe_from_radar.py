import argparse
import datetime
import glob
import multiprocessing
import pathlib
from functools import partial

import h5py
import numpy as np
from tqdm import tqdm

from src.data.process.RadarData import RadarData
from src.utils.general_utils import print_warning


def task_dt(dt, ni: int, nj: int, radar_folder: str, grid: np.ndarray, feature: str):
    datetime_key = dt.strftime("%Y%m%d/%H%M")
    for i in range(5):
        dt_lag = dt - datetime.timedelta(minutes=i)
        datetime_str = dt_lag.strftime("%Y-%m-%d %H%M")
        date_str_sep, time_str = datetime_str.split(" ")
        date_str = date_str_sep.replace("-", "")

        file_list = list(
            glob.glob(
                f"{radar_folder}/{date_str_sep}/*-{feature}-{date_str}-{time_str}*.hdf",
                recursive=True,
            )  # aqui
        )
        if len(file_list) > 0:
            break

    if len(file_list) < 1:
        data = np.ones((ni, nj)) * np.nan
    else:
        file = file_list[0]
        radar_data = RadarData.load_hdf(file)
        if radar_data.compressed:
            data = radar_data.unwrap_data().interp_at_grid(grid)
        else:
            data = radar_data.interp_at_grid(grid)
        assert data.shape == (ni, nj)
    return (datetime_key, data)


def build_dataframe_from_radar(
    dt=None,
    verbose=False,
    overwrite=False,
    feature="DBZH",
    process_type="d2CMAX",
    num_workers=1,
    grid_file="rio_de_janeiro-res=700m-256x256.npy",
    timestep=10,
):
    datetimes = []

    radar_folder = f"data/processed/processed_PPI_MDN/{process_type}/{feature}"
    if dt is None:
        dt = datetime.datetime.now(datetime.timezone.utc)
    else:
        pass
    dt_rounded = datetime.datetime(
        dt.year, dt.month, dt.day, dt.hour, dt.minute - dt.minute % timestep
    )
    for i in range((60 * 6) // timestep):
        dt = dt_rounded - datetime.timedelta(minutes=i * timestep)
        datetimes.append(dt)

    datetimes = sorted(datetimes)

    output_filepath = pathlib.Path(
        f"pipelines/precipitation_model/impa/data/dataframes/MDN-{process_type}-{feature}-real_time/test.hdf"
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

    grid = np.load(f"data/dataframe_grids/{grid_file}")
    ni, nj = grid.shape[:2]
    with h5py.File(output_filepath, "w") as f:
        what = f.create_group("what")
        what.attrs["feature"] = feature
        what.attrs["process_type"] = process_type
        what.attrs["ni"] = ni
        what.attrs["nj"] = nj
        what.attrs["grid_file"] = grid_file
        what.attrs["timestep"] = timestep
        relevant_datetimes = [dt_rounded, dt_rounded - datetime.timedelta(hours=3)]
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
            radar_folder=radar_folder,
            grid=grid,
            feature=feature,
        )

        for i, j in tqdm(zip(chunk_indices, chunk_indices[1:]), total=len(chunk_indices)):
            chunk_iterable = datetimes[i:j]
            # with get_context("spawn").Pool(args_dict["n_jobs"]) as pool:
            with multiprocessing.Pool(num_workers) as pool:
                datasets = list(pool.map(task_dt_partial, chunk_iterable))
            for datetime_key, data in datasets:
                f.create_dataset(datetime_key, data=data.astype(np.float32), compression="lzf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output.",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="If true, overwrites output; otherwise, skips existing files.",
    )
    parser.add_argument(
        "--feature",
        "-f",
        default="DBZH",
        help="Feature to be processed (DBZH, DBZV, ...)",
        type=str,
    )
    parser.add_argument(
        "--process_type",
        "-t",
        default="CMAX",
        help="Type of processing (CMAX, PSEUDO-CAPPI, ...)",
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
        "--grid_file",
        "-gf",
        default="rio_de_janeiro-res=700m-256x256.npy",
        type=str,
        help="File containing grid.",
    )
    parser.add_argument("--timestep", "-ts", default=10, type=int, help="Timestep in minutes.")
    args = parser.parse_args()
    build_dataframe_from_radar(**vars(args))
