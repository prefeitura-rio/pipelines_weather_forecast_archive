# -*- coding: utf-8 -*-
"""
Process satellite data
"""

# flake8: noqa: E501
# pylint: disable=invalid-name, line-too-long, too-many-locals, too-many-arguments

import gc
import os

# import os
# from argparse import ArgumentParser
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xarray as xr

# from joblib import Parallel, delayed  # pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.logging import log  # pylint: disable=E0611, E0401
from pyproj import Proj
from tqdm import tqdm  # pylint: disable=E0611, E0401


def process_file(
    file_path: str,
    bands: list[str],
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
    include_dataset_name: bool = False,
) -> pd.DataFrame:
    """Returns processed satellite data for desired region and bands.

    Conversion of coordinates to latitudes and longitudes based on [1].

    Args:
        file_path: path to netcdf file with satellite data
        bands: bands to be extracted from data
        lat_bounds: minimum and maximum latitudes to consider
        lon_bounds: minimum and maximum longitudes to consider

    References:
        [1] https://github.com/blaylockbk/pyBKB_v3/blob/master/BB_GOES/mapping_GOES16_TrueColor.ipynb
        [2] https://proj4.org/operations/projections/geos.html
    """

    # cpu_usage = psutil.cpu_percent(interval=1, percpu=True)  # Lista de uso de cada núcleo
    # cpu_usage_total = sum(cpu_usage) / len(cpu_usage)  # Média de uso de CPU (total)

    # # Uso total de memória do sistema
    # memory_info = psutil.virtual_memory()
    # total_memory = memory_info.total / (1024**3)  # Convertendo para GB
    # used_memory = memory_info.used / (1024**3)
    # free_memory = memory_info.available / (1024**3)

    # # Exibir os resultados
    # log(
    #     f"Uso total de CPU por núcleo (%): {cpu_usage}, Uso médio total de CPU (%): {cpu_usage_total:.2f}%"
    # )
    # log(
    #     f"Memória total: {total_memory:.2f} GB, Memória usada: {used_memory:.2f} GB, Memória livre: {free_memory:.2f} GB"
    # )

    # Read satellite data
    dataset = xr.open_dataset(file_path)

    # Retrieve datetimes of file creation and start and end of scan (UTC)
    # scan_start = datetime.strptime(dataset.time_coverage_start, "%Y-%m-%dT%H:%M:%S.%fZ")
    # scan_end = datetime.strptime(dataset.time_coverage_end, "%Y-%m-%dT%H:%M:%S.%fZ")
    creation = datetime.strptime(dataset.date_created, "%Y-%m-%dT%H:%M:%S.%fZ")

    if hasattr(dataset, "lon") and hasattr(dataset, "lat"):
        lons, lats = dataset["lon"], dataset["lat"]
    else:
        # Load satellite height, longitude and sweep
        sat_h = dataset["goes_imager_projection"].perspective_point_height
        sat_lon = dataset["goes_imager_projection"].longitude_of_projection_origin
        sat_sweep = dataset["goes_imager_projection"].sweep_angle_axis

        # Calculate projection x and y coordinates as the scanning angle (in radians)
        #   multiplied by the satellite height (cf. [2])
        x = dataset["x"][:] * sat_h
        y = dataset["y"][:] * sat_h

        # Create a pyproj geostationary map object
        p = Proj(proj="geos", h=sat_h, lon_0=sat_lon, sweep=sat_sweep)

        # Perform cartographic transformation, that is, convert
        #  image projection coordinates (x and y) to latitude and longitude values
        XX, YY = np.meshgrid(x, y)
        lons, lats = p(XX, YY, inverse=True)

    # Load data from bands of interest and append to latitude and longitude
    data = [lats, lons] + [dataset[band].data for band in bands]

    # Construct dataframe
    df = pd.DataFrame(
        np.array(data).astype(dtype=np.float32).reshape(len(data), -1).T,
        columns=["lat", "lon", *bands],
    )

    # Remove nan and infinite values
    df.replace(np.inf, np.nan, inplace=True)
    df.dropna(how="any", axis=0, inplace=True)

    # Discard observations for latitudes and longitudes outside bounds
    if lat_bounds:
        df.drop(df[(df["lat"] <= lat_bounds[0]) | (df["lat"] >= lat_bounds[1])].index, inplace=True)
    if lon_bounds:
        df.drop(df[(df["lon"] <= lon_bounds[0]) | (df["lon"] >= lon_bounds[1])].index, inplace=True)

    # Include datetimes of file creation and start and end of scan (UTC-3)
    # df["start"] = scan_start  # - timedelta(hours=3)
    # df["end"] = scan_end  # - timedelta(hours=3)
    df["creation"] = creation  # - timedelta(hours=3)

    if include_dataset_name:
        df["name"] = "_".join(dataset.dataset_name.split("_")[:2])

    return df


def load_entire_day(
    product, ts: pd.Timestamp, lat_bounds, lon_bounds, download_base_path
) -> pd.DataFrame:
    """Load and concatenate all files from that day"""
    year = ts.year
    day = ts.dayofyear

    if not Path(f"{download_base_path}/{product}/{year}/{day:03d}").exists():
        log(f"No files found for {product} {year} {day:03d}")
        return pd.DataFrame()

    match product:
        case "ABI-L2-MCMIPF":  # Cloud and Moisture Imagery
            bands = ["CMI_C08", "CMI_C09", "CMI_C10", "CMI_C11"]
            include_dataset_name = False
        case "ABI-L2-RRQPEF":  # Rainfall Rate (Quantitative Precipitation Estimate)
            bands = ["RRQPE"]
            include_dataset_name = False
        case "ABI-L2-DMWF":
            bands = [
                "wind_direction",
                "wind_speed",
                "temperature",
                "pressure",
            ]
            include_dataset_name = True
        case "ABI-L2-ACHAF":
            bands = ["HT"]
            include_dataset_name = False
        case _:
            raise ValueError("Unsupported product selected.")

    # Check if files exist inside path
    path_ = f"{download_base_path}/{product}/{year}/"

    all_files = []
    for root, dirs, files in os.walk(path_):
        for file in files:
            all_files.append(os.path.join(root, file))

    log(f"Files to be processed: {all_files[:5]} {len(all_files)}")

    # return pd.concat(
    #     Parallel(n_jobs=num_workers)(
    #         delayed(process_file)(file, bands, lat_bounds, lon_bounds, include_dataset_name)
    #         for file in glob(f"{download_base_path}/{product}/{year}/{day:03d}/*/*.nc")
    #     )
    # )
    dfr_list = []
    for file in glob(f"{download_base_path}/{product}/{year}/{day:03d}/*/*.nc"):
        df = process_file(file, bands, lat_bounds, lon_bounds, include_dataset_name)
        dfr_list.append(df)

    return pd.concat(dfr_list, ignore_index=True)


def process_satellite(
    product="ABI-L2-RRQPEF",
    lat_min=-26.0,
    lat_max=-19.0,
    lon_min=-47.0,
    lon_max=-40.0,
    num_workers=16,
    day=-1,
    year=-1,
    n_historical_days=1,
    download_base_path="pipelines/precipitation_model/impa/data/raw/satellite",
):
    """Empty"""
    log(f"Processing satellite {product}")

    lat_bounds = lat_min, lat_max
    lon_bounds = lon_min, lon_max

    end_date = datetime(year, 1, 1) + timedelta(day - 1)
    today_file = Path(
        f"pipelines/precipitation_model/impa/data/processed/satellite/{product}/{end_date.date()}.feather"
    )
    if today_file.is_file():
        # Do not process older dates
        start_date = end_date
    else:
        start_date = datetime(year, 1, 1) + timedelta(day - n_historical_days - 1)

    log(f"DEBUG start_date: {start_date}, end_date: {end_date}")
    log(f"Start loading entire day of start_date {start_date}")
    df_current = load_entire_day(
        product, pd.Timestamp(start_date), lat_bounds, lon_bounds, download_base_path
    )
    output_path = Path(f"pipelines/precipitation_model/impa/data/processed/satellite/{product}")
    output_path.mkdir(exist_ok=True, parents=True)

    for date in tqdm(
        pd.date_range(start_date, end_date),
        desc="Saving files",
    ):
        next_date = date + timedelta(days=1)
        try:
            df_next = load_entire_day(
                product, next_date, lat_bounds, lon_bounds, download_base_path
            )
            df_current = pd.concat([df_current, df_next])
        except ValueError:
            df_next = None
        df_current = df_current[df_current["creation"].dt.date == date.date()]
        df_current.reset_index(drop=True, inplace=True)
        df_current.to_feather(f"{output_path}/{date.date()}.feather")
        del df_current
        try:
            df_current = df_next.copy()
        except AttributeError:
            pass
        del df_next
        gc.collect()


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--product", type=str, default="ABI-L2-RRQPEF")
#     parser.add_argument("--lat_min", type=float, default=-26.0)
#     parser.add_argument("--lat_max", type=float, default=-19.0)
#     parser.add_argument("--lon_min", type=float, default=-47.0)
#     parser.add_argument("--lon_max", type=float, default=-40.0)
#     parser.add_argument("--num_workers", type=int, default=16)
#     parser.add_argument("--day", type=int, default=-1)
#     parser.add_argument("--year", type=int, default=-1)
#     args = parser.parse_args()

#     process_satellite(**vars(args))
