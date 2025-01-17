# -*- coding: utf-8 -*-
# flake8: noqa: E501

import copy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import interpolate

# MAP_CENTER = {"lat": -22.9932804107666, "lon": -43.26795928955078}
# LAT_LON_BOUNDS = {"lon_min": -44.4458, "lon_max": -42.5939, "lat_min": -23.3282, "lat_max": -22.4758}
# SCAN_TIME = 410
# KNOTS_TO_MPS = 463 / 900

SAT_LON = 75.2
SAT_LAT = 0.0
SAT_ALT = 35786200


class SatelliteData:
    def __init__(
        self,
        data: pd.DataFrame,
        product: list[str],
        value: str | None = None,
        file_stem: str | None = None,
        folder: str | None = None,
    ) -> None:
        self.data = data
        self.product = product
        self.value = value
        self.stem = file_stem
        self.folder = folder

    @classmethod
    def load_data(cls, input_filepath: Path | str, value: str | None = None):
        return cls(
            pd.read_feather(input_filepath),
            Path(input_filepath).parent.name,
            value,
            Path(input_filepath).stem,
            Path(input_filepath).parents[1],
        )

    def _load_cloud_height(self) -> pd.DataFrame:
        return pd.read_feather(f"{self.folder}/ABI-L2-ACHAF/{self.stem}.feather")

    def correct_parallax(self):
        from satpy.modifiers.parallax import get_parallax_corrected_lonlats

        df_height = self._load_cloud_height()
        timestamps = self.data.creation.unique()

        new_sd = copy.deepcopy(self)

        updated_lats = None
        updated_lons = None
        for timestamp in timestamps:
            filtered_df = self.data[self.data["creation"] == timestamp]
            lats = np.array(filtered_df.lat)
            lons = np.array(filtered_df.lon)

            closest_timestamp = df_height.loc[df_height["creation"] <= timestamp, "creation"].max()
            filtered_df_height = df_height[df_height["creation"] == closest_timestamp]
            height_lats = np.array(filtered_df_height.lat)
            height_lons = np.array(filtered_df_height.lon)
            heights = np.array(filtered_df_height.HT)
            points = np.stack((height_lons, height_lats)).T
            h_interp = interpolate.griddata(points, heights, (lons, lats), method="linear")

            new_lons, new_lats = get_parallax_corrected_lonlats(
                SAT_LON, SAT_LAT, SAT_ALT, lons, lats, h_interp
            )

            if updated_lats is None:
                updated_lats = new_lats
                updated_lons = new_lons
            else:
                updated_lats = np.concatenate((updated_lats, new_lats))
                updated_lons = np.concatenate((updated_lons, new_lons))
        new_sd.data.lat = updated_lats
        new_sd.data.lon = updated_lons
        return new_sd

    def interp_at_grid(self, band: str, timestamp: datetime, target_grid: NDArray):
        assert (
            timestamp >= self.data["creation"]
        ).any(), "Timestamp passed precedes all timestamps in the data"
        closest_timestamp = self.data.loc[self.data["creation"] <= timestamp, "creation"].max()
        df = self.data[self.data["creation"] == closest_timestamp]
        column = f"{self.value}_{band}" if self.product == "ABI-L2-MCMIPF" else self.value

        x = np.array(df["lon"])
        y = np.array(df["lat"])
        values = np.array(df[column])
        nan_indices = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~nan_indices]
        y = y[~nan_indices]
        values = values[~nan_indices]
        points = np.stack((x, y)).T
        shape = target_grid.shape[:2]
        interp_values = interpolate.griddata(
            points,
            values,
            (target_grid[:, :, 1].flatten(), target_grid[:, :, 0].flatten()),
            method="linear",
        ).reshape(shape)

        return interp_values
