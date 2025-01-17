# -*- coding: utf-8 -*-
# flake8: noqa: E501

import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils import data

from pipelines.precipitation_model.impa.src.utils.dataframe_utils import (
    N_AFTER,
    N_BEFORE,
    fetch_future_datetimes,
    fetch_pred_keys,
    fetch_reversed_past_datetimes,
)
from pipelines.precipitation_model.impa.src.utils.hdf_utils import get_dataset_keys

MIN_WEIGHT = 100

elevation_file_small = "pipelines/precipitation_model/impa/data/processed/elevations_data/elevation_{location}-res=2km-256x256.npy"
elevation_file_large = "pipelines/precipitation_model/impa/data/processed/elevations_data/elevation_{location}-res=4km-256x256.npy"

latlon_file_small = "pipelines/precipitation_model/impa/data/dataframe_grids/{location}-res=2km-256x256.npy"
latlon_file_large = "pipelines/precipitation_model/impa/data/dataframe_grids/{location}-res=4km-256x256.npy"


class PredHDFDatasetLocations(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Dataset filepath.
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(
        self,
        dataframe,
        locations,
        dataset,
        n_predictions,
        n_before=N_BEFORE,
        n_after=N_AFTER,
        x_transform=None,
        y_transform=None,
        leadtime_conditioning=False,
    ):
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.leadtime_conditioning = leadtime_conditioning

        self.filepaths = []
        self.predict_filepaths = []
        for location in locations:
            path = Path(dataframe.format(location=location)).resolve()
            predict_filepath = Path(dataset)

            self.n_predictions = n_predictions
            shm_path = str(path).replace(str(path.parents[4]), "/dev/shm")
            if Path(shm_path).exists():
                # print_ok("File found in /dev/shm, using it.")
                filepath = shm_path
            else:
                # print_warning("File not found in /dev/shm, using original path.")
                filepath = path
            subprocess.run(f"cat {filepath} > /dev/null", shell=True)
            subprocess.run(f"cat {predict_filepath} > /dev/null", shell=True)
            self.filepaths.append(filepath)
            self.predict_filepaths.append(predict_filepath)

        self.n_before = n_before
        self.n_after = n_after
        self.elevation_tensors = []
        self.latlon_tensors = []
        for location in locations:
            el1 = torch.tensor(np.load(elevation_file_small.format(location=location)))
            el2 = torch.tensor(np.load(elevation_file_large.format(location=location)))
            el = torch.stack((el1, el2), dim=-1).type(torch.float32)
            self.elevation_tensors.append(el)

            latlon1 = torch.tensor(np.load(latlon_file_small.format(location=location)))
            latlon2 = torch.tensor(np.load(latlon_file_large.format(location=location)))
            latlon = torch.stack((latlon1, latlon2), dim=-1).type(torch.float32)
            self.latlon_tensors.append(latlon)
        self._load_keys()

    def _load_keys(self):
        self.ni = None
        self.nj = None
        timestep = None
        self.subgrid_i1 = None
        self.subgrid_i2 = None
        self.subgrid_j1 = None
        self.subgrid_j2 = None

        self.ds_indices = []
        self.keys = np.array([])
        self.past_keys = np.array([])
        self.future_keys = np.array([])
        self.pred_keys = np.array([])
        for i, filepath in enumerate(self.filepaths):
            with h5py.File(filepath) as hdf:
                what = hdf["what"]
                if i == 0:
                    timestep = int(what.attrs["timestep"])
                    self.ni = what.attrs["ni"]
                    self.nj = what.attrs["nj"]
                else:
                    assert timestep == int(what.attrs["timestep"])
                    assert self.ni == what.attrs["ni"]
                    assert self.nj == what.attrs["nj"]

                if "split_info/split_datetime_keys" in hdf:
                    new_keys = hdf["split_info"]["split_datetime_keys"]
                else:
                    new_keys = get_dataset_keys(hdf)

                self.keys = np.append(self.keys, new_keys)
                try:
                    self.past_keys = np.vstack(
                        [
                            self.past_keys,
                            np.array(
                                fetch_reversed_past_datetimes(new_keys, self.n_before, timestep)
                            ),
                        ]
                    )
                except ValueError:
                    self.past_keys = np.array(
                        fetch_reversed_past_datetimes(new_keys, self.n_before, timestep)
                    )
                try:
                    self.future_keys = np.vstack(
                        [
                            self.future_keys,
                            np.array(fetch_future_datetimes(new_keys, self.n_after, timestep)),
                        ]
                    )
                except ValueError:
                    self.future_keys = np.array(
                        fetch_future_datetimes(new_keys, self.n_after, timestep)
                    )
                try:
                    self.pred_keys = np.vstack(
                        [
                            self.pred_keys,
                            np.array(fetch_pred_keys(new_keys, self.n_predictions, timestep)),
                        ]
                    )
                except ValueError:
                    self.pred_keys = np.array(
                        fetch_pred_keys(new_keys, self.n_predictions, timestep)
                    )
                try:
                    self.ds_indices.append(len(new_keys) + self.ds_indices[-1])
                except IndexError:
                    self.ds_indices.append(len(new_keys))

    def _get_hdf_index(self, index):
        for i, ds_index in enumerate(self.ds_indices):
            if index < ds_index:
                return i

    def __getitem__(self, index):
        X = torch.empty((self.ni, self.nj, 2 * self.n_before + self.n_predictions))
        if self.leadtime_conditioning:
            pass
        else:
            Y = torch.empty((self.ni, self.nj, 2 * self.n_after))

        if self.leadtime_conditioning:
            leadtime_index = index % self.n_after
            index = index // self.n_after

        hdf_index = self._get_hdf_index(index)
        with h5py.File(self.filepaths[hdf_index], "r") as hdf:
            for i, key in enumerate(self.past_keys[index]):
                tensor_ind = 2 * self.n_before - 2 * i - 2
                try:
                    X[:, :, tensor_ind : tensor_ind + 2] = torch.as_tensor(
                        np.array(hdf[key])
                    ).reshape((self.ni, self.nj, 2))
                except KeyError:
                    X[:, :, tensor_ind : tensor_ind + 2] = (
                        torch.ones((self.ni, self.nj, 2)) * np.nan
                    )
            if self.leadtime_conditioning:
                try:
                    Y = torch.as_tensor(np.array(hdf[self.future_keys[index][leadtime_index]]))
                except KeyError:
                    Y = torch.ones((self.ni, self.nj, 2)) * np.nan
            else:
                for i, key in enumerate(self.future_keys[index]):
                    try:
                        Y[:, :, 2 * i : 2 * i + 2] = torch.as_tensor(np.array(hdf[key])).reshape(
                            (self.ni, self.nj, 2)
                        )
                    except KeyError:
                        Y[:, :, 2 * i : 2 * i + 2] = torch.ones((self.ni, self.nj, 2)) * np.nan
        with h5py.File(self.predict_filepaths[hdf_index], "r") as pred_hdf:
            for i, key in enumerate(self.pred_keys[index]):
                tensor_ind = 2 * self.n_before + i
                try:
                    X[:, :, tensor_ind] = torch.as_tensor(np.array(pred_hdf[key]))
                except KeyError:
                    raise KeyError("Prediction key not found.")
        if self.x_transform:
            X = self.x_transform(X)
        if self.y_transform:
            Y = self.y_transform(Y)

        ### Metadata
        date = self.past_keys[index][-1]
        # year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        hour = int(date[9:11])
        minute = int(date[11:13])
        date = torch.tensor(
            [month / 12, day / 31, hour / 24, minute / 60], dtype=torch.float32
        ).reshape((1, 1, -1))
        date_tensor = date.expand((self.ni, self.nj, -1))

        metadata_tensor = torch.cat(
            (
                self.elevation_tensors[hdf_index],
                self.latlon_tensors[hdf_index].reshape(self.ni, self.nj, -1),
                date_tensor,
            ),
            dim=-1,
        )
        if self.leadtime_conditioning:
            return (X, Y, metadata_tensor, leadtime_index)
        else:
            return (X, Y, metadata_tensor)

    def __len__(self):
        if self.leadtime_conditioning:
            return len(self.keys) * self.n_after
        else:
            return len(self.keys)

    def get_sample_weights(self):
        weights = None
        for filepath in self.filepaths:
            filepath = Path(filepath)
            stem = filepath.stem
            weights_filepath = filepath.parents[0] / f"{stem}_sample_weights.npy"
            inds_filepath = filepath.parents[0] / f"{stem}_split_datetime_inds.npy"
            pre_weights = np.load(weights_filepath)
            pre_weights_size = len(pre_weights)
            pre_weights = np.append(pre_weights, np.array([pre_weights.min()]))
            inds = np.load(inds_filepath).reshape(-1, 1)
            deltas = np.arange(-self.n_before + 1, self.n_after + 1, dtype=int).reshape(1, -1)
            slices = inds + deltas
            slices[np.logical_or(slices < 0, slices >= pre_weights_size)] = pre_weights_size
            summed_weights = pre_weights[slices].sum(axis=1).reshape(-1)

            if weights is None:
                weights = summed_weights
            else:
                weights = np.append(weights, summed_weights)
        if self.leadtime_conditioning:
            weights = np.tile(weights, (self.n_after, 1)).T.flatten()
        assert len(weights) == len(self)
        return weights / weights.sum()
