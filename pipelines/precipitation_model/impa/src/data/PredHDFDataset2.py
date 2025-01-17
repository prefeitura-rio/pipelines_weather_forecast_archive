# -*- coding: utf-8 -*-
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
from pipelines.precipitation_model.impa.src.utils.general_utils import (
    print_ok,
    print_warning,
)
from pipelines.precipitation_model.impa.src.utils.hdf_utils import get_dataset_keys

MIN_WEIGHT = 100


class PredHDFDataset2(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Dataset filepath.
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(
        self,
        filepath,
        dataset,
        n_predictions,
        n_before=N_BEFORE,
        n_after=N_AFTER,
        x_transform=None,
        y_transform=None,
        get_item_output=["X", "Y", "latent_field", "motion_field", "intensities", "index"],
        autoencoder_hash=None,
        leadtime_conditioning=False,
        use_datetime_keys=False,
    ):
        self.x_transform = x_transform
        self.y_transform = y_transform
        # Search for all h5 files
        filepath = Path(filepath).resolve()
        shm_path = str(filepath).replace(str(filepath.parents[4]), "/dev/shm")
        if Path(shm_path).exists():
            # print_ok("File found in /dev/shm, using it.")
            self.filepath = Path(shm_path)
        else:
            # print_warning("File not found in /dev/shm, using original path.")
            self.filepath = filepath

        self.predict_filepath = Path(dataset)  # aqui

        self.n_predictions = n_predictions
        self.n_before = n_before
        self.n_after = n_after
        self.get_item_output = get_item_output

        self.leadtime_conditioning = leadtime_conditioning

        self._load_keys(use_datetime_keys=use_datetime_keys)

        if (
            len(
                set(["latent_field", "motion_field", "intensities"]).intersection(
                    set(get_item_output)
                )
            )
            > 0
        ):
            if autoencoder_hash is None:
                autoencoder_hash = "001178e117f50cf17817f336b86a809f"
            parent_path = filepath.parents[0]
            if "train" in filepath.stem:
                self.latent_field_filepath = Path(
                    f"{parent_path}/train_latent_{autoencoder_hash}.hdf"
                )
                self.motion_field_filepath = Path(f"{parent_path}/train_motion.hdf")
                self.intensities_filepath = Path(f"{parent_path}/train_intensities.hdf")
            elif "val" in filepath.stem:
                self.latent_field_filepath = Path(
                    f"{parent_path}/val_latent_{autoencoder_hash}.hdf"
                )
                self.motion_field_filepath = Path(f"{parent_path}/val_motion.hdf")
                self.intensities_filepath = Path(f"{parent_path}/val_intensities.hdf")
            else:
                raise ValueError("Dataset must be either train or val.")

            shm_latent_field_path = str(self.latent_field_filepath).replace(
                str(self.latent_field_filepath.parents[4]), "/dev/shm"
            )
            shm_motion_field_path = str(self.motion_field_filepath).replace(
                str(self.motion_field_filepath.parents[4]), "/dev/shm"
            )
            shm_intensities_path = str(self.intensities_filepath).replace(
                str(self.intensities_filepath.parents[4]), "/dev/shm"
            )
            if Path(shm_latent_field_path).exists():
                print_ok("Latent field file found in /dev/shm, using it.")
                self.latent_field_filepath = Path(shm_latent_field_path)
            else:
                print_warning("Latent field file not found in /dev/shm, using original path.")
            if Path(shm_motion_field_path).exists():
                print_ok("Motion field file found in /dev/shm, using it.")
                self.motion_field_filepath = Path(shm_motion_field_path)
            else:
                print_warning("Motion field file not found in /dev/shm, using original path.")
            if Path(shm_intensities_path).exists():
                print_ok("Intensities file found in /dev/shm, using it.")
                self.intensities_filepath = Path(shm_intensities_path)
            else:
                print_warning("Intensities file not found in /dev/shm, using original path.")

            with h5py.File(self.latent_field_filepath, "r") as hdf:
                self.latent_field_shape = hdf[self.keys[0]].shape

    def _load_keys(self, use_datetime_keys=False):
        with h5py.File(self.filepath) as hdf:
            what = hdf["what"]
            timestep = int(what.attrs["timestep"])
            self.ni = what.attrs["ni"]
            self.nj = what.attrs["nj"]
            try:
                self.subgrid_i1 = hdf["subgrid"].attrs["i1"]
                self.subgrid_i2 = hdf["subgrid"].attrs["i2"]
                self.subgrid_j1 = hdf["subgrid"].attrs["j1"]
                self.subgrid_j2 = hdf["subgrid"].attrs["j2"]
            except KeyError:
                pass
            if use_datetime_keys:
                split_keys = get_dataset_keys(hdf)
                self.keys = [
                    key.decode("utf-8")
                    for key in hdf["what/datetime_keys"]
                    if key.decode("utf-8") in split_keys
                ]
                self.past_keys = fetch_reversed_past_datetimes(self.keys, self.n_before, timestep)
                self.future_keys = fetch_future_datetimes(self.keys, self.n_after, timestep)
                self.pred_keys = fetch_pred_keys(self.keys, self.n_predictions, timestep)
            else:
                total_keys = get_dataset_keys(hdf)
                self.past_keys = fetch_reversed_past_datetimes(total_keys, self.n_before, timestep)
                if "latent_field" in self.get_item_output and self.n_before < N_BEFORE:
                    past_keys_temp = fetch_reversed_past_datetimes(total_keys, N_BEFORE, timestep)
                else:
                    past_keys_temp = self.past_keys
                self.pred_keys = fetch_pred_keys(total_keys, self.n_predictions, timestep)

                self.future_keys = fetch_future_datetimes(total_keys, self.n_after, timestep)
                valid_key_indices = []
                for i in range(len(total_keys)):
                    valid = True
                    # Accept missing past keys but not future keys
                    missing_past_keys = 0
                    for key in past_keys_temp[i]:
                        if key not in total_keys:
                            missing_past_keys += 1
                    if missing_past_keys > self.n_before // 3:
                        valid = False
                    for key in self.future_keys[i]:
                        if key not in total_keys:
                            valid = False
                            break
                    if valid:
                        valid_key_indices.append(i)
                self.keys = [total_keys[i] for i in valid_key_indices]
                self.past_keys = [self.past_keys[i] for i in valid_key_indices]
                self.future_keys = [self.future_keys[i] for i in valid_key_indices]
                self.pred_keys = [self.pred_keys[i] for i in valid_key_indices]

            if "train_log_mean" in what.attrs.keys() and "train_log_std" in what.attrs.keys():
                self.train_log_mean = what.attrs["train_log_mean"]
                self.train_log_std = what.attrs["train_log_std"]
            if "train_mean" in what.attrs.keys() and "train_std" in what.attrs.keys():
                self.train_mean = what.attrs["train_mean"]
                self.train_std = what.attrs["train_std"]

    # flake8: noqa: C901
    def __getitem__(self, index, get_item_output=None):
        if self.leadtime_conditioning:
            leadtime_index = index % self.n_after
            index = index // self.n_after
        if get_item_output is None:
            get_item_output = self.get_item_output
        with h5py.File(self.filepath, "r") as hdf:
            if "X" in get_item_output:
                X = torch.empty((self.ni, self.nj, self.n_before + self.n_predictions))
                for i, key in enumerate(self.past_keys[index]):
                    try:
                        X[:, :, self.n_before - i - 1] = torch.as_tensor(np.array(hdf[key]))
                    except KeyError:
                        X[:, :, self.n_before - i - 1] = torch.ones((self.ni, self.nj)) * np.nan
            if "Y" in get_item_output:
                if self.leadtime_conditioning:
                    key = self.future_keys[index][leadtime_index]
                    if leadtime_index == 0:
                        last_key = self.past_keys[index][-1]
                    else:
                        last_key = self.future_keys[index][leadtime_index - 1]
                    try:
                        Y = torch.empty((self.ni, self.nj, 2))
                        Y[:, :, 0] = torch.as_tensor(np.array(hdf[last_key]))
                        Y[:, :, 1] = torch.as_tensor(np.array(hdf[key]))
                    except KeyError:
                        Y = torch.ones((self.ni, self.nj, 2)) * np.nan
                else:
                    Y = torch.empty((self.ni, self.nj, self.n_after))
                    for i, key in enumerate(self.future_keys[index]):
                        try:
                            Y[:, :, i] = torch.as_tensor(np.array(hdf[key]))
                        except KeyError:
                            Y[:, :, i] = torch.ones((self.ni, self.nj)) * np.nan
                if self.y_transform:
                    Y = self.y_transform(Y)
        if "X" in get_item_output:
            with h5py.File(self.predict_filepath, "r") as pred_hdf:
                for i, key in enumerate(self.pred_keys[index]):
                    try:
                        X[:, :, self.n_before + i] = torch.as_tensor(np.array(pred_hdf[key]))
                    except KeyError:
                        raise KeyError("Prediction key not found.")
                        # X[:, :, self.n_before + i] = torch.ones((self.ni, self.nj)) * np.nan
            if self.x_transform:
                X = self.x_transform(X)

        future_keys = [self.past_keys[index][-1]] + list(self.future_keys[index][:-1])
        if "latent_field" in get_item_output:
            if self.leadtime_conditioning:
                key = future_keys[leadtime_index]
                with h5py.File(self.latent_field_filepath, "r") as hdf:
                    try:
                        latent_field = torch.as_tensor(np.array(hdf[key]))
                    except KeyError:
                        raise KeyError(f"Key {key} not found in latent field file.")
            else:
                latent_field = torch.empty((*self.latent_field_shape, self.n_after))
                with h5py.File(self.latent_field_filepath, "r") as hdf:
                    for i, key in enumerate(future_keys):
                        try:
                            latent_field[:, :, :, i] = torch.as_tensor(np.array(hdf[key]))
                        except KeyError:
                            raise KeyError(f"Key {key} not found in latent field file.")
                            # latent_field[:, :, i] = torch.ones((self.ni, self.nj)) * np.nan
        if "motion_field" in get_item_output:
            if self.leadtime_conditioning:
                key = future_keys[leadtime_index]
                with h5py.File(self.motion_field_filepath, "r") as hdf:
                    try:
                        motion_field = torch.as_tensor(np.array(hdf[key]))
                    except KeyError:
                        raise KeyError(f"Key {key} not found in motion field file.")
            else:
                motion_field = torch.empty((2, self.ni, self.nj, self.n_after))
                with h5py.File(self.motion_field_filepath, "r") as hdf:
                    for i, key in enumerate(future_keys):
                        try:
                            motion_field[:, :, :, i] = torch.as_tensor(np.array(hdf[key]))
                        except KeyError:
                            raise KeyError(f"Key {key} not found in motion field file.")
                            # motion_field[:, :, :, i] = torch.ones((2, self.ni, self.nj)) * np.nan
        if "intensities" in get_item_output:
            if self.leadtime_conditioning:
                key = future_keys[leadtime_index]
                with h5py.File(self.intensities_filepath, "r") as hdf:
                    try:
                        intensities = torch.as_tensor(np.array(hdf[key])[0])
                    except KeyError:
                        raise KeyError(f"Key {key} not found in intensities file.")
            else:
                intensities = torch.empty((self.ni, self.nj, self.n_after))
                with h5py.File(self.intensities_filepath, "r") as hdf:
                    for i, key in enumerate(future_keys):
                        try:
                            intensities[:, :, i] = torch.as_tensor(np.array(hdf[key]))
                        except KeyError:
                            raise KeyError(f"Key {key} not found in intensities file.")
                            # intensities[:, :, i] = torch.ones((self.ni, self.nj)) * np.nan

        output = ()
        for out in get_item_output:
            if out == "X":
                output += (X,)
            elif out == "Y":
                output += (Y,)
            elif out == "latent_field":
                output += (latent_field,)
            elif out == "motion_field":
                output += (motion_field,)
            elif out == "intensities":
                output += (intensities,)
            elif out == "index":
                output += (index,)
            else:
                raise ValueError(f"Output {out} not recognized.")
        if self.leadtime_conditioning:
            return output + (leadtime_index,)
        return output

    def __len__(self):
        return len(self.keys)

    def get_sample_weights(self, overwrite_if_exists=False, verbose=True):
        assert (
            self.filepath.stem == "train"
        ), "Sample weights can only be calculated for the train dataset."
        weights_filepath = (
            self.filepath.parents[0]
            / f"train_sample_weights2-n_before={self.n_before}-n_after={self.n_after}.npy"
        )
        if not overwrite_if_exists:
            if weights_filepath.is_file():
                print_ok("Loading existing sample weights.", verbose=verbose)
                weights = np.load(weights_filepath)
                if self.leadtime_conditioning:
                    return np.tile(weights, (self.n_after, 1)).T.flatten()
                return np.load(weights_filepath)
            print_warning("Sample weights file missing; calculating weights...", verbose=verbose)

        weights = []
        x_transform = self.x_transform
        y_transform = self.y_transform
        self.x_transform = None
        self.y_transform = None
        import tqdm
        from joblib import Parallel, delayed

        def task(i):
            X, Y = self.__getitem__(i, get_item_output=["X", "Y"])
            X = X[:, :, :, : self.n_before]
            weight = torch.sum(1 - torch.exp(-torch.nan_to_num(X)))
            weight += torch.sum(1 - torch.exp(-torch.nan_to_num(Y)))
            weight += MIN_WEIGHT
            return weight

        import os

        cpu_count = os.cpu_count()
        weights = Parallel(n_jobs=cpu_count)(delayed(task)(i) for i in tqdm.tqdm(range(len(self))))

        self.x_transform = x_transform
        self.y_transform = y_transform
        weights = np.array(weights)
        weights = weights / weights.sum()
        np.save(weights_filepath, weights)
        print_ok("Saved sample weights.", verbose=verbose)
        if self.leadtime_conditioning:
            return np.tile(weights, (self.n_after, 1)).T.flatten()
        return weights
