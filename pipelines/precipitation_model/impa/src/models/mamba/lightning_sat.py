# -*- coding: utf-8 -*-
# flake8: noqa: E501

import torch
from einops import rearrange
from pytorch_lightning import LightningModule

from src.models.mamba.vmamba_sat import VSSM
from src.utils.data_utils import data_modification_options


class Vmamba_lightning(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        context: torch.Tensor = None,
        truth: torch.Tensor = None,
        elev: torch.Tensor = None,
        context_val: torch.Tensor = None,
        truth_val: torch.Tensor = None,
        elev_val: torch.Tensor = None,
        x_mean: float = 0,
        x_std: float = 1,
        data_modification=None,
        normalized: int = 0,
        std_fac: float = 10.0,
        n_before: int = 18,
        n_after: int = 18,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.flip = False
        if context is not None:
            self.fixed_context = context[None, :, :, :]
        if truth is not None:
            self.ground_truth = truth[None, :, :, :]
        if context_val is not None:
            self.fixed_context_val = context_val[None, :, :, :]
        if truth_val is not None:
            self.ground_truth_val = truth_val[None, :, :, :]
        if elev is not None:
            self.elev = elev[None, :, :, :]
        if elev_val is not None:
            self.elev_val = elev_val[None, :, :, :]

        self.data_modification = data_modification
        self.dm_option = data_modification_options
        if data_modification is not None:
            for _, value in enumerate(self.data_modification):
                self.dm_option[value] = True

        self.normalized = normalized
        self.std_fac = std_fac
        self.show_im = 1
        self.x_mean = x_mean
        self.x_std = x_std

        self.automatic_optimization = True
        self.val_loss_steps = []
        self.b1 = b1
        self.b2 = b2
        self.learning_rate = learning_rate
        self.type_loss = "l1"
        self.lr = learning_rate

        self.batch_train = None
        self.batch_val = None
        self.cond = True
        self.n_after = n_after
        self.n_before = n_before

        self.use_elev = self.dm_option["Elevation"]
        self.use_date = self.dm_option["Hour_data"]
        self.use_latlon = self.dm_option["Lat_lon"]
        self.use_leadtime = self.dm_option["Lead_time_cond"]
        self.no_context_train = self.dm_option["No_context"]
        self.predict_context = self.dm_option["Pred_context"]

        self.model = VSSM(
            in_chans=n_before + n_before * (1 - self.no_context_train),
            out_chans=n_after + self.predict_context * n_after,
            patch_size=4,
        )

    def process_input(self, batch):
        x, y, metadata = batch
        x_small = x[:, : 2 * self.n_before : 2]
        x_large = x[:, 1 : 2 * self.n_before + 1 : 2]
        x = torch.cat([x_small, x_large], dim=1)

        if self.predict_context:
            y = rearrange(y, "b h w c -> b c h w")
        else:
            y = rearrange(y, "b h w c -> b c h w")
            y = y[:, : 2 * self.n_after : 2]

        metadata = rearrange(metadata, "b h w r -> b r h w")
        elev, latlon, date = metadata[:, :2], metadata[:, 2:6], metadata[:, 6:]

        if self.no_context_train:
            x = x[:, : self.n_before]

        if self.use_elev:
            x = torch.cat((x, elev), dim=1)
            y = torch.cat((y, elev), dim=1)

        if self.use_date:
            x = torch.cat((x, date), dim=1)
            y = torch.cat((y, elev), dim=1)

        if self.use_latlon:
            x = torch.cat((x, latlon), dim=1)
            y = torch.cat((y, elev), dim=1)
        return x, y

    def predict_step(self, batch, batch_idx):
        x_before, _ = self.process_input(batch)
        predict = self.model(x_before)[:, :18]
        return predict
