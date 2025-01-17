# -*- coding: utf-8 -*-
# flake8: noqa: E203

import torch
from einops import rearrange
from pytorch_lightning import LightningModule

from pipelines.precipitation_model.impa.src.utils.data_utils import (
    data_modification_options,
)


# define the LightningModule
class LModule(LightningModule):
    def __init__(
        self,
        truth: torch.Tensor = None,
        context: tuple = None,
        truth_val: torch.Tensor = None,
        context_val: tuple = None,
        n_before: int = 9,
        n_after: int = 20,
        normalized: int = 0,
        loss: int = 1,
        xmax: float = 3 * 20 / 5,
        x_mean: float = 0,
        x_std: float = 1,
        amplification: float = 1,
        data_modification: list = None,
        merge: bool = False,
        needs_prediction: bool = False,
        satellite: bool = False,
    ):
        super().__init__()

        assert len(context) == len(context_val)

        self.save_hyperparameters()
        self.ground_truth = truth
        self.ground_truth_val = truth_val
        self.old = True
        self.n_before = self.hparams.n_before
        self.n_after = self.hparams.n_after
        self.normalized = self.hparams.normalized
        self.type_loss = self.hparams.loss
        self.weighted = self.hparams.weights
        self.data_modification = self.hparams.data_modification
        self.needs_prediction = needs_prediction
        self.merge = merge
        print("Satellite: ", satellite)
        self.sat = satellite
        self.dm_option = data_modification_options
        if self.data_modification is not None:
            for i, value in enumerate(self.data_modification):
                self.dm_option[value] = True
        self.meta_data_ind = [
            self.dm_option["Elevation"],
            self.dm_option["Elevation"],
            self.dm_option["Lat_lon"],
            self.dm_option["Lat_lon"],
            self.dm_option["Lat_lon"],
            self.dm_option["Lat_lon"],
            self.dm_option["Hour_data"],
            self.dm_option["Hour_data"],
            self.dm_option["Hour_data"],
            self.dm_option["Hour_data"],
        ]

        if not self.merge:
            assert not self.dm_option["No_satellite"]

        if len(context) == 2:
            self.channels_in = (
                2 * self.n_before
                + 2 * self.dm_option["Elevation"]
                - self.n_before * self.dm_option["No_context"]
                + 4 * self.dm_option["Lat_lon"]
                + 4 * self.dm_option["Hour_data"]
                + self.dm_option["Add_lead_to_input"]
                + self.n_before * self.merge
                - self.n_before * 2 * self.dm_option["No_satellite"]
            )

            self.old = False
        elif len(context) == 1:
            self.channels_in = self.n_before

        print("Old_data: ", self.old)
        self.channels_out = (
            1
            if self.dm_option["Lead_time_cond"]
            else self.n_after * (self.dm_option["Pred_context"] + 1)
        )

    def obtain_xand_y(self, batch):
        try:
            self.merge
        except AttributeError:
            self.merge = False

        try:
            self.needs_prediction
        except AttributeError:
            self.needs_prediction = False

        if not self.old or self.merge:
            if self.dm_option["Lead_time_cond"]:
                x, y, elev, lead_time = batch
            else:
                x, y, elev = batch
                lead_time = None

            # Use only the desire metadata
            meta_data = elev[:, :, :, self.meta_data_ind]
            # The reorder was done by first the time of the normal and the of the context

            x = rearrange(x, "b h w c -> b c h w")

            if self.needs_prediction:
                x = torch.cat([x[:, : 2 * self.n_before : 2], x[:, -self.n_after :]], axis=1)

            if self.merge:
                # Transform the radar data with log1p
                x[:, : self.n_before] = torch.log1p(x[:, : self.n_before])
                if self.dm_option["No_satellite"]:
                    x = x[:, : self.n_before]
            if self.dm_option["No_context"]:
                x = x[:, : self.n_before]

            if self.dm_option["Add_lead_to_input"]:
                x = torch.cat(
                    (
                        x,
                        lead_time.reshape((x.shape[0], 1, 1, 1)).expand(
                            x.shape[0], 1, x.shape[2], x.shape[3]
                        ),
                    ),
                    dim=1,
                )
                lead_time = None

            if meta_data.shape[3] > 0:
                meta_data = rearrange(meta_data, "b h w r -> b r h w")
                x = torch.cat((x, meta_data), dim=1)

            # Rearrange the prediction
            y = rearrange(y, "b h w c -> b c h w")
            if not self.dm_option["Pred_context"]:
                if not self.dm_option["Lead_time_cond"]:
                    y = y[:, ::2]
                else:
                    y = y[:, 0:1]
                if self.merge:
                    y = torch.log1p(y)
            else:
                assert not self.merge
        else:
            # the transformation must have already being done with the normaization
            x, y = batch[0], batch[1]
            if self.dm_option["Lead_time_cond"]:
                lead_time = batch[-1]
            else:
                lead_time = None
        return x, y, lead_time

    def forward_lead_time_all(self, x):
        assert self.dm_option["Lead_time_cond"]
        out = torch.empty((x.shape[0], self.n_after, x.shape[-1], x.shape[-1]), device=self.device)
        for i in range(self.n_after):
            lead_time = torch.ones((x.shape[0]), device=self.device, dtype=torch.int) * int(i)
            out[:, i] = self.forward(x, lead_time)[:, 0]
        return out

    def predict_step(self, batch, batch_idx):
        x = self.obtain_xand_y(batch)[0]
        if self.dm_option["Lead_time_cond"] and not self.dm_option["Add_lead_to_input"]:
            y_hat = self.forward_lead_time_all(x)
        else:
            y_hat = self.forward(x)

        # if self.dm_option["Lead_time_cond"]:
        #     y_hat = y_hat[:, 0, :, :]

        if type(y_hat) is tuple:
            y_hat = y_hat[0]

        if self.merge and self.normalized == 3:
            y_hat = torch.expm1(y_hat)

        return y_hat
