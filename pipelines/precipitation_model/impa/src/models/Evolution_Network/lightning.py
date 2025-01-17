# -*- coding: utf-8 -*-
# flake8: noqa: E501

import torch

from pipelines.precipitation_model.impa.src.models.Evolution_Network.evolution_network import (
    Evolution_Encoder_Decoder,
)
from pipelines.precipitation_model.impa.src.models.Evolution_Network.utils import (
    make_grid,
    warp,
)
from pipelines.precipitation_model.impa.src.models.lightning_module import LModule


class model(LModule):
    def __init__(
        self,
        learning_rate: float = 0.0001,
        context: torch.Tensor = None,
        truth: torch.Tensor = None,
        context_val: torch.Tensor = None,
        truth_val: torch.Tensor = None,
        n_before: int = 9,
        n_after: int = 20,
        normalized: int = 0,
        loss: int = 1,
        lambda_: float = 1e-2,
        max_clip: float = 3 * 6 / 5,
        x_mean: float = 0,
        x_std: float = 1,
        amplification: float = 1,
        n_epochs: int = 100,
        merge: bool = False,
        satellite: bool = False,
        **kwargs,
    ):
        super(model, self).__init__(
            truth=truth,
            context=context,
            truth_val=truth_val,
            context_val=context_val,
            n_before=n_before,
            n_after=n_after,
            normalized=normalized,
            loss=loss,
            xmax=max_clip,
            x_mean=x_mean,
            x_std=x_std,
            merge=merge,
            satellite=satellite,
        )

        self.lr = self.hparams.learning_rate
        self.lambda_ = self.hparams.lambda_
        self.n_epochs = self.hparams.n_epochs

        self.g1 = torch.broadcast_to(
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32),
            (self.n_after, 1, 3, 3),
        )

        self.g2 = torch.broadcast_to(
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32),
            (self.n_after, 1, 3, 3),
        )

        self.evo_net = Evolution_Encoder_Decoder(self.channels_in, self.n_after, base_c=32)

        x_dim = self.ground_truth.shape[-2]

        sample_tensor = torch.zeros(1, 1, x_dim, x_dim)
        self.grid = make_grid(sample_tensor)

    def forward(self, x):
        intensity, motion = self.evo_net(torch.flip(x, dims=[1]))
        motion_ = motion.reshape(x.shape[0], self.n_after, 2, x.shape[2], x.shape[3])
        intensity_ = intensity.reshape(x.shape[0], self.n_after, 1, x.shape[2], x.shape[3])
        series = []
        x_bili = []
        last_frames = x[:, 0:1, :, :].detach()
        grid = self.grid.repeat(x.shape[0], 1, 1, 1)
        for i in range(self.n_after):
            x_bili.append(
                warp(
                    last_frames,
                    motion_[:, i],
                    grid.to(self.device),
                    padding_mode="border",
                )
            )
            with torch.no_grad():
                last_frames = warp(
                    last_frames,
                    motion_[:, i],
                    grid.to(self.device),
                    mode="nearest",
                    padding_mode="border",
                )
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
            last_frames = last_frames.detach()
        evo_result = torch.cat(series, dim=1)
        bili_results = torch.cat(x_bili, dim=1)

        return evo_result, bili_results, motion_

    def predict_step(self, batch, batch_idx):
        batch = self.obtain_xand_y(batch)
        x = batch[0]
        # intensity, motion = self.evo_net(torch.flip(x, dims=[1]))

        # motion_ = motion.reshape(x.shape[0], self.n_after, 2, x.shape[2], x.shape[3])
        # intensity_ = intensity.reshape(x.shape[0], self.n_after, 1, x.shape[2], x.shape[3])

        y_hat = self.forward(x)

        if self.dm_option["Lead_time_cond"]:
            y_hat = y_hat[:, 0, :, :]

        if type(y_hat) is tuple:
            y_hat = y_hat[0]

        if self.merge and self.normalized == 3:
            y_hat = torch.expm1(y_hat)

        # return torch.concat([intensity_, motion_], dim=2)
        return y_hat
