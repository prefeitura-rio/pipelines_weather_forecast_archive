# -*- coding: utf-8 -*-
# flake8: noqa: E501

import torch

from pipelines.precipitation_model.impa.src.models.gan.gan_unet.model import (
    NowcasnetGenerator,
    TemporalDiscriminator,
)
from pipelines.precipitation_model.impa.src.models.lightning_module import LModule


# define the LightningModule
class model(LModule):
    def __init__(
        self,
        model_name: str = "NowcastNet",
        latent_dim: int = 32,
        discriminator_learning_rate: float = 0.0002,
        generator_learning_rate: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        n_after: int = 18,
        n_before: int = 4,
        context: torch.Tensor = None,
        truth: torch.Tensor = None,
        context_val: torch.Tensor = None,
        truth_val: torch.Tensor = None,
        satellite: bool = False,
        xmax: float = 0,
        x_mean: float = 0,
        x_std: float = 1,
        alpha: float = 6,
        beta: float = 20,
        generation_steps: int = 1,
        amplification: float = 1,
        weights: int = 1,
        needs_prediction: bool = True,
        **kwargs,
    ):
        super().__init__(
            truth=truth,
            context=context,
            truth_val=truth_val,
            context_val=context_val,
            n_before=n_before,
            n_after=n_after,
            normalized=2,
            loss=1,
            xmax=xmax,
            x_mean=x_mean,
            x_std=x_std,
            needs_prediction=needs_prediction,
            satellite=satellite,
        )

        self.save_hyperparameters()

        self.automatic_optimization = False
        if self.old:
            self.h_dim_noise = 7 if satellite else 8
        else:
            self.h_dim_noise = 8

        # If we use predictions or not
        self.predictions = self.hparams.predictions
        self.dlr = self.hparams.discriminator_learning_rate
        self.glr = self.hparams.generator_learning_rate
        self.satellite = self.hparams.satellite
        self.lr = (self.dlr + self.glr) / 2
        self.alpha = self.hparams.alpha
        self.beta = self.hparams.beta
        self.generation_steps = self.hparams.generation_steps
        self.amplification = self.hparams.amplification
        self.weights = self.hparams.weights

        old_sat = self.old and self.satellite

        self.discriminator = TemporalDiscriminator(
            in_channel=self.n_before + self.n_after, sat=self.satellite, old=self.old
        )

        self.generator = NowcasnetGenerator(
            channel_in=self.n_before + self.n_after,
            latent_dim=self.hparams.latent_dim,
            n_after=self.n_after,
            sat=old_sat,
        )

    def forward(self, x):
        # sample noise, is just the prediction.
        z = torch.randn(x.shape[0], self.hparams.latent_dim, self.h_dim_noise, self.h_dim_noise)
        z = z.type_as(x)
        return self.generator(x, z)

    def predict_step(self, batch, batch_idx):
        assert not self.dm_option["Lead_time_cond"]
        batch = self.obtain_xand_y(batch)
        x_before = batch[0]
        predictions = [self.forward(x_before) for _ in range(self.generation_steps)]
        generated_samples = torch.stack(predictions, dim=0)

        x_pred = torch.mean(generated_samples, dim=0)

        return x_pred
