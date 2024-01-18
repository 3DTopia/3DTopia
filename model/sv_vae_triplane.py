import os
import imageio
import numpy as np
import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import wandb

import lpips
from pytorch_msssim import SSIM

from utility.initialize import instantiate_from_config

class VAE(pl.LightningModule):
    def __init__(self, vae_configs, renderer_configs, lr=1e-3, weight_decay=1e-2,
                kld_weight=1, mse_weight=1, lpips_weight=0.1, ssim_weight=0.1,
                log_image_freq=50):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.kld_weight = kld_weight
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.ssim_weight = ssim_weight
        self.log_image_freq = log_image_freq

        self.vae = instantiate_from_config(vae_configs)
        self.renderer = instantiate_from_config(renderer_configs)

        self.lpips_fn = lpips.LPIPS(net='alex')
        self.ssim_fn = SSIM(data_range=1, size_average=True, channel=3)

        self.triplane_render_kwargs = {
            'depth_resolution': 64,
            'disparity_space_sampling': False,
            'box_warp': 2.4,
            'depth_resolution_importance': 64,
            'clamp_mode': 'softplus',
            'white_back': True,
        }

    def forward(self, batch, is_train):
        encoder_img, input_img, input_ray_o, input_ray_d, \
        target_img, target_ray_o, target_ray_d = batch
        grid, mu, logvar = self.vae(encoder_img, is_train)

        cat_ray_o = torch.cat([input_ray_o, target_ray_o], 0)
        cat_ray_d = torch.cat([input_ray_d, target_ray_d], 0)
        render_out = self.renderer(torch.cat([grid, grid], 0), cat_ray_o, cat_ray_d, self.triplane_render_kwargs)
        render_gt = torch.cat([input_img, target_img], 0)

        return render_out['rgb_marched'], render_out['depth_final'], \
               render_out['weights'], mu, logvar, render_gt

    def calc_loss(self, render, mu, logvar, render_gt):
        mse = torch.mean((render - render_gt) ** 2)
        ssim_loss = 1 - self.ssim_fn(render, render_gt)
        lpips_loss = self.lpips_fn((render * 2) - 1, (render_gt * 2) - 1).mean()
        kld_loss = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))

        loss = self.mse_weight * mse + self.ssim_weight * ssim_loss + \
               self.lpips_weight * lpips_loss + self.kld_weight * kld_loss

        return {
            'loss': loss,
            'mse': mse,
            'ssim': ssim_loss,
            'lpips': lpips_loss,
            'kld': kld_loss,
        }

    def log_dict(self, loss_dict, prefix):
        for k, v in loss_dict.items():
            self.log(prefix + k, v, on_step=True, logger=True)

    def make_grid(self, render, depth, render_gt):
        bs = render.shape[0] // 2
        grid = torchvision.utils.make_grid(
            torch.stack([render_gt[0], render_gt[bs], render[0], depth[0], render[bs], depth[bs]], 0))
        grid = (grid.detach().cpu().permute(1, 2, 0) * 255.).numpy().astype(np.uint8)
        return grid

    def training_step(self, batch, batch_idx):
        render, depth, weights, mu, logvar, render_gt = self.forward(batch, True)
        loss_dict = self.calc_loss(render, mu, logvar, render_gt)
        self.log_dict(loss_dict, 'train/')
        if batch_idx % self.log_image_freq == 0:
            self.logger.experiment.log({
                'train/vis': [wandb.Image(self.make_grid(
                    render, depth, render_gt
                ))]
            })
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        render, depth, _, mu, logvar, render_gt = self.forward(batch, False)
        loss_dict = self.calc_loss(render, mu, logvar, render_gt)
        self.log_dict(loss_dict, 'val/')
        if batch_idx % self.log_image_freq == 0:
            self.logger.experiment.log({
                'val/vis': [wandb.Image(self.make_grid(
                    render, depth, render_gt
                ))]
            })

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
