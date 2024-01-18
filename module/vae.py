import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch import ViT
from vit_pytorch.vit import Transformer
from einops import rearrange, repeat


class TriplaneDecoder(nn.Module):
    def __init__(self, token_num, in_dim, depth, heads, mlp_dim, out_channel, out_reso, dim_head = 64, dropout=0):
        super().__init__()
        self.token_num = token_num
        self.out_reso = out_reso
        self.out_channel = out_channel

        self.input_net = nn.Linear(in_dim, mlp_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, token_num, mlp_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(
            mlp_dim, depth, heads, dim_head, mlp_dim, dropout
        )

        assert int(token_num ** 0.5) ** 2 == token_num
        self.H = int(token_num ** 0.5)
        self.out_patch_size = out_reso // int(token_num ** 0.5)
        self.out_patch_dim = (self.out_patch_size ** 2) * out_channel
        self.output_net = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, self.out_patch_dim),
            nn.LayerNorm(self.out_patch_dim),
            nn.Linear(self.out_patch_dim, self.out_patch_dim),
        )

    def forward(self, x):
        b, n, _ = x.shape
        assert n == self.token_num
        x = self.input_net(x)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.output_net(x)
        x = x.reshape(b, self.H, self.H, self.out_patch_size, self.out_patch_size, self.out_channel)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(b, 3, self.out_channel//3, self.out_reso, self.out_reso).contiguous()
        return x


class SingleImageToTriplaneVAE(nn.Module):
    def __init__(self, backbone='dino_vits8', input_reso=256, out_reso=128, out_channel=18, z_dim=32,
                 decoder_depth=16, decoder_heads=16, decoder_mlp_dim=1024, decoder_dim_head=64, dropout=0):
        super().__init__()
        self.backbone = backbone
        
        self.input_image_size = input_reso
        self.out_reso = out_reso
        self.out_channel = out_channel
        self.z_dim = z_dim
        
        self.decoder_depth = decoder_depth
        self.decoder_heads = decoder_heads
        self.decoder_mlp_dim = decoder_mlp_dim
        self.decoder_dim_head = decoder_dim_head

        self.dropout = dropout
        self.patch_size = 8 if '8' in backbone else 16

        if 'dino' in backbone:
            self.vit = torch.hub.load('facebookresearch/dino:main', backbone)
            self.embed_dim = self.vit.embed_dim
            self.preprocess = None
        else:
            raise NotImplementedError

        self.fc_mu = nn.Linear(self.embed_dim, self.z_dim)
        self.fc_var = nn.Linear(self.embed_dim, self.z_dim)

        self.vit_decoder = TriplaneDecoder((self.input_image_size // self.patch_size) ** 2, self.z_dim,
            depth=self.decoder_depth, heads=self.decoder_heads, mlp_dim=self.decoder_mlp_dim,
            out_channel=self.out_channel, out_reso=self.out_reso, dim_head = self.decoder_dim_head, dropout=0)

    def forward(self, x, is_train):
        assert x.shape[-1] == self.input_image_size
        bs = x.shape[0]
        if 'dino' in self.backbone:
            z = self.vit.get_intermediate_layers(x, n=1)[0][:, 1:] # [bs, 1024, self.embed_dim]
        else:
            raise NotImplementedError
        
        z = z.reshape(-1, z.shape[-1])
        mu = self.fc_mu(z)
        logvar = self.fc_var(z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if is_train:
            rep_z = eps * std + mu
        else:
            rep_z = eps
        rep_z = rep_z.reshape(bs, -1, self.z_dim)
        out = self.vit_decoder(rep_z)

        return out, mu, logvar
