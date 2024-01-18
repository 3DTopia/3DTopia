import os
import imageio
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from module.model_2d import Encoder, Decoder, DiagonalGaussianDistribution, Encoder_GroupConv, Decoder_GroupConv, Encoder_GroupConv_LateFusion, Decoder_GroupConv_LateFusion
from utility.initialize import instantiate_from_config
from utility.triplane_renderer.renderer import get_embedder, NeRF, run_network, render_path1, to8b, img2mse, mse2psnr
from utility.triplane_renderer.eg3d_renderer import Renderer_TriPlane
from module.quantise import VectorQuantiser
from module.quantize_taming import EMAVectorQuantizer, VectorQuantizer2, QuantizeEMAReset

class CVQVAE(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 learning_rate=1e-3,
                 ckpt_path=None,
                 ignore_keys=[],
                 colorize_nlabels=None,
                 monitor=None,
                 decoder_ckpt=None,
                 norm=True,
                 renderer_type='nerf',
                 is_cvqvae=False,
                 renderer_config=dict(
                    rgbnet_dim=18,
                    rgbnet_width=128,
                    viewpe=0,
                    feape=0
                 ),
                 vector_quantizer_config=dict(
                    num_embed=1024,
                    beta=0.25,
                    distance='cos',
                    anchor='closest',
                    first_batch=False,
                    contras_loss=True,
                 )
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.norm = norm
        self.renderer_config = renderer_config
        self.learning_rate = learning_rate

        ddconfig['double_z'] = False
        self.encoder = Encoder_GroupConv(**ddconfig)
        self.decoder = Decoder_GroupConv(**ddconfig)

        self.lossconfig = lossconfig

        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.decoder_ckpt = decoder_ckpt
        self.renderer_type = renderer_type
        if decoder_ckpt is not None:
            self.triplane_decoder, self.triplane_render_kwargs = self.create_eg3d_decoder(decoder_ckpt)

        vector_quantizer_config['embed_dim'] = embed_dim

        if is_cvqvae:
            self.vector_quantizer = VectorQuantiser(
                **vector_quantizer_config
            )
        else:
            self.vector_quantizer = EMAVectorQuantizer(
                n_embed=vector_quantizer_config['num_embed'],
                codebook_dim = embed_dim,
                beta=vector_quantizer_config['beta']
            )
            # self.vector_quantizer = VectorQuantizer2(
            #     n_e = vector_quantizer_config['num_embed'],
            #     e_dim = embed_dim,
            #     beta = vector_quantizer_config['beta']
            # )
            # self.vector_quantizer = QuantizeEMAReset(
            #     nb_code = vector_quantizer_config['num_embed'],
            #     code_dim = embed_dim,
            #     mu = vector_quantizer_config['beta'],
            # )

        self.psum = torch.zeros([1])
        self.psum_sq = torch.zeros([1])
        self.psum_min = torch.zeros([1])
        self.psum_max = torch.zeros([1])
        self.count = 0
        self.len_dset = 0
        self.latent_list = []

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=True)
        print(f"Restored from {path}")

    def encode(self, x, rollout=False):
        if rollout:
            x = self.rollout(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        z_q, loss, (perplexity, min_encodings, encoding_indices) = self.vector_quantizer(moments)
        return z_q, loss, perplexity, encoding_indices

    def decode(self, z, unrollout=False):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if unrollout:
            dec = self.unrollout(dec)
        return dec

    def forward(self, input):
        z_q, loss, perplexity, encoding_indices = self.encode(input)
        dec = self.decode(z_q)
        return dec, loss, perplexity, encoding_indices

    def rollout(self, triplane):
        res = triplane.shape[-1]
        ch = triplane.shape[1]
        triplane = triplane.reshape(-1, 3, ch//3, res, res).permute(0, 2, 3, 1, 4).reshape(-1, ch//3, res, 3 * res)
        return triplane

    def to3daware(self, triplane):
        res = triplane.shape[-2]
        plane1 = triplane[..., :res]
        plane2 = triplane[..., res:2*res]
        plane3 = triplane[..., 2*res:3*res]

        x_mp = torch.nn.MaxPool2d((res, 1))
        y_mp = torch.nn.MaxPool2d((1, res))
        x_mp_rep = lambda i: x_mp(i).repeat(1, 1, res, 1).permute(0, 1, 3, 2)
        y_mp_rep = lambda i: y_mp(i).repeat(1, 1, 1, res).permute(0, 1, 3, 2)
        # for plane1
        plane21 = x_mp_rep(plane2)
        plane31 = torch.flip(y_mp_rep(plane3), (3,))
        new_plane1 = torch.cat([plane1, plane21, plane31], 1)
        # for plane2
        plane12 = y_mp_rep(plane1)
        plane32 = x_mp_rep(plane3)
        new_plane2 = torch.cat([plane2, plane12, plane32], 1)
        # for plane3
        plane13 = torch.flip(x_mp_rep(plane1), (2,))
        plane23 = y_mp_rep(plane2)
        new_plane3 = torch.cat([plane3, plane13, plane23], 1)

        new_plane = torch.cat([new_plane1, new_plane2, new_plane3], -1).contiguous()
        return new_plane

    def unrollout(self, triplane):
        res = triplane.shape[-2]
        ch = 3 * triplane.shape[1]
        triplane = triplane.reshape(-1, ch//3, res, 3, res).permute(0, 3, 1, 2, 4).reshape(-1, ch, res, res)
        return triplane

    def training_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, vq_loss, perplexity, encoding_indices = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, vq_loss, prefix='train/', batch=batch)
        log_dict_ae['train/perplexity'] = perplexity
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, vq_loss, perplexity, encoding_indices = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, vq_loss, prefix='val/', batch=None)
        log_dict_ae['val/perplexity'] = perplexity
        self.log_dict(log_dict_ae)

        reconstructions = self.unrollout(reconstructions)
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec
        batch_size = inputs.shape[0]
        for b in range(batch_size):
            rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
            )
            rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                reconstructions[b:b+1], batch['batch_rays'][b], batch['img'][b],
            )

            cur_psnr_list_rec = []
            for i in range(rgb.shape[0]):
                cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

            rgb_input = to8b(rgb_input.detach().cpu().numpy())
            rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
            rgb = to8b(rgb.detach().cpu().numpy())
            
            if b % 4 == 0 and batch_idx < 10:
                rgb_all = np.concatenate([rgb_gt[1], rgb_input[1], rgb[1]], 1)
                self.logger.experiment.log({
                    "val/vis": [wandb.Image(rgb_all)],
                })

            psnr_list += cur_psnr_list
            psnr_input_list += cur_psnr_list_input
            psnr_rec_list += cur_psnr_list_rec

        self.log("val/psnr_input_gt", torch.Tensor(psnr_input_list).mean(), prog_bar=True)
        self.log("val/psnr_input_rec", torch.Tensor(psnr_rec_list).mean(), prog_bar=True)
        self.log("val/psnr_rec_gt", torch.Tensor(psnr_list).mean(), prog_bar=True)

        return self.log_dict

    def to_rgb(self, plane):
        x = plane.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = torch.nn.functional.conv2d(x, weight=self.colorize)
        x = ((x - x.min()) / (x.max() - x.min()) * 255.).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        return x

    def to_rgb_triplane(self, plane):
        x = plane.float()
        if not hasattr(self, "colorize_triplane"):
            self.colorize_triplane = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = torch.nn.functional.conv2d(x, weight=self.colorize_triplane)
        x = ((x - x.min()) / (x.max() - x.min()) * 255.).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        return x
    
    def to_rgb_3daware(self, plane):
        x = plane.float()
        if not hasattr(self, "colorize_3daware"):
            self.colorize_3daware = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = torch.nn.functional.conv2d(x, weight=self.colorize_3daware)
        x = ((x - x.min()) / (x.max() - x.min()) * 255.).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        return x

    def test_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, vq_loss, perplexity, encoding_indices = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, vq_loss, prefix='test/', batch=None)
        log_dict_ae['test/perplexity'] = perplexity
        self.log_dict(log_dict_ae)

        batch_size = inputs.shape[0]
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec

        colorize_triplane_input = self.to_rgb_triplane(inputs)[0]
        colorize_triplane_output = self.to_rgb_triplane(reconstructions)[0]

        reconstructions = self.unrollout(reconstructions)

        if self.norm:
            assert NotImplementedError
        else:
            reconstructions_unnormalize = reconstructions

        if True:
            for b in range(batch_size):
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )

                cur_psnr_list_rec = []
                for i in range(rgb.shape[0]):
                    cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

                rgb_input = to8b(rgb_input.detach().cpu().numpy())
                rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
                rgb = to8b(rgb.detach().cpu().numpy())
                
                if batch_idx < 10:
                    imageio.imwrite(os.path.join(self.logger.log_dir, "{}_{}_input.png".format(batch_idx, b)), rgb_input[1])
                    imageio.imwrite(os.path.join(self.logger.log_dir, "{}_{}_rec.png".format(batch_idx, b)), rgb[1])
                    imageio.imwrite(os.path.join(self.logger.log_dir, "{}_{}_gt.png".format(batch_idx, b)), rgb_gt[1])

                psnr_list += cur_psnr_list
                psnr_input_list += cur_psnr_list_input
                psnr_rec_list += cur_psnr_list_rec

            self.log("test/psnr_input_gt", torch.Tensor(psnr_input_list).mean(), prog_bar=True)
            self.log("test/psnr_input_rec", torch.Tensor(psnr_rec_list).mean(), prog_bar=True)
            self.log("test/psnr_rec_gt", torch.Tensor(psnr_list).mean(), prog_bar=True)

    def on_test_epoch_end(self):
        mean = self.psum / self.count
        mean_min = self.psum_min / self.len_dset
        mean_max = self.psum_max / self.len_dset
        var = (self.psum_sq / self.count) - (mean ** 2)
        std = torch.sqrt(var)

        print("mean min: {}".format(mean_min))
        print("mean max: {}".format(mean_max))
        print("mean: {}".format(mean))
        print("std: {}".format(std))

        latent = np.concatenate(self.latent_list)
        q75, q25 = np.percentile(latent.reshape(-1), [75 ,25])
        median = np.median(latent.reshape(-1))
        iqr = q75 - q25
        norm_iqr = iqr * 0.7413
        print("Norm IQR: {}".format(norm_iqr))
        print("Inverse Norm IQR: {}".format(1/norm_iqr))
        print("Median: {}".format(median))

    def loss(self, inputs, reconstructions, vq_loss, prefix, batch=None):
        reconstructions = reconstructions.contiguous()
        rec_loss = F.mse_loss(inputs.contiguous(), reconstructions)
        loss = self.lossconfig.rec_weight * rec_loss + self.lossconfig.vq_weight * vq_loss

        ret_dict = {
            prefix+'mean_rec_loss': torch.abs(inputs.contiguous() - reconstructions.contiguous()).mean().detach(),
            prefix+'rec_loss': rec_loss,
            prefix+'vq_loss': vq_loss,
            prefix+'loss': loss,
        }

        render_weight = self.lossconfig.get("render_weight", 0)
        tv_weight = self.lossconfig.get("tv_weight", 0)
        l1_weight = self.lossconfig.get("l1_weight", 0)
        latent_tv_weight = self.lossconfig.get("latent_tv_weight", 0)
        latent_l1_weight = self.lossconfig.get("latent_l1_weight", 0)

        triplane_rec = self.unrollout(reconstructions)
        if render_weight > 0 and batch is not None:
            rgb_rendered, target = self.render_triplane_eg3d_decoder_sample_pixel(triplane_rec, batch['batch_rays'], batch['img'])
            render_loss = F.mse(rgb_rendered, target)
            loss += render_weight * render_loss
            ret_dict[prefix + 'render_loss'] = render_loss
        if tv_weight > 0:
            tvloss_y = torch.abs(triplane_rec[:, :, :-1] - triplane_rec[:, :, 1:]).mean()
            tvloss_x = torch.abs(triplane_rec[:, :, :, :-1] - triplane_rec[:, :, :, 1:]).mean()
            tvloss = tvloss_y + tvloss_x
            loss += tv_weight * tvloss
            ret_dict[prefix + 'tv_loss'] = tvloss
        if l1_weight > 0:
            l1 = (triplane_rec ** 2).mean()
            loss += l1_weight * l1
            ret_dict[prefix + 'l1_loss'] = l1

        ret_dict[prefix+'loss'] = loss

        return loss, ret_dict

    def create_eg3d_decoder(self, decoder_ckpt):
        triplane_decoder = Renderer_TriPlane(**self.renderer_config)
        pretrain_pth = torch.load(decoder_ckpt, map_location='cpu')
        pretrain_pth = {
            '.'.join(k.split('.')[1:]): v for k, v in pretrain_pth.items()
        }
        # import pdb; pdb.set_trace()
        triplane_decoder.load_state_dict(pretrain_pth)
        render_kwargs = {
            'depth_resolution': 128,
            'disparity_space_sampling': False,
            'box_warp': 2.4,
            'depth_resolution_importance': 128,
            'clamp_mode': 'softplus',
            'white_back': True,
            'det': True
        }
        return triplane_decoder, render_kwargs

    def render_triplane_eg3d_decoder(self, triplane, batch_rays, target):
        ray_o = batch_rays[:, 0]
        ray_d = batch_rays[:, 1]
        psnr_list = []
        rec_img_list = []
        res = triplane.shape[-2]
        for i in range(ray_o.shape[0]):
            with torch.no_grad():
                render_out = self.triplane_decoder(triplane.reshape(1, 3, -1, res, res),
                            ray_o[i:i+1], ray_d[i:i+1], self.triplane_render_kwargs, whole_img=True, tvloss=False)
            rec_img = render_out['rgb_marched'].permute(0, 2, 3, 1)
            psnr = mse2psnr(img2mse(rec_img[0], target[i]))
            psnr_list.append(psnr)
            rec_img_list.append(rec_img)
        return torch.cat(rec_img_list, 0), psnr_list

    def render_triplane_eg3d_decoder_sample_pixel(self, triplane, batch_rays, target, sample_num=1024):
        assert batch_rays.shape[1] == 1
        sel = torch.randint(batch_rays.shape[-2], [sample_num])
        ray_o = batch_rays[:, 0, 0, sel]
        ray_d = batch_rays[:, 0, 1, sel]
        res = triplane.shape[-2]
        render_out = self.triplane_decoder(triplane.reshape(triplane.shape[0], 3, -1, res, res),
                    ray_o, ray_d, self.triplane_render_kwargs, whole_img=False, tvloss=False)
        rec_img = render_out['rgb_marched']
        target = target.reshape(triplane.shape[0], -1, 3)[:, sel, :]
        return rec_img, target

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.vector_quantizer.parameters()),
                                  lr=lr)
        return opt_ae
