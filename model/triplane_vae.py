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

class AutoencoderKL(pl.LightningModule):
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
                 norm=False,
                 renderer_type='nerf',
                 renderer_config=dict(
                    rgbnet_dim=18,
                    rgbnet_width=128,
                    viewpe=0,
                    feape=0
                 ),
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.norm = norm
        self.renderer_config = renderer_config
        self.learning_rate = learning_rate
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # self.loss = instantiate_from_config(lossconfig)
        self.lossconfig = lossconfig
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.decoder_ckpt = decoder_ckpt
        self.renderer_type = renderer_type
        # if decoder_ckpt is not None:
        assert self.renderer_type in ['nerf', 'eg3d']
        if self.renderer_type == 'nerf':
            self.triplane_decoder, self.triplane_render_kwargs = self.create_nerf(decoder_ckpt)
        elif self.renderer_type == 'eg3d':
            self.triplane_decoder, self.triplane_render_kwargs = self.create_eg3d_decoder(decoder_ckpt)
        else:
            raise NotImplementedError

        self.psum = torch.zeros([1])
        self.psum_sq = torch.zeros([1])
        self.psum_min = torch.zeros([1])
        self.psum_max = torch.zeros([1])
        self.count = 0
        self.len_dset = 0
        self.latent_list = []

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, rollout=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, unrollout=False):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def unrollout(self, *args, **kwargs):
        pass

    def loss(self, inputs, reconstructions, posteriors, prefix, batch=None):
        reconstructions = reconstructions.contiguous()
        rec_loss = torch.abs(inputs.contiguous() - reconstructions)
        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = self.lossconfig.rec_weight * rec_loss + self.lossconfig.kl_weight * kl_loss

        ret_dict = {
            prefix+'mean_rec_loss': torch.abs(inputs.contiguous() - reconstructions.contiguous()).mean().detach(),
            prefix+'rec_loss': rec_loss,
            prefix+'kl_loss': kl_loss,
            prefix+'loss': loss,
            prefix+'mean': posteriors.mean.mean(),
            prefix+'logvar': posteriors.logvar.mean(),
        }

        render_weight = self.lossconfig.get("render_weight", 0)
        tv_weight = self.lossconfig.get("tv_weight", 0)
        l1_weight = self.lossconfig.get("l1_weight", 0)
        latent_tv_weight = self.lossconfig.get("latent_tv_weight", 0)
        latent_l1_weight = self.lossconfig.get("latent_l1_weight", 0)

        triplane_rec = self.unrollout(reconstructions)
        if render_weight > 0 and batch is not None:
            rgb_rendered, target = self.render_triplane_eg3d_decoder_sample_pixel(triplane_rec, batch['batch_rays'], batch['img'])
            render_loss = ((rgb_rendered - target) ** 2).sum() / rgb_rendered.shape[0] * 256
            loss += render_weight * render_loss
            ret_dict[prefix + 'render_loss'] = render_loss
        if tv_weight > 0:
            tvloss_y = torch.abs(triplane_rec[:, :, :-1] - triplane_rec[:, :, 1:]).sum() / triplane_rec.shape[0]
            tvloss_x = torch.abs(triplane_rec[:, :, :, :-1] - triplane_rec[:, :, :, 1:]).sum() / triplane_rec.shape[0]
            tvloss = tvloss_y + tvloss_x
            loss += tv_weight * tvloss
            ret_dict[prefix + 'tv_loss'] = tvloss
        if l1_weight > 0:
            l1 = (triplane_rec ** 2).sum() / triplane_rec.shape[0]
            loss += l1_weight * l1
            ret_dict[prefix + 'l1_loss'] = l1
        if latent_tv_weight > 0:
            latent = posteriors.mean
            latent_tv_y = torch.abs(latent[:, :, :-1] - latent[:, :, 1:]).sum() / latent.shape[0]
            latent_tv_x = torch.abs(latent[:, :, :, :-1] - latent[:, :, :, 1:]).sum() / latent.shape[0]
            latent_tv_loss = latent_tv_y + latent_tv_x
            loss += latent_tv_loss * latent_tv_weight
            ret_dict[prefix + 'latent_tv_loss'] = latent_tv_loss
            ret_dict[prefix + 'latent_max'] = latent.max()
            ret_dict[prefix + 'latent_min'] = latent.min()
        if latent_l1_weight > 0:
            latent = posteriors.mean
            latent_l1_loss = (latent ** 2).sum() / latent.shape[0]
            loss += latent_l1_loss * latent_l1_weight
            ret_dict[prefix + 'latent_l1_loss'] = latent_l1_loss

        return loss, ret_dict

    def training_step(self, batch, batch_idx):
        # inputs = self.get_input(batch, self.image_key)
        inputs = batch['triplane']
        reconstructions, posterior = self(inputs)

        # if optimizer_idx == 0:
        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='train/')
        # self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

        # if optimizer_idx == 1:
        #     # train the discriminator
        #     discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
        #                                         last_layer=self.get_last_layer(), split="train")

        #     self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        #     return discloss

    def validation_step(self, batch, batch_idx):
        # # inputs = self.get_input(batch, self.image_key)
        # inputs = batch['triplane']
        # reconstructions, posterior = self(inputs, sample_posterior=False)
        # aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='val/')

        # # discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
        # #                                     last_layer=self.get_last_layer(), split="val")

        # # self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        # self.log_dict(log_dict_ae)
        # # self.log_dict(log_dict_disc)
        # return self.log_dict

        inputs = batch['triplane']
        reconstructions, posterior = self(inputs, sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='val/')
        self.log_dict(log_dict_ae)

        assert not self.norm
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec
        batch_size = inputs.shape[0]
        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

            cur_psnr_list_rec = []
            for i in range(rgb.shape[0]):
                cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

            rgb_input = to8b(rgb_input.detach().cpu().numpy())
            rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
            rgb = to8b(rgb.detach().cpu().numpy())
            
            if b % 4 == 0 and batch_idx < 10:
                rgb_all = np.concatenate([rgb_gt[1], rgb_input[1], rgb[1]], 1)
                self.logger.experiment.log({
                    "val/vis": [wandb.Image(rgb_all)]
                })

            psnr_list += cur_psnr_list
            psnr_input_list += cur_psnr_list_input
            psnr_rec_list += cur_psnr_list_rec

        self.log("val/psnr_input_gt", torch.Tensor(psnr_input_list).mean(), prog_bar=True)
        self.log("val/psnr_input_rec", torch.Tensor(psnr_rec_list).mean(), prog_bar=True)
        self.log("val/psnr_rec_gt", torch.Tensor(psnr_list).mean(), prog_bar=True)

        return self.log_dict

    def create_eg3d_decoder(self, decoder_ckpt):
        triplane_decoder = Renderer_TriPlane(**self.renderer_config)
        if decoder_ckpt is not None:
            pretrain_pth = torch.load(decoder_ckpt, map_location='cpu')
            pretrain_pth = {
                '.'.join(k.split('.')[1:]): v for k, v in pretrain_pth.items()
            }
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

    def create_nerf(self, decoder_ckpt):
        # decoder_ckpt = '/mnt/petrelfs/share_data/caoziang/shapenet_triplane_car/003000.tar'

        multires = 10
        netchunk = 1024*64
        i_embed = 0
        perturb = 0
        raw_noise_std = 0

        triplanechannel=18
        triplanesize=256
        chunk=4096
        num_instance=1
        batch_size=1
        use_viewdirs = True
        white_bkgd = False
        lrate_decay = 6
        netdepth=1
        netwidth=64
        N_samples = 512
        N_importance = 0
        N_rand = 8192
        multires_views=10
        precrop_iters = 0
        precrop_frac = 0.5
        i_weights=3000

        embed_fn, input_ch = get_embedder(multires, i_embed)
        embeddirs_fn, input_ch_views = get_embedder(multires_views, i_embed)
        output_ch = 4
        skips = [4]
        model = NeRF(D=netdepth, W=netwidth,
                    input_ch=triplanechannel, size=triplanesize,output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=use_viewdirs, num_instance=num_instance)
        
        network_query_fn = lambda inputs, viewdirs, label,network_fn : \
                    run_network(inputs, viewdirs, network_fn,
                                embed_fn=embed_fn,
                                embeddirs_fn=embeddirs_fn,label=label,
                                netchunk=netchunk)

        ckpt = torch.load(decoder_ckpt)
        model.load_state_dict(ckpt['network_fn_state_dict'])

        render_kwargs_test = {
            'network_query_fn' : network_query_fn,
            'perturb' : perturb,
            'N_samples' : N_samples,
            # 'network_fn' : model,
            'use_viewdirs' : use_viewdirs,
            'white_bkgd' : white_bkgd,
            'raw_noise_std' : raw_noise_std,
        }
        render_kwargs_test['ndc'] = False
        render_kwargs_test['lindisp'] = False
        render_kwargs_test['perturb'] = False
        render_kwargs_test['raw_noise_std'] = 0.

        return model, render_kwargs_test

    def render_triplane(self, triplane, batch_rays, target, near, far, chunk=4096):
        self.triplane_decoder.tri_planes.copy_(triplane.detach())
        self.triplane_render_kwargs['network_fn'] = self.triplane_decoder
        # print(triplane.device)
        # print(batch_rays.device)
        # print(target.device)
        # print(near.device)
        # print(far.device)
        with torch.no_grad():
            rgb, _, _, psnr_list = \
                render_path1(batch_rays, chunk, self.triplane_render_kwargs, gt_imgs=target,
                            near=near, far=far, label=torch.Tensor([0]).long().to(triplane.device))
        return rgb, psnr_list

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

    def test_step(self, batch, batch_idx):
        # inputs = batch['triplane']
        # reconstructions, posterior = self(inputs, sample_posterior=False)
        # aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='test/')
        # self.log_dict(log_dict_ae)

        # batch_size = inputs.shape[0]
        # psnr_list = [] # between rec and gt
        # psnr_input_list = [] # between input and gt
        # psnr_rec_list = [] # between input and rec

        # mean = torch.Tensor([
        #     0.2820,  0.4103, -0.2988,  0.1491,  0.4429, -0.3117,  0.2830,  0.4115,
        #     -0.3032,  0.1530,  0.4466, -0.3165,  0.2617,  0.3837, -0.2692,  0.1098,
        #     0.4101, -0.2922
        # ]).reshape(1, 18, 1, 1).to(inputs.device)
        # std = torch.Tensor([
        #     1.1696, 1.1287, 1.1733, 1.1583, 1.1238, 1.1675, 1.1978, 1.1585, 1.1949,
        #     1.1660, 1.1576, 1.1998, 1.1987, 1.1546, 1.1930, 1.1724, 1.1450, 1.2027
        # ]).reshape(1, 18, 1, 1).to(inputs.device)

        # if self.norm:
        #     reconstructions_unnormalize = reconstructions * std + mean
        # else:
        #     reconstructions_unnormalize = reconstructions

        # for b in range(batch_size):
        #     # rgb_input, cur_psnr_list_input = self.render_triplane(
        #     #     batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
        #     #     batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
        #     # )
        #     # rgb, cur_psnr_list = self.render_triplane(
        #     #     reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
        #     #     batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
        #     # )

        #     if self.renderer_type == 'nerf':
        #         rgb_input, cur_psnr_list_input = self.render_triplane(
        #             batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
        #             batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
        #         )
        #         rgb, cur_psnr_list = self.render_triplane(
        #             reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
        #             batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
        #         )
        #     elif self.renderer_type == 'eg3d':
        #         rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
        #             batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
        #         )
        #         rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
        #             reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img'][b],
        #         )
        #     else:
        #         raise NotImplementedError

        #     cur_psnr_list_rec = []
        #     for i in range(rgb.shape[0]):
        #         cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

        #     rgb_input = to8b(rgb_input.detach().cpu().numpy())
        #     rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
        #     rgb = to8b(rgb.detach().cpu().numpy())
            
        #     if batch_idx < 1:
        #         imageio.imwrite(os.path.join(self.logger.log_dir, "{}_{}_input.png".format(batch_idx, b)), rgb_input[1])
        #         imageio.imwrite(os.path.join(self.logger.log_dir, "{}_{}_rec.png".format(batch_idx, b)), rgb[1])
        #         imageio.imwrite(os.path.join(self.logger.log_dir, "{}_{}_gt.png".format(batch_idx, b)), rgb_gt[1])

        #     psnr_list += cur_psnr_list
        #     psnr_input_list += cur_psnr_list_input
        #     psnr_rec_list += cur_psnr_list_rec

        # self.log("test/psnr_input_gt", torch.Tensor(psnr_input_list).mean(), prog_bar=True)
        # self.log("test/psnr_input_rec", torch.Tensor(psnr_rec_list).mean(), prog_bar=True)
        # self.log("test/psnr_rec_gt", torch.Tensor(psnr_list).mean(), prog_bar=True)

        inputs = batch['triplane']
        reconstructions, posterior = self(inputs, sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='test/', batch=None)
        self.log_dict(log_dict_ae)

        batch_size = inputs.shape[0]
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec

        z = posterior.mode()
        colorize_z = self.to_rgb(z)[0]
        colorize_triplane_input = self.to_rgb_triplane(inputs)[0]
        colorize_triplane_output = self.to_rgb_triplane(reconstructions)[0]
        # colorize_triplane_rollout_3daware = self.to_rgb_3daware(self.to3daware(inputs))[0]
        # res = inputs.shape[1]
        # colorize_triplane_rollout_3daware_1 = self.to_rgb_triplane(self.to3daware(inputs)[:,res:2*res])[0]
        # colorize_triplane_rollout_3daware_2 = self.to_rgb_triplane(self.to3daware(inputs)[:,2*res:3*res])[0]
        if batch_idx < 10:
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_z_{}.png".format(batch_idx)), colorize_z)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_{}.png".format(batch_idx)), colorize_triplane_input)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_output_{}.png".format(batch_idx)), colorize_triplane_output)
            # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}.png".format(batch_idx)), colorize_triplane_rollout_3daware)
            # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}_1.png".format(batch_idx)), colorize_triplane_rollout_3daware_1)
            # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}_2.png".format(batch_idx)), colorize_triplane_rollout_3daware_2)

        np_z = z.detach().cpu().numpy()
        # with open(os.path.join(self.logger.log_dir, "latent_{}.npz".format(batch_idx)), 'wb') as f:
        #     np.save(f, np_z)

        self.latent_list.append(np_z)

        if self.psum.device != z.device:
            self.psum = self.psum.to(z.device)
            self.psum_sq = self.psum_sq.to(z.device)
            self.psum_min = self.psum_min.to(z.device)
            self.psum_max = self.psum_max.to(z.device)
        self.psum += z.sum()
        self.psum_sq += (z ** 2).sum()
        self.psum_min += z.reshape(-1).min(-1)[0]
        self.psum_max += z.reshape(-1).max(-1)[0]
        assert len(z.shape) == 4
        self.count += z.shape[0] * z.shape[1] * z.shape[2] * z.shape[3]
        self.len_dset += 1

        if self.norm:
            assert NotImplementedError
        else:
            reconstructions_unnormalize = reconstructions

        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

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

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
        #                             lr=lr, betas=(0.5, 0.9))
        # return [opt_ae, opt_disc], []
        return opt_ae

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


class AutoencoderKLRollOut(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psum = torch.zeros([1])
        self.psum_sq = torch.zeros([1])
        self.psum_min = torch.zeros([1])
        self.psum_max = torch.zeros([1])
        self.count = 0
        self.len_dset = 0

    def rollout(self, triplane):
        res = triplane.shape[-1]
        ch = triplane.shape[1]
        triplane = triplane.reshape(-1, 3, ch//3, res, res).permute(0, 2, 3, 1, 4).reshape(-1, ch//3, res, 3 * res)
        return triplane

    def unrollout(self, triplane):
        res = triplane.shape[-2]
        ch = 3 * triplane.shape[1]
        triplane = triplane.reshape(-1, ch//3, res, 3, res).permute(0, 3, 1, 2, 4).reshape(-1, ch, res, res)
        return triplane

    def encode(self, x, rollout=False):
        if rollout:
            x = self.rollout(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, unrollout=False):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if unrollout:
            dec = self.unrollout(dec)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='train/')
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(inputs, sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='val/')
        self.log_dict(log_dict_ae)

        assert not self.norm
        reconstructions = self.unrollout(reconstructions)
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec
        batch_size = inputs.shape[0]
        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

            cur_psnr_list_rec = []
            for i in range(rgb.shape[0]):
                cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

            rgb_input = to8b(rgb_input.detach().cpu().numpy())
            rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
            rgb = to8b(rgb.detach().cpu().numpy())
            
            if b % 4 == 0 and batch_idx < 10:
                rgb_all = np.concatenate([rgb_gt[1], rgb_input[1], rgb[1]], 1)
                self.logger.experiment.log({
                    "val/vis": [wandb.Image(rgb_all)]
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

    def test_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(inputs, sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='test/')
        self.log_dict(log_dict_ae)

        batch_size = inputs.shape[0]
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec

        z = posterior.mode()
        colorize_z = self.to_rgb(z)[0]
        colorize_triplane_input = self.to_rgb_triplane(inputs)[0]
        colorize_triplane_output = self.to_rgb_triplane(reconstructions)[0]
        # if batch_idx < 1:
        # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_z_{}.png".format(batch_idx)), colorize_z)
        # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_{}.png".format(batch_idx)), colorize_triplane_input)
        # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_output_{}.png".format(batch_idx)), colorize_triplane_output)

        reconstructions = self.unrollout(reconstructions)

        if self.psum.device != z.device:
            self.psum = self.psum.to(z.device)
            self.psum_sq = self.psum_sq.to(z.device)
            self.psum_min = self.psum_min.to(z.device)
            self.psum_max = self.psum_max.to(z.device)
        self.psum += z.sum()
        self.psum_sq += (z ** 2).sum()
        self.psum_min += z.reshape(-1).min(-1)[0]
        self.psum_max += z.reshape(-1).max(-1)[0]
        assert len(z.shape) == 4
        self.count += z.shape[0] * z.shape[1] * z.shape[2] * z.shape[3]
        self.len_dset += 1

        # mean = torch.Tensor([
        #     0.2820,  0.4103, -0.2988,  0.1491,  0.4429, -0.3117,  0.2830,  0.4115,
        #     -0.3032,  0.1530,  0.4466, -0.3165,  0.2617,  0.3837, -0.2692,  0.1098,
        #     0.4101, -0.2922
        # ]).reshape(1, 18, 1, 1).to(inputs.device)
        # std = torch.Tensor([
        #     1.1696, 1.1287, 1.1733, 1.1583, 1.1238, 1.1675, 1.1978, 1.1585, 1.1949,
        #     1.1660, 1.1576, 1.1998, 1.1987, 1.1546, 1.1930, 1.1724, 1.1450, 1.2027
        # ]).reshape(1, 18, 1, 1).to(inputs.device)

        mean = torch.Tensor([
            -1.8449, -1.8242,  0.9667, -1.0187,  1.0647, -0.5422, -1.8632, -1.8435,
            0.9314, -1.0261,  1.0356, -0.5484, -1.8543, -1.8348,  0.9109, -1.0169,
            1.0160, -0.5467
        ]).reshape(1, 18, 1, 1).to(inputs.device)
        std = torch.Tensor([
            1.7593, 1.6127, 2.7132, 1.5500, 2.7893, 0.7707, 2.1114, 1.9198, 2.6586,
            1.8021, 2.5473, 1.0305, 1.7042, 1.7507, 2.4270, 1.4365, 2.2511, 0.8792
        ]).reshape(1, 18, 1, 1).to(inputs.device)

        if self.norm:
            reconstructions_unnormalize = reconstructions * std + mean
        else:
            reconstructions_unnormalize = reconstructions

        # for b in range(batch_size):
        #     if self.renderer_type == 'nerf':
        #         rgb_input, cur_psnr_list_input = self.render_triplane(
        #             batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
        #             batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
        #         )
        #         rgb, cur_psnr_list = self.render_triplane(
        #             reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
        #             batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
        #         )
        #     elif self.renderer_type == 'eg3d':
        #         rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
        #             batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
        #         )
        #         rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
        #             reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img'][b],
        #         )
        #     else:
        #         raise NotImplementedError

        #     cur_psnr_list_rec = []
        #     for i in range(rgb.shape[0]):
        #         cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

        #     rgb_input = to8b(rgb_input.detach().cpu().numpy())
        #     rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
        #     rgb = to8b(rgb.detach().cpu().numpy())
            
        #     # if batch_idx < 1:
        #     imageio.imwrite(os.path.join(self.logger.log_dir, "{}_{}_input.png".format(batch_idx, b)), rgb_input[1])
        #     imageio.imwrite(os.path.join(self.logger.log_dir, "{}_{}_rec.png".format(batch_idx, b)), rgb[1])
        #     imageio.imwrite(os.path.join(self.logger.log_dir, "{}_{}_gt.png".format(batch_idx, b)), rgb_gt[1])

        #     psnr_list += cur_psnr_list
        #     psnr_input_list += cur_psnr_list_input
        #     psnr_rec_list += cur_psnr_list_rec

        # self.log("test/psnr_input_gt", torch.Tensor(psnr_input_list).mean(), prog_bar=True)
        # self.log("test/psnr_input_rec", torch.Tensor(psnr_rec_list).mean(), prog_bar=True)
        # self.log("test/psnr_rec_gt", torch.Tensor(psnr_list).mean(), prog_bar=True)

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


class AutoencoderKLRollOut3DAware(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        try:
            ckpt_path = kwargs['ckpt_path']
            kwargs['ckpt_path'] = None
        except:
            ckpt_path = None
        
        super().__init__(*args, **kwargs)
        self.psum = torch.zeros([1])
        self.psum_sq = torch.zeros([1])
        self.psum_min = torch.zeros([1])
        self.psum_max = torch.zeros([1])
        self.count = 0
        self.len_dset = 0

        ddconfig = kwargs['ddconfig']
        ddconfig['z_channels'] *= 3
        del self.decoder
        del self.post_quant_conv
        self.decoder = Decoder(**ddconfig)
        self.post_quant_conv = torch.nn.Conv2d(kwargs['embed_dim'] * 3, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

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

    def encode(self, x, rollout=False):
        if rollout:
            x = self.to3daware(self.rollout(x))
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, unrollout=False):
        z = self.to3daware(z)
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if unrollout:
            dec = self.unrollout(dec)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(self.to3daware(inputs))
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='train/', batch=batch)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(self.to3daware(inputs), sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='val/', batch=None)
        self.log_dict(log_dict_ae)

        assert not self.norm
        reconstructions = self.unrollout(reconstructions)
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec
        batch_size = inputs.shape[0]
        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

            cur_psnr_list_rec = []
            for i in range(rgb.shape[0]):
                cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

            rgb_input = to8b(rgb_input.detach().cpu().numpy())
            rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
            rgb = to8b(rgb.detach().cpu().numpy())
            
            if b % 4 == 0 and batch_idx < 10:
                rgb_all = np.concatenate([rgb_gt[1], rgb_input[1], rgb[1]], 1)
                self.logger.experiment.log({
                    "val/vis": [wandb.Image(rgb_all)]
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
        reconstructions, posterior = self(self.to3daware(inputs), sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='test/', batch=None)
        self.log_dict(log_dict_ae)

        batch_size = inputs.shape[0]
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec

        z = posterior.mode()
        colorize_z = self.to_rgb(z)[0]
        colorize_triplane_input = self.to_rgb_triplane(inputs)[0]
        colorize_triplane_output = self.to_rgb_triplane(reconstructions)[0]
        colorize_triplane_rollout_3daware = self.to_rgb_3daware(self.to3daware(inputs))[0]
        res = inputs.shape[1]
        colorize_triplane_rollout_3daware_1 = self.to_rgb_triplane(self.to3daware(inputs)[:,res:2*res])[0]
        colorize_triplane_rollout_3daware_2 = self.to_rgb_triplane(self.to3daware(inputs)[:,2*res:3*res])[0]
        if batch_idx < 10:
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_z_{}.png".format(batch_idx)), colorize_z)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_{}.png".format(batch_idx)), colorize_triplane_input)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_output_{}.png".format(batch_idx)), colorize_triplane_output)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}.png".format(batch_idx)), colorize_triplane_rollout_3daware)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}_1.png".format(batch_idx)), colorize_triplane_rollout_3daware_1)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}_2.png".format(batch_idx)), colorize_triplane_rollout_3daware_2)

        reconstructions = self.unrollout(reconstructions)

        if self.psum.device != z.device:
            self.psum = self.psum.to(z.device)
            self.psum_sq = self.psum_sq.to(z.device)
            self.psum_min = self.psum_min.to(z.device)
            self.psum_max = self.psum_max.to(z.device)
        self.psum += z.sum()
        self.psum_sq += (z ** 2).sum()
        self.psum_min += z.reshape(-1).min(-1)[0]
        self.psum_max += z.reshape(-1).max(-1)[0]
        assert len(z.shape) == 4
        self.count += z.shape[0] * z.shape[1] * z.shape[2] * z.shape[3]
        self.len_dset += 1

        if self.norm:
            assert NotImplementedError
        else:
            reconstructions_unnormalize = reconstructions

        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

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


class AutoencoderKLRollOut3DAwareOnlyInput(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        try:
            ckpt_path = kwargs['ckpt_path']
            kwargs['ckpt_path'] = None
        except:
            ckpt_path = None
        
        super().__init__(*args, **kwargs)
        self.psum = torch.zeros([1])
        self.psum_sq = torch.zeros([1])
        self.psum_min = torch.zeros([1])
        self.psum_max = torch.zeros([1])
        self.count = 0
        self.len_dset = 0

        # ddconfig = kwargs['ddconfig']
        # ddconfig['z_channels'] *= 3
        # self.decoder = Decoder(**ddconfig)
        # self.post_quant_conv = torch.nn.Conv2d(kwargs['embed_dim'] * 3, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

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

    def encode(self, x, rollout=False):
        if rollout:
            x = self.to3daware(self.rollout(x))
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, unrollout=False):
        # z = self.to3daware(z)
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if unrollout:
            dec = self.unrollout(dec)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(self.to3daware(inputs))
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='train/')
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(self.to3daware(inputs), sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='val/')
        self.log_dict(log_dict_ae)

        assert not self.norm
        reconstructions = self.unrollout(reconstructions)
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec
        batch_size = inputs.shape[0]
        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

            cur_psnr_list_rec = []
            for i in range(rgb.shape[0]):
                cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

            rgb_input = to8b(rgb_input.detach().cpu().numpy())
            rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
            rgb = to8b(rgb.detach().cpu().numpy())
            
            if b % 4 == 0 and batch_idx < 10:
                rgb_all = np.concatenate([rgb_gt[1], rgb_input[1], rgb[1]], 1)
                self.logger.experiment.log({
                    "val/vis": [wandb.Image(rgb_all)]
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

    def test_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(self.to3daware(inputs), sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='test/')
        self.log_dict(log_dict_ae)

        batch_size = inputs.shape[0]
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec

        z = posterior.mode()
        colorize_z = self.to_rgb(z)[0]
        colorize_triplane_input = self.to_rgb_triplane(inputs)[0]
        colorize_triplane_output = self.to_rgb_triplane(reconstructions)[0]
        if batch_idx < 10:
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_z_{}.png".format(batch_idx)), colorize_z)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_{}.png".format(batch_idx)), colorize_triplane_input)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_output_{}.png".format(batch_idx)), colorize_triplane_output)

        reconstructions = self.unrollout(reconstructions)

        if self.psum.device != z.device:
            self.psum = self.psum.to(z.device)
            self.psum_sq = self.psum_sq.to(z.device)
            self.psum_min = self.psum_min.to(z.device)
            self.psum_max = self.psum_max.to(z.device)
        self.psum += z.sum()
        self.psum_sq += (z ** 2).sum()
        self.psum_min += z.reshape(-1).min(-1)[0]
        self.psum_max += z.reshape(-1).max(-1)[0]
        assert len(z.shape) == 4
        self.count += z.shape[0] * z.shape[1] * z.shape[2] * z.shape[3]
        self.len_dset += 1

        if self.norm:
            assert NotImplementedError
        else:
            reconstructions_unnormalize = reconstructions

        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

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


class AutoencoderKLRollOut3DAwareMeanPool(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        try:
            ckpt_path = kwargs['ckpt_path']
            kwargs['ckpt_path'] = None
        except:
            ckpt_path = None
        
        super().__init__(*args, **kwargs)
        self.psum = torch.zeros([1])
        self.psum_sq = torch.zeros([1])
        self.psum_min = torch.zeros([1])
        self.psum_max = torch.zeros([1])
        self.count = 0
        self.len_dset = 0

        ddconfig = kwargs['ddconfig']
        ddconfig['z_channels'] *= 3
        self.decoder = Decoder(**ddconfig)
        self.post_quant_conv = torch.nn.Conv2d(kwargs['embed_dim'] * 3, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

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

        x_mp = torch.nn.AvgPool2d((res, 1))
        y_mp = torch.nn.AvgPool2d((1, res))
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

    def encode(self, x, rollout=False):
        if rollout:
            x = self.to3daware(self.rollout(x))
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, unrollout=False):
        z = self.to3daware(z)
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if unrollout:
            dec = self.unrollout(dec)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(self.to3daware(inputs))
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='train/')
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(self.to3daware(inputs), sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='val/')
        self.log_dict(log_dict_ae)

        assert not self.norm
        reconstructions = self.unrollout(reconstructions)
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec
        batch_size = inputs.shape[0]
        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

            cur_psnr_list_rec = []
            for i in range(rgb.shape[0]):
                cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

            rgb_input = to8b(rgb_input.detach().cpu().numpy())
            rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
            rgb = to8b(rgb.detach().cpu().numpy())
            
            if b % 4 == 0 and batch_idx < 10:
                rgb_all = np.concatenate([rgb_gt[1], rgb_input[1], rgb[1]], 1)
                self.logger.experiment.log({
                    "val/vis": [wandb.Image(rgb_all)]
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
        reconstructions, posterior = self(self.to3daware(inputs), sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='test/')
        self.log_dict(log_dict_ae)

        batch_size = inputs.shape[0]
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec

        z = posterior.mode()
        colorize_z = self.to_rgb(z)[0]
        colorize_triplane_input = self.to_rgb_triplane(inputs)[0]
        colorize_triplane_output = self.to_rgb_triplane(reconstructions)[0]
        colorize_triplane_rollout_3daware = self.to_rgb_3daware(self.to3daware(inputs))[0]
        res = inputs.shape[1]
        colorize_triplane_rollout_3daware_1 = self.to_rgb_triplane(self.to3daware(inputs)[:,res:2*res])[0]
        colorize_triplane_rollout_3daware_2 = self.to_rgb_triplane(self.to3daware(inputs)[:,2*res:3*res])[0]
        if batch_idx < 10:
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_z_{}.png".format(batch_idx)), colorize_z)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_{}.png".format(batch_idx)), colorize_triplane_input)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_output_{}.png".format(batch_idx)), colorize_triplane_output)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}.png".format(batch_idx)), colorize_triplane_rollout_3daware)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}_1.png".format(batch_idx)), colorize_triplane_rollout_3daware_1)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}_2.png".format(batch_idx)), colorize_triplane_rollout_3daware_2)

        reconstructions = self.unrollout(reconstructions)

        if self.psum.device != z.device:
            self.psum = self.psum.to(z.device)
            self.psum_sq = self.psum_sq.to(z.device)
            self.psum_min = self.psum_min.to(z.device)
            self.psum_max = self.psum_max.to(z.device)
        self.psum += z.sum()
        self.psum_sq += (z ** 2).sum()
        self.psum_min += z.reshape(-1).min(-1)[0]
        self.psum_max += z.reshape(-1).max(-1)[0]
        assert len(z.shape) == 4
        self.count += z.shape[0] * z.shape[1] * z.shape[2] * z.shape[3]
        self.len_dset += 1

        if self.norm:
            assert NotImplementedError
        else:
            reconstructions_unnormalize = reconstructions

        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

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


class AutoencoderKLGroupConv(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        try:
            ckpt_path = kwargs['ckpt_path']
            kwargs['ckpt_path'] = None
        except:
            ckpt_path = None

        super().__init__(*args, **kwargs)
        self.latent_list = []
        self.psum = torch.zeros([1])
        self.psum_sq = torch.zeros([1])
        self.psum_min = torch.zeros([1])
        self.psum_max = torch.zeros([1])
        self.count = 0
        self.len_dset = 0

        ddconfig = kwargs['ddconfig']
        # ddconfig['z_channels'] *= 3
        del self.decoder
        del self.encoder
        self.encoder = Encoder_GroupConv(**ddconfig)
        self.decoder = Decoder_GroupConv(**ddconfig)

        if "mean" in ddconfig:
            print("Using mean std!!")
            self.triplane_mean = torch.Tensor(ddconfig['mean']).reshape(-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
            self.triplane_std = torch.Tensor(ddconfig['std']).reshape(-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
        else:
            self.triplane_mean = None
            self.triplane_std = None

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

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

    def encode(self, x, rollout=False):
        if rollout:
            # x = self.to3daware(self.rollout(x))
            x = self.rollout(x)
        if self.triplane_mean is not None:
            x = (x - self.triplane_mean.to(x.device)) / self.triplane_std.to(x.device)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, unrollout=False):
        # z = self.to3daware(z)
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if self.triplane_mean is not None:
            dec = dec * self.triplane_std.to(dec.device) + self.triplane_mean.to(dec.device)
        if unrollout:
            dec = self.unrollout(dec)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='train/', batch=batch)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(inputs, sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='val/', batch=None)
        self.log_dict(log_dict_ae)

        z = posterior.mode()
        colorize_z = self.to_rgb(z)[0]
        assert not self.norm
        reconstructions = self.unrollout(reconstructions)
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec
        batch_size = inputs.shape[0]
        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

            cur_psnr_list_rec = []
            for i in range(rgb.shape[0]):
                cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

            rgb_input = to8b(rgb_input.detach().cpu().numpy())
            rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
            rgb = to8b(rgb.detach().cpu().numpy())

            rgb_input = np.stack([rgb_input[..., 2], rgb_input[..., 1], rgb_input[..., 0]], -1)
            rgb = np.stack([rgb[..., 2], rgb[..., 1], rgb[..., 0]], -1)
            
            if b % 2 == 0 and batch_idx < 10:
                rgb_all = np.concatenate([rgb_gt[1], rgb_input[1], rgb[1]], 1)
                self.logger.experiment.log({
                    "val/vis": [wandb.Image(rgb_all)],
                    "val/latent_vis": [wandb.Image(colorize_z)]
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
        reconstructions, posterior = self(inputs, sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='test/', batch=None)
        self.log_dict(log_dict_ae)

        batch_size = inputs.shape[0]
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec

        z = posterior.mode()
        colorize_z = self.to_rgb(z)[0]
        colorize_triplane_input = self.to_rgb_triplane(inputs)[0]
        colorize_triplane_output = self.to_rgb_triplane(reconstructions)[0]

        import os
        import random
        import string
        # z_np = z.detach().cpu().numpy()
        z_np = inputs.detach().cpu().numpy()
        fname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)) + '.npy'
        with open(os.path.join('/mnt/lustre/hongfangzhou.p/AE3D/tmp', fname), 'wb') as f:
            np.save(f, z_np)

        # colorize_triplane_rollout_3daware = self.to_rgb_3daware(self.to3daware(inputs))[0]
        # res = inputs.shape[1]
        # colorize_triplane_rollout_3daware_1 = self.to_rgb_triplane(self.to3daware(inputs)[:,res:2*res])[0]
        # colorize_triplane_rollout_3daware_2 = self.to_rgb_triplane(self.to3daware(inputs)[:,2*res:3*res])[0]
        if batch_idx < 0:
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_z_{}.png".format(batch_idx)), colorize_z)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_{}.png".format(batch_idx)), colorize_triplane_input)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_output_{}.png".format(batch_idx)), colorize_triplane_output)
            # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}.png".format(batch_idx)), colorize_triplane_rollout_3daware)
            # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}_1.png".format(batch_idx)), colorize_triplane_rollout_3daware_1)
            # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}_2.png".format(batch_idx)), colorize_triplane_rollout_3daware_2)

        np_z = z.detach().cpu().numpy()
        # with open(os.path.join(self.logger.log_dir, "latent_{}.npz".format(batch_idx)), 'wb') as f:
        #     np.save(f, np_z)

        self.latent_list.append(np_z)

        reconstructions = self.unrollout(reconstructions)

        if self.psum.device != z.device:
            self.psum = self.psum.to(z.device)
            self.psum_sq = self.psum_sq.to(z.device)
            self.psum_min = self.psum_min.to(z.device)
            self.psum_max = self.psum_max.to(z.device)
        self.psum += z.sum()
        self.psum_sq += (z ** 2).sum()
        self.psum_min += z.reshape(-1).min(-1)[0]
        self.psum_max += z.reshape(-1).max(-1)[0]
        assert len(z.shape) == 4
        self.count += z.shape[0] * z.shape[1] * z.shape[2] * z.shape[3]
        self.len_dset += 1

        if self.norm:
            assert NotImplementedError
        else:
            reconstructions_unnormalize = reconstructions

        if True:
            for b in range(batch_size):
                if self.renderer_type == 'nerf':
                    rgb_input, cur_psnr_list_input = self.render_triplane(
                        batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                        batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                    )
                    rgb, cur_psnr_list = self.render_triplane(
                        reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                        batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                    )
                elif self.renderer_type == 'eg3d':
                    rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                        batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                    )
                    rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                        reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img'][b],
                    )
                else:
                    raise NotImplementedError

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

    def loss(self, inputs, reconstructions, posteriors, prefix, batch=None):
        reconstructions = reconstructions.contiguous()
        # rec_loss = torch.abs(inputs.contiguous() - reconstructions)
        # rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]
        rec_loss = F.mse_loss(inputs.contiguous(), reconstructions)
        kl_loss = posteriors.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = kl_loss.mean()
        loss = self.lossconfig.rec_weight * rec_loss + self.lossconfig.kl_weight * kl_loss

        ret_dict = {
            prefix+'mean_rec_loss': torch.abs(inputs.contiguous() - reconstructions.contiguous()).mean().detach(),
            prefix+'rec_loss': rec_loss,
            prefix+'kl_loss': kl_loss,
            prefix+'loss': loss,
            prefix+'mean': posteriors.mean.mean(),
            prefix+'logvar': posteriors.logvar.mean(),
        }


        latent = posteriors.mean
        ret_dict[prefix + 'latent_max'] = latent.max()
        ret_dict[prefix + 'latent_min'] = latent.min()

        render_weight = self.lossconfig.get("render_weight", 0)
        tv_weight = self.lossconfig.get("tv_weight", 0)
        l1_weight = self.lossconfig.get("l1_weight", 0)
        latent_tv_weight = self.lossconfig.get("latent_tv_weight", 0)
        latent_l1_weight = self.lossconfig.get("latent_l1_weight", 0)

        triplane_rec = self.unrollout(reconstructions)
        if render_weight > 0 and batch is not None:
            rgb_rendered, target = self.render_triplane_eg3d_decoder_sample_pixel(triplane_rec, batch['batch_rays'], batch['img'])
            # render_loss = ((rgb_rendered - target) ** 2).sum() / rgb_rendered.shape[0] * 256
            render_loss = F.mse_loss(rgb_rendered, target)
            loss += render_weight * render_loss
            ret_dict[prefix + 'render_loss'] = render_loss
        if tv_weight > 0:
            tvloss_y = F.mse_loss(triplane_rec[:, :, :-1], triplane_rec[:, :, 1:])
            tvloss_x = F.mse_loss(triplane_rec[:, :, :, :-1], triplane_rec[:, :, :, 1:])
            tvloss = tvloss_y + tvloss_x
            loss += tv_weight * tvloss
            ret_dict[prefix + 'tv_loss'] = tvloss
        if l1_weight > 0:
            l1 = (triplane_rec ** 2).mean()
            loss += l1_weight * l1
            ret_dict[prefix + 'l1_loss'] = l1
        if latent_tv_weight > 0:
            latent = posteriors.mean
            latent_tv_y = F.mse_loss(latent[:, :, :-1], latent[:, :, 1:])
            latent_tv_x = F.mse_loss(latent[:, :, :, :-1], latent[:, :, :, 1:])
            latent_tv_loss = latent_tv_y + latent_tv_x
            loss += latent_tv_loss * latent_tv_weight
            ret_dict[prefix + 'latent_tv_loss'] = latent_tv_loss
        if latent_l1_weight > 0:
            latent = posteriors.mean
            latent_l1_loss = (latent ** 2).mean()
            loss += latent_l1_loss * latent_l1_weight
            ret_dict[prefix + 'latent_l1_loss'] = latent_l1_loss

        return loss, ret_dict


class AutoencoderKLGroupConvLateFusion(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        try:
            ckpt_path = kwargs['ckpt_path']
            kwargs['ckpt_path'] = None
        except:
            ckpt_path = None

        super().__init__(*args, **kwargs)
        self.latent_list = []
        self.psum = torch.zeros([1])
        self.psum_sq = torch.zeros([1])
        self.psum_min = torch.zeros([1])
        self.psum_max = torch.zeros([1])
        self.count = 0
        self.len_dset = 0

        ddconfig = kwargs['ddconfig']
        del self.decoder
        del self.encoder
        self.encoder = Encoder_GroupConv_LateFusion(**ddconfig)
        self.decoder = Decoder_GroupConv_LateFusion(**ddconfig)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

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

    def encode(self, x, rollout=False):
        if rollout:
            x = self.rollout(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, unrollout=False):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if unrollout:
            dec = self.unrollout(dec)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='train/', batch=batch)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(inputs, sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='val/', batch=None)
        self.log_dict(log_dict_ae)

        assert not self.norm
        reconstructions = self.unrollout(reconstructions)
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec
        batch_size = inputs.shape[0]
        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

            cur_psnr_list_rec = []
            for i in range(rgb.shape[0]):
                cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

            rgb_input = to8b(rgb_input.detach().cpu().numpy())
            rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
            rgb = to8b(rgb.detach().cpu().numpy())
            
            if b % 4 == 0 and batch_idx < 10:
                rgb_all = np.concatenate([rgb_gt[1], rgb_input[1], rgb[1]], 1)
                self.logger.experiment.log({
                    "val/vis": [wandb.Image(rgb_all)]
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
        reconstructions, posterior = self(inputs, sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='test/', batch=None)
        self.log_dict(log_dict_ae)

        batch_size = inputs.shape[0]
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec

        z = posterior.mode()
        colorize_z = self.to_rgb(z)[0]
        colorize_triplane_input = self.to_rgb_triplane(inputs)[0]
        colorize_triplane_output = self.to_rgb_triplane(reconstructions)[0]
        # colorize_triplane_rollout_3daware = self.to_rgb_3daware(self.to3daware(inputs))[0]
        # res = inputs.shape[1]
        # colorize_triplane_rollout_3daware_1 = self.to_rgb_triplane(self.to3daware(inputs)[:,res:2*res])[0]
        # colorize_triplane_rollout_3daware_2 = self.to_rgb_triplane(self.to3daware(inputs)[:,2*res:3*res])[0]
        if batch_idx < 10:
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_z_{}.png".format(batch_idx)), colorize_z)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_{}.png".format(batch_idx)), colorize_triplane_input)
            imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_output_{}.png".format(batch_idx)), colorize_triplane_output)
            # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}.png".format(batch_idx)), colorize_triplane_rollout_3daware)
            # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}_1.png".format(batch_idx)), colorize_triplane_rollout_3daware_1)
            # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}_2.png".format(batch_idx)), colorize_triplane_rollout_3daware_2)

        np_z = z.detach().cpu().numpy()
        # with open(os.path.join(self.logger.log_dir, "latent_{}.npz".format(batch_idx)), 'wb') as f:
        #     np.save(f, np_z)

        self.latent_list.append(np_z)

        reconstructions = self.unrollout(reconstructions)

        if self.psum.device != z.device:
            self.psum = self.psum.to(z.device)
            self.psum_sq = self.psum_sq.to(z.device)
            self.psum_min = self.psum_min.to(z.device)
            self.psum_max = self.psum_max.to(z.device)
        self.psum += z.sum()
        self.psum_sq += (z ** 2).sum()
        self.psum_min += z.reshape(-1).min(-1)[0]
        self.psum_max += z.reshape(-1).max(-1)[0]
        assert len(z.shape) == 4
        self.count += z.shape[0] * z.shape[1] * z.shape[2] * z.shape[3]
        self.len_dset += 1

        if self.norm:
            assert NotImplementedError
        else:
            reconstructions_unnormalize = reconstructions

        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

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


from module.model_2d import ViTEncoder, ViTDecoder

class AutoencoderVIT(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        try:
            ckpt_path = kwargs['ckpt_path']
            kwargs['ckpt_path'] = None
        except:
            ckpt_path = None

        super().__init__(*args, **kwargs)
        self.latent_list = []
        self.psum = torch.zeros([1])
        self.psum_sq = torch.zeros([1])
        self.psum_min = torch.zeros([1])
        self.psum_max = torch.zeros([1])
        self.count = 0
        self.len_dset = 0

        ddconfig = kwargs['ddconfig']
        # ddconfig['z_channels'] *= 3
        del self.decoder
        del self.encoder
        del self.quant_conv
        del self.post_quant_conv

        assert ddconfig["z_channels"] == 256
        self.encoder = ViTEncoder(
            image_size=(256, 256*3),
            patch_size=(256//32, 256//32),
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            channels=8)
        self.decoder = ViTDecoder(
            image_size=(256, 256*3),
            patch_size=(256//32, 256//32),
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            channels=8)

        self.quant_conv = torch.nn.Conv2d(768, 2*self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, 768, 1)

        if "mean" in ddconfig:
            print("Using mean std!!")
            self.triplane_mean = torch.Tensor(ddconfig['mean']).reshape(-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
            self.triplane_std = torch.Tensor(ddconfig['std']).reshape(-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
        else:
            self.triplane_mean = None
            self.triplane_std = None

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

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

    def encode(self, x, rollout=False):
        if rollout:
            # x = self.to3daware(self.rollout(x))
            x = self.rollout(x)
        if self.triplane_mean is not None:
            x = (x - self.triplane_mean.to(x.device)) / self.triplane_std.to(x.device)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, unrollout=False):
        # z = self.to3daware(z)
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if self.triplane_mean is not None:
            dec = dec * self.triplane_std.to(dec.device) + self.triplane_mean.to(dec.device)
        if unrollout:
            dec = self.unrollout(dec)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='train/', batch=batch)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.rollout(batch['triplane'])
        reconstructions, posterior = self(inputs, sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='val/', batch=None)
        self.log_dict(log_dict_ae)

        assert not self.norm
        reconstructions = self.unrollout(reconstructions)
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec
        batch_size = inputs.shape[0]
        for b in range(batch_size):
            if self.renderer_type == 'nerf':
                rgb_input, cur_psnr_list_input = self.render_triplane(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
                rgb, cur_psnr_list = self.render_triplane(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                    batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                )
            elif self.renderer_type == 'eg3d':
                rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                    batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
                rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                    reconstructions[b:b+1], batch['batch_rays'][b], batch['img'][b],
                )
            else:
                raise NotImplementedError

            cur_psnr_list_rec = []
            for i in range(rgb.shape[0]):
                cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

            rgb_input = to8b(rgb_input.detach().cpu().numpy())
            rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
            rgb = to8b(rgb.detach().cpu().numpy())
            
            if b % 4 == 0 and batch_idx < 10:
                rgb_all = np.concatenate([rgb_gt[1], rgb_input[1], rgb[1]], 1)
                self.logger.experiment.log({
                    "val/vis": [wandb.Image(rgb_all)]
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
        reconstructions, posterior = self(inputs, sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, prefix='test/', batch=None)
        self.log_dict(log_dict_ae)

        batch_size = inputs.shape[0]
        psnr_list = [] # between rec and gt
        psnr_input_list = [] # between input and gt
        psnr_rec_list = [] # between input and rec

        z = posterior.mode()
        colorize_z = self.to_rgb(z)[0]
        colorize_triplane_input = self.to_rgb_triplane(inputs)[0]
        colorize_triplane_output = self.to_rgb_triplane(reconstructions)[0]

        import os
        import random
        import string
        # z_np = z.detach().cpu().numpy()
        z_np = inputs.detach().cpu().numpy()
        fname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)) + '.npy'
        with open(os.path.join('/mnt/lustre/hongfangzhou.p/AE3D/tmp', fname), 'wb') as f:
            np.save(f, z_np)

        # colorize_triplane_rollout_3daware = self.to_rgb_3daware(self.to3daware(inputs))[0]
        # res = inputs.shape[1]
        # colorize_triplane_rollout_3daware_1 = self.to_rgb_triplane(self.to3daware(inputs)[:,res:2*res])[0]
        # colorize_triplane_rollout_3daware_2 = self.to_rgb_triplane(self.to3daware(inputs)[:,2*res:3*res])[0]
        # if batch_idx < 10:
        #     imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_z_{}.png".format(batch_idx)), colorize_z)
        #     imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_{}.png".format(batch_idx)), colorize_triplane_input)
        #     imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_output_{}.png".format(batch_idx)), colorize_triplane_output)
            # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}.png".format(batch_idx)), colorize_triplane_rollout_3daware)
            # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}_1.png".format(batch_idx)), colorize_triplane_rollout_3daware_1)
            # imageio.imwrite(os.path.join(self.logger.log_dir, "colorize_input_3daware_{}_2.png".format(batch_idx)), colorize_triplane_rollout_3daware_2)

        np_z = z.detach().cpu().numpy()
        # with open(os.path.join(self.logger.log_dir, "latent_{}.npz".format(batch_idx)), 'wb') as f:
        #     np.save(f, np_z)

        self.latent_list.append(np_z)

        reconstructions = self.unrollout(reconstructions)

        if self.psum.device != z.device:
            self.psum = self.psum.to(z.device)
            self.psum_sq = self.psum_sq.to(z.device)
            self.psum_min = self.psum_min.to(z.device)
            self.psum_max = self.psum_max.to(z.device)
        self.psum += z.sum()
        self.psum_sq += (z ** 2).sum()
        self.psum_min += z.reshape(-1).min(-1)[0]
        self.psum_max += z.reshape(-1).max(-1)[0]
        assert len(z.shape) == 4
        self.count += z.shape[0] * z.shape[1] * z.shape[2] * z.shape[3]
        self.len_dset += 1

        if self.norm:
            assert NotImplementedError
        else:
            reconstructions_unnormalize = reconstructions

        if True:
            for b in range(batch_size):
                if self.renderer_type == 'nerf':
                    rgb_input, cur_psnr_list_input = self.render_triplane(
                        batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                        batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                    )
                    rgb, cur_psnr_list = self.render_triplane(
                        reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img_flat'][b],
                        batch['near'][b].unsqueeze(-1), batch['far'][b].unsqueeze(-1)
                    )
                elif self.renderer_type == 'eg3d':
                    rgb_input, cur_psnr_list_input = self.render_triplane_eg3d_decoder(
                        batch['triplane_ori'][b:b+1], batch['batch_rays'][b], batch['img'][b],
                    )
                    rgb, cur_psnr_list = self.render_triplane_eg3d_decoder(
                        reconstructions_unnormalize[b:b+1], batch['batch_rays'][b], batch['img'][b],
                    )
                else:
                    raise NotImplementedError

                cur_psnr_list_rec = []
                for i in range(rgb.shape[0]):
                    cur_psnr_list_rec.append(mse2psnr(img2mse(rgb_input[i], rgb[i])))

                rgb_input = to8b(rgb_input.detach().cpu().numpy())
                rgb_gt = to8b(batch['img'][b].detach().cpu().numpy())
                rgb = to8b(rgb.detach().cpu().numpy())
                
                # if batch_idx < 10:
                #     imageio.imwrite(os.path.join(self.logger.log_dir, "{}_{}_input.png".format(batch_idx, b)), rgb_input[1])
                #     imageio.imwrite(os.path.join(self.logger.log_dir, "{}_{}_rec.png".format(batch_idx, b)), rgb[1])
                #     imageio.imwrite(os.path.join(self.logger.log_dir, "{}_{}_gt.png".format(batch_idx, b)), rgb_gt[1])

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
