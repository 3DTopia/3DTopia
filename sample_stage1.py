import os
import cv2
import json
import torch
import mcubes
import trimesh
import argparse
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import pytorch_lightning as pl
from omegaconf import OmegaConf

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from utility.initialize import instantiate_from_config, get_obj_from_str
from utility.triplane_renderer.eg3d_renderer import sample_from_planes, generate_planes
from utility.triplane_renderer.renderer import get_rays, to8b
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def add_text(rgb, caption):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    gap = 30
    org = (gap, gap)
    # fontScale
    fontScale = 0.6
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 1
    break_caption = []
    for i in range(len(caption) // 30 + 1):
        break_caption_i = caption[i*30:(i+1)*30]
        break_caption.append(break_caption_i)
    for i, bci in enumerate(break_caption):
        cv2.putText(rgb, bci, (gap, gap*(i+1)), font, fontScale, color, thickness, cv2.LINE_AA)
    return rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/default.yaml')
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--test_folder", type=str, default="stage1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sampler", type=str, default="ddpm")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--text", nargs='+', default='a robot')
    parser.add_argument("--text_file", type=str, default=None)
    parser.add_argument("--no_video", action='store_true', default=False)
    parser.add_argument("--render_res", type=int, default=128)
    parser.add_argument("--no_mcubes", action='store_true', default=False)
    parser.add_argument("--mcubes_res", type=int, default=128)
    parser.add_argument("--cfg_scale", type=float, default=1)
    args = parser.parse_args()

    if args.text is not None:
        text = [' '.join(args.text),]
    elif args.text_file is not None:
        if args.text_file.endswith('.json'):
            with open(args.text_file, 'r') as f:
                json_file = json.load(f)
                text = json_file
                text = [l.strip('.') for l in text]
        else:
            with open(args.text_file, 'r') as f:
                text = f.readlines()
                text = [l.strip() for l in text]
    else:
        raise NotImplementedError

    print(text)

    configs = OmegaConf.load(args.config)
    if args.seed is not None:
        pl.seed_everything(args.seed)

    log_dir = os.path.join('results', args.config.split('/')[-1].split('.')[0], args.test_folder)
    os.makedirs(log_dir, exist_ok=True)

    if args.ckpt == None:
        ckpt = hf_hub_download(repo_id="hongfz16/3DTopia", filename="model.safetensors")
    else:
        ckpt = args.ckpt

    if ckpt.endswith(".ckpt"):
        model = get_obj_from_str(configs.model["target"]).load_from_checkpoint(ckpt, map_location='cpu', strict=False, **configs.model.params)
    elif ckpt.endswith(".safetensors"):
        model = get_obj_from_str(configs.model["target"])(**configs.model.params)
        model_ckpt = load_file(ckpt)
        model.load_state_dict(model_ckpt)
    else:
        raise NotImplementedError
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    class DummySampler:
        def __init__(self, model):
            self.model = model

        def sample(self, S, batch_size, shape, verbose, conditioning=None, *args, **kwargs):
            return self.model.sample(
                conditioning, batch_size, shape=[batch_size, ] + shape, *args, **kwargs
            ), None

    if args.sampler == 'dpm':
        raise NotImplementedError
        # sampler = DPMSolverSampler(model)
    elif args.sampler == 'plms':
        raise NotImplementedError
        # sampler = PLMSSampler(model)
    elif args.sampler == 'ddim':
        sampler = DDIMSampler(model)
    elif args.sampler == 'ddpm':
        sampler = DummySampler(model)
    else:
        raise NotImplementedError

    img_size = configs.model.params.unet_config.params.image_size
    channels = configs.model.params.unet_config.params.in_channels
    shape = [channels, img_size, img_size * 3]
    plane_axes = generate_planes()

    pose_folder = 'assets/sample_data/pose'
    poses_fname = sorted([os.path.join(pose_folder, f) for f in os.listdir(pose_folder)])
    batch_rays_list = []
    H = args.render_res
    ratio = 512 // H
    for p in poses_fname:
        c2w = np.loadtxt(p).reshape(4, 4)
        c2w[:3, 3] *= 2.2
        c2w = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]) @ c2w

        k = np.array([
            [560 / ratio, 0, H * 0.5],
            [0, 560 / ratio, H * 0.5],
            [0, 0, 1]
        ])

        rays_o, rays_d = get_rays(H, H, torch.Tensor(k), torch.Tensor(c2w[:3, :4]))
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, H-1, H), indexing='ij'), -1)
        coords = torch.reshape(coords, [-1,2]).long()
        rays_o = rays_o[coords[:, 0], coords[:, 1]]
        rays_d = rays_d[coords[:, 0], coords[:, 1]]
        batch_rays = torch.stack([rays_o, rays_d], 0)
        batch_rays_list.append(batch_rays)
    batch_rays_list = torch.stack(batch_rays_list, 0)

    for text_idx, text_i in enumerate(text):
        text_connect = '_'.join(text_i.split(' '))
        for s in range(args.samples):
            batch_size = args.batch_size
            with torch.no_grad():
                # with model.ema_scope():
                noise = None
                c = model.get_learned_conditioning([text_i])
                unconditional_c = torch.zeros_like(c)
                if args.cfg_scale != 1:
                    assert args.sampler == 'ddim'
                    sample, _ = sampler.sample(
                        S=args.steps,
                        batch_size=batch_size,
                        shape=shape,
                        verbose=False,
                        x_T = noise,
                        conditioning = c.repeat(batch_size, 1, 1),
                        unconditional_guidance_scale=args.cfg_scale,
                        unconditional_conditioning=unconditional_c.repeat(batch_size, 1, 1)
                    )
                else:
                    sample, _ = sampler.sample(
                        S=args.steps,
                        batch_size=batch_size,
                        shape=shape,
                        verbose=False,
                        x_T = noise,
                        conditioning = c.repeat(batch_size, 1, 1),
                    )
                decode_res = model.decode_first_stage(sample)

                for b in range(batch_size):
                    def render_img(v):
                        rgb_sample, _ = model.first_stage_model.render_triplane_eg3d_decoder(
                            decode_res[b:b+1], batch_rays_list[v:v+1].to(device), torch.zeros(1, H, H, 3).to(device),
                        )
                        rgb_sample = to8b(rgb_sample.detach().cpu().numpy())[0]
                        rgb_sample = np.stack(
                            [rgb_sample[..., 2], rgb_sample[..., 1], rgb_sample[..., 0]], -1
                        )
                        # rgb_sample = add_text(rgb_sample, text_i)
                        return rgb_sample

                    if not args.no_mcubes:
                        # prepare volumn for marching cube
                        res = args.mcubes_res
                        c_list = torch.linspace(-1.2, 1.2, steps=res)
                        grid_x, grid_y, grid_z = torch.meshgrid(
                            c_list, c_list, c_list, indexing='ij'
                        )
                        coords = torch.stack([grid_x, grid_y, grid_z], -1).to(device)
                        plane_axes = generate_planes()
                        feats = sample_from_planes(
                            plane_axes, decode_res[b:b+1].reshape(1, 3, -1, 256, 256), coords.reshape(1, -1, 3), padding_mode='zeros', box_warp=2.4
                        )
                        fake_dirs = torch.zeros_like(coords)
                        fake_dirs[..., 0] = 1
                        out = model.first_stage_model.triplane_decoder.decoder(feats, fake_dirs)
                        u = out['sigma'].reshape(res, res, res).detach().cpu().numpy()
                        del out

                        # marching cube
                        vertices, triangles = mcubes.marching_cubes(u, 10)
                        min_bound = np.array([-1.2, -1.2, -1.2])
                        max_bound = np.array([1.2, 1.2, 1.2])
                        vertices = vertices / (res - 1) * (max_bound - min_bound)[None, :] + min_bound[None, :]
                        pt_vertices = torch.from_numpy(vertices).to(device)

                        # extract vertices color
                        res_triplane = 256
                        render_kwargs = {
                            'depth_resolution': 128,
                            'disparity_space_sampling': False,
                            'box_warp': 2.4,
                            'depth_resolution_importance': 128,
                            'clamp_mode': 'softplus',
                            'white_back': True,
                            'det': True
                        }
                        rays_o_list = [
                            np.array([0, 0, 2]),
                            np.array([0, 0, -2]),
                            np.array([0, 2, 0]),
                            np.array([0, -2, 0]),
                            np.array([2, 0, 0]),
                            np.array([-2, 0, 0]),
                        ]
                        rgb_final = None
                        diff_final = None
                        for rays_o in tqdm(rays_o_list):
                            rays_o = torch.from_numpy(rays_o.reshape(1, 3)).repeat(vertices.shape[0], 1).float().to(device)
                            rays_d = pt_vertices.reshape(-1, 3) - rays_o
                            rays_d = rays_d / torch.norm(rays_d, dim=-1).reshape(-1, 1)
                            dist = torch.norm(pt_vertices.reshape(-1, 3) - rays_o, dim=-1).cpu().numpy().reshape(-1)

                            render_out = model.first_stage_model.triplane_decoder(
                                decode_res[b:b+1].reshape(1, 3, -1, res_triplane, res_triplane),
                                rays_o.unsqueeze(0), rays_d.unsqueeze(0), render_kwargs,
                                whole_img=False, tvloss=False
                            )
                            rgb = render_out['rgb_marched'].reshape(-1, 3).detach().cpu().numpy()
                            depth = render_out['depth_final'].reshape(-1).detach().cpu().numpy()
                            depth_diff = np.abs(dist - depth)

                            if rgb_final is None:
                                rgb_final = rgb.copy()
                                diff_final = depth_diff.copy()

                            else:
                                ind = diff_final > depth_diff
                                rgb_final[ind] = rgb[ind]
                                diff_final[ind] = depth_diff[ind]


                        # bgr to rgb
                        rgb_final = np.stack([
                            rgb_final[:, 2], rgb_final[:, 1], rgb_final[:, 0]
                        ], -1)

                        # export to ply
                        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=(rgb_final * 255).astype(np.uint8))
                        trimesh.exchange.export.export_mesh(mesh, os.path.join(log_dir, f"{text_connect}_{s}_{b}.ply"), file_type='ply')

                    if not args.no_video:
                        view_num = len(batch_rays_list)
                        video_list = []
                        for v in tqdm(range(view_num//4, view_num//4 * 3, 2)):
                            rgb_sample = render_img(v)
                            video_list.append(rgb_sample)
                        imageio.mimwrite(os.path.join(log_dir, "{}_{}_{}.mp4".format(text_connect, s, b)), np.stack(video_list, 0))
                    else:
                        rgb_sample = render_img(104)
                        imageio.imwrite(os.path.join(log_dir, "{}_{}_{}.jpg".format(text_connect, s, b)), rgb_sample)

if __name__ == '__main__':
    main()
