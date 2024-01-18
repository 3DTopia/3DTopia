import os
import torch
import argparse
import mcubes
import trimesh
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from utility.initialize import instantiate_from_config, get_obj_from_str
from utility.triplane_renderer.eg3d_renderer import sample_from_planes, generate_planes

# load model
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, required=True)
parser.add_argument("--ckpt", type=str, default=None, required=True)
args = parser.parse_args()
configs = OmegaConf.load(args.config)
device = 'cuda'
vae = get_obj_from_str(configs.model.params.first_stage_config['target'])(**configs.model.params.first_stage_config['params'])
vae = vae.to(device)
vae.eval()

model = get_obj_from_str(configs.model["target"]).load_from_checkpoint(args.ckpt, map_location='cpu', strict=False, **configs.model.params)
model = model.to(device)

def extract_mesh(triplane_fname, save_name=None):
    latent = torch.from_numpy(np.load(triplane_fname)).to(device)
    with torch.no_grad():
        with model.ema_scope():
            triplane = model.decode_first_stage(latent)

    # prepare volumn for marching cube
    res = 128
    c_list = torch.linspace(-1.2, 1.2, steps=res)
    grid_x, grid_y, grid_z = torch.meshgrid(
        c_list, c_list, c_list, indexing='ij'
    )
    coords = torch.stack([grid_x, grid_y, grid_z], -1).to(device) # 256x256x256x3
    plane_axes = generate_planes()
    feats = sample_from_planes(
        plane_axes, triplane.reshape(1, 3, -1, 256, 256), coords.reshape(1, -1, 3), padding_mode='zeros', box_warp=2.4
    )
    fake_dirs = torch.zeros_like(coords)
    fake_dirs[..., 0] = 1
    with torch.no_grad():
        out = vae.triplane_decoder.decoder(feats, fake_dirs)
    u = out['sigma'].reshape(res, res, res).detach().cpu().numpy()
    del out

    # marching cube
    vertices, triangles = mcubes.marching_cubes(u, 8)
    min_bound = np.array([-1.2, -1.2, -1.2])
    max_bound = np.array([1.2, 1.2, 1.2])
    vertices = vertices / (res - 1) * (max_bound - min_bound)[None, :] + min_bound[None, :]
    pt_vertices = torch.from_numpy(vertices).to(device)

    # extract vertices color
    res_triplane = 256
    # rays_d = torch.from_numpy(-vertices / np.sqrt((vertices ** 2).sum(-1)).reshape(-1, 1)).to(device).unsqueeze(0)
    # rays_o = -rays_d * 2.0
    render_kwargs = {
        'depth_resolution': 128,
        'disparity_space_sampling': False,
        'box_warp': 2.4,
        'depth_resolution_importance': 128,
        'clamp_mode': 'softplus',
        'white_back': True,
        'det': True
    }
    # render_out = vae.triplane_decoder(triplane.reshape(1, 3, -1, res_triplane, res_triplane), rays_o, rays_d, render_kwargs, whole_img=False, tvloss=False)
    # rgb = render_out['rgb_marched'].reshape(-1, 3).detach().cpu().numpy()
    # rgb = (rgb * 255).astype(np.uint8)
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

        # batch_size = 2**14
        # batch_num = (rays_o.shape[0] // batch_size) + 1
        # rgb_list = []
        # depth_diff_list = []
        # for b in range(batch_num):
        # cur_rays_o = rays_o[b * batch_size: (b + 1) * batch_size]
        # cur_rays_d = rays_d[b * batch_size: (b + 1) * batch_size]
        with torch.no_grad():
            render_out = vae.triplane_decoder(triplane.reshape(1, 3, -1, res_triplane, res_triplane),
                                            rays_o.unsqueeze(0), rays_d.unsqueeze(0), render_kwargs,
                                            whole_img=False, tvloss=False)
        rgb = render_out['rgb_marched'].reshape(-1, 3).detach().cpu().numpy()
        depth = render_out['depth_final'].reshape(-1).detach().cpu().numpy()
        depth_diff = np.abs(dist - depth)

            # rgb_list.append(rgb)
            # depth_diff_list.append(depth_diff)
    
            # del render_out
            # torch.cuda.empty_cache()

        # rgb = np.concatenate(rgb_list, 0)
        # depth_diff = np.concatenate(depth_diff_list, 0)

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
    if save_name:
        trimesh.exchange.export.export_mesh(mesh, save_name, file_type='ply')
    else:
        trimesh.exchange.export.export_mesh(mesh, triplane_fname[:-4] + '.ply', file_type='ply')

# load triplane
# fname = 'log/diff_res32ch8_preprocess_ca_text/sample_mesh_1/sample_16_0.npy'
# u = np.load(fname)
# triplane_fname = 'log/diff_res32ch8_preprocess_ca_text/sample_mesh_1/triplane_16_0.npy'
# folder = 'log/diff_res32ch8_preprocess_ca_text/sample_mesh_opt'
# folder = 'log/diff_res32ch8_preprocess_ca_text/sample_mesh_opt_simple'
folder = '/mnt/lustre/hongfangzhou.p/AE3D/log/diff_res32ch8_preprocess_ca_text_new_triplane_96_full_openaimodel_only_cap3d_high_quality_7w/sample_demo_424_prompts_for_demo_30_60_10'
save_folder = folder + '_extract_mesh'
os.makedirs(save_folder, exist_ok=True)
fnames = [f.replace('_sample', 'triplane').replace('mp4', 'npy') for f in os.listdir(folder) if f.startswith('_')]
prompts = [l.strip() for l in open('test/prompts_for_demo_2.txt', 'r').readlines()][30:60]
# fnames = [os.path.join(folder, f) for f in os.listdir(folder) if (f.startswith('triplane') and f.endswith('.npy'))]
fnames = sorted(fnames)

def extract_number(s):
    return int(s.split('_')[-2])

def extract_id(s):
    return s.split('_')[-1].split('.')[0]

for fname in fnames:
    try:
        print(fname)
        extract_mesh(os.path.join(folder, fname), os.path.join(save_folder, prompts[extract_number(fname)].replace(' ', '_') + '_' + extract_id(fname) + '.ply'))
    except Exception as e:
        print(e)
