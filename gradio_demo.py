import os
import cv2
import time
import tyro
import json
import kiui
import tqdm
import torch
import mcubes
import trimesh
import datetime
import argparse
import subprocess
import numpy as np
import gradio as gr
import imageio.v2 as imageio
import pytorch_lightning as pl
from omegaconf import OmegaConf
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from utility.initialize import instantiate_from_config, get_obj_from_str
from utility.triplane_renderer.eg3d_renderer import sample_from_planes, generate_planes
from utility.triplane_renderer.renderer import get_rays, to8b

from threefiner.gui import GUI
from threefiner.opt import config_defaults, config_doc, check_options, Options

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

###################################### INIT STAGE 1 #########################################
config = "configs/default.yaml"
download_ckpt = "checkpoints/3dtopia_diffusion_state_dict.ckpt"
if not os.path.exists(download_ckpt):
    ckpt = hf_hub_download(repo_id="hongfz16/3DTopia", filename="model.safetensors")
else:
    ckpt = download_ckpt
configs = OmegaConf.load(config)
os.makedirs("tmp", exist_ok=True)

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
sampler = DDIMSampler(model)

img_size = configs.model.params.unet_config.params.image_size
channels = configs.model.params.unet_config.params.in_channels
shape = [channels, img_size, img_size * 3]

pose_folder = 'assets/sample_data/pose'
poses_fname = sorted([os.path.join(pose_folder, f) for f in os.listdir(pose_folder)])
batch_rays_list = []
H = 128
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
###################################### INIT STAGE 1 #########################################

###################################### INIT STAGE 2 #########################################
GRADIO_SAVE_PATH_MESH = 'gradio_output.glb'
GRADIO_SAVE_PATH_VIDEO = 'gradio_output.mp4'

# opt = tyro.cli(tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc))
opt = Options(
    mode='IF2',
    iters=400,
)

# hacks for not loading mesh at initialization
# opt.mesh = 'tmp/_2024-01-25_19:33:03.110191_if2.glb'
opt.save = GRADIO_SAVE_PATH_MESH
opt.prompt = ''
opt.text_dir = True
opt.front_dir = '+z'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gui = GUI(opt)
###################################### INIT STAGE 2 #########################################

def add_text(rgb, caption):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    gap = 10
    org = (gap, gap)
    # fontScale
    fontScale = 0.3
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

def marching_cube(b, text, global_info):
    # prepare volumn for marching cube
    res = 128
    assert 'decode_res' in global_info
    decode_res = global_info['decode_res']
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
    for rays_o in tqdm.tqdm(rays_o_list):
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
    path = os.path.join('tmp', f"{text.replace(' ', '_')}_{str(datetime.datetime.now()).replace(' ', '_')}.ply")
    trimesh.exchange.export.export_mesh(mesh, path, file_type='ply')

    del vertices, triangles, rgb_final
    torch.cuda.empty_cache()

    return path

def infer(prompt, samples, steps, scale, seed, global_info):
    prompt = prompt.replace('/', '')
    pl.seed_everything(seed)
    batch_size = samples
    with torch.no_grad():
        noise = None
        c = model.get_learned_conditioning([prompt])
        unconditional_c = torch.zeros_like(c)
        sample, _ = sampler.sample(
            S=steps,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            x_T = noise,
            conditioning = c.repeat(batch_size, 1, 1),
            unconditional_guidance_scale=scale,
            unconditional_conditioning=unconditional_c.repeat(batch_size, 1, 1)
        )
        decode_res = model.decode_first_stage(sample)

        big_video_list = []

        global_info['decode_res'] = decode_res

        for b in range(batch_size):
            def render_img(v):
                rgb_sample, _ = model.first_stage_model.render_triplane_eg3d_decoder(
                    decode_res[b:b+1], batch_rays_list[v:v+1].to(device), torch.zeros(1, H, H, 3).to(device),
                )
                rgb_sample = to8b(rgb_sample.detach().cpu().numpy())[0]
                rgb_sample = np.stack(
                    [rgb_sample[..., 2], rgb_sample[..., 1], rgb_sample[..., 0]], -1
                )
                rgb_sample = add_text(rgb_sample, str(b))
                return rgb_sample

            view_num = len(batch_rays_list)
            video_list = []
            for v in tqdm.tqdm(range(view_num//8*3, view_num//8*5, 2)):
                rgb_sample = render_img(v)
                video_list.append(rgb_sample)
            big_video_list.append(video_list)
        # if batch_size == 2:
        #     cat_video_list = [
        #         np.concatenate([big_video_list[j][i] for j in range(len(big_video_list))], 1) \
        #         for i in range(len(big_video_list[0]))
        #     ]
        # elif batch_size > 2:
        #     if batch_size == 3:
        #         big_video_list.append(
        #             [np.zeros_like(f) for f in big_video_list[0]]
        #         )
        #     cat_video_list = [
        #         np.concatenate([
        #             np.concatenate([big_video_list[0][i], big_video_list[1][i]], 1),
        #             np.concatenate([big_video_list[2][i], big_video_list[3][i]], 1),
        #         ], 0) \
        #         for i in range(len(big_video_list[0]))
        #     ]
        # else:
        #     cat_video_list = big_video_list[0]

        for _ in range(4 - batch_size):
            big_video_list.append(
                [np.zeros_like(f) + 255 for f in big_video_list[0]]
            )
        cat_video_list = [
            np.concatenate([
                np.concatenate([big_video_list[0][i], big_video_list[1][i]], 1),
                np.concatenate([big_video_list[2][i], big_video_list[3][i]], 1),
            ], 0) \
            for i in range(len(big_video_list[0]))
        ]

        path = f"tmp/{prompt.replace(' ', '_')}_{str(datetime.datetime.now()).replace(' ', '_')}.mp4"
        imageio.mimwrite(path, np.stack(cat_video_list, 0))

    return global_info, path

def infer_stage2(prompt, selection, seed, global_info, iters):
    prompt = prompt.replace('/', '')
    mesh_path = marching_cube(int(selection), prompt, global_info)
    mesh_name = mesh_path.split('/')[-1][:-4]
    # if2_cmd = f"threefiner if2 --mesh {mesh_path} --prompt \"{prompt}\" --outdir tmp --save {mesh_name}_if2.glb --text_dir --front_dir=-y"
    # print(if2_cmd)
    # subprocess.Popen(if2_cmd, shell=True).wait()
    # torch.cuda.empty_cache()
    video_path = f"tmp/{prompt.replace(' ', '_')}_{str(datetime.datetime.now()).replace(' ', '_')}.mp4"
    # render_cmd = f"kire {os.path.join('tmp', mesh_name + '_if2.glb')} --save_video {video_path} --wogui --force_cuda_rast --H 256 --W 256"
    # print(render_cmd)
    # subprocess.Popen(render_cmd, shell=True).wait()
    # torch.cuda.empty_cache()

    process_stage2(mesh_path, prompt, "down", iters, f'tmp/{mesh_name}_if2.glb', video_path)
    torch.cuda.empty_cache()

    return video_path, f'tmp/{mesh_name}_if2.glb'

def process_stage2(input_model, input_text, input_dir, iters, output_model, output_video):
    # set front facing direction (map from gradio model3D's mysterious coordinate system to OpenGL...)
    opt.text_dir = True
    if input_dir == 'front':
        opt.front_dir = '-z'
    elif input_dir == 'back':
        opt.front_dir = '+z'
    elif input_dir == 'left':
        opt.front_dir = '+x'
    elif input_dir == 'right':
        opt.front_dir = '-x'
    elif input_dir == 'up':
        opt.front_dir = '+y'
    elif input_dir == 'down':
        opt.front_dir = '-y'
    else:
        # turn off text_dir
        opt.text_dir = False
        opt.front_dir = '+z'
    
    # set mesh path
    opt.mesh = input_model

    # load mesh!
    gui.renderer = gui.renderer_class(opt, device).to(device)

    # set prompt
    gui.prompt = opt.positive_prompt + ', ' + input_text

    # train
    gui.prepare_train() # update optimizer and prompt embeddings
    for i in tqdm.trange(iters):
        gui.train_step()

    # save mesh & video
    gui.save_model(output_model)
    gui.save_model(output_video)

markdown=f'''
  # 3DTopia
  A two-stage text-to-3D generation model. The first stage uses diffusion model to quickly generate candidates. The second stage refines the assets chosen from the first stage.

  ### Usage:
  First enter prompt for a 3D object, hit "Generate 3D". Then choose one candidate from the dropdown options for the second stage refinement and hit "Start Refinement". The final mesh can be downloaded from the bottom right box.
  
  ### Runtime:
  The first stage takes 30s if generating 4 samples. The second stage takes roughly 1m30s.

  ### Useful links:
  [Github Repo](https://github.com/3DTopia/3DTopia)
'''

block = gr.Blocks()

with block:
    global_info = gr.State(dict())
    gr.Markdown(markdown)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                text = gr.Textbox(
                    label = "Enter your prompt",
                    max_lines = 1,
                    placeholder = "Enter your prompt",
                    container = False,
                )
                btn = gr.Button("Generate 3D")
            gallery = gr.Video(height=512)
            # advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")
            with gr.Row(elem_id="advanced-options"):
                with gr.Tab("Advanced options"):
                    samples = gr.Slider(label="Number of Samples", minimum=1, maximum=4, value=4, step=1)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=500, value=50, step=1)
                    scale = gr.Slider(
                        label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=2147483647,
                        step=1,
                        randomize=True,
                    )
            gr.on([text.submit, btn.click], infer, inputs=[text, samples, steps, scale, seed, global_info], outputs=[global_info, gallery])
            # advanced_button.click(
            #     None,
            #     [],
            #     text,
            # )
        with gr.Column():
            with gr.Row():
                dropdown = gr.Dropdown(
                    ['0', '1', '2', '3'], label="Choose a Candidate For Stage2", value='0'
                )
                btn_stage2 = gr.Button("Start Refinement")
            gallery = gr.Video(height=512)
            with gr.Row(elem_id="advanced-options"):
                with gr.Tab("Advanced options"):
                    # input_dir = gr.Radio(['front', 'back', 'left', 'right', 'up', 'down'], value='down', label="front-facing direction")
                    iters = gr.Slider(minimum=100, maximum=1000, step=100, value=400, label="Refine iterations")
            download = gr.File(label="Download Mesh", file_count="single", height=100)
            gr.on([btn_stage2.click], infer_stage2, inputs=[text, dropdown, seed, global_info, iters], outputs=[gallery, download])

block.launch(share=True)
