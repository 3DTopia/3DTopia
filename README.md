<p align="center">
    <picture>
    <img alt="logo" src="assets/3dtopia.jpeg" width="20%">
    </picture>
</p>
<div align="center">
  <h1>3DTopia</h1>
  A two-stage text-to-3D generation model. The first stage uses diffusion model to quickly generate candidates. The second stage refines the assets chosen from the first stage.

https://github.com/3DTopia/3DTopia/assets/23376858/c9716cf0-6e61-4983-82b2-2e8f579bd46c

</div>

## 1. Quick Start

### 1.1 Install Environment for this Repository
```bash
conda env create -f environment.yml
```

### 1.2 Install Second Stage Refiner
Please refer to [threefiner](https://github.com/3DTopia/threefiner) to install our second stage mesh refiner.

### 1.3 Download Checkpoints
Download checkpoint from [huggingface](https://huggingface.co/hongfz16/3DTopia). Put the checkpoint `3dtopia_diffusion_state_dict.ckpt` under the folder `checkpoints`.

## 2. Inference

### 2.1 First Stage
Run the following command to sample `a robot` as the first stage. Results will be located under the folder `results`.
```bash
python -u sample_stage1.py --text "a robot" --samples 1 --seed 0
```

Other arguments:
- `--test_folder` controls which subfolder to put all the results;
- `--seed` will fix random seeds; `--sampler` can be set to `ddim` for DDIM sampling (By default, we use 1000 steps DDPM sampling);
- `--steps` controls sampling steps only for DDIM;
- `--samples` controls number of samples;
- `--text` is the input text;
- `--no_video` and `--no_mcubes` suppress rendering multi-view videos and marching cubes, which are by-default enabled;
- `--mcubes_res` controls the resolution of the 3D volumn sampled for marching cubes; One can lower this resolution to save graphics memory;
- `--render_res` controls the resolution of the rendered video;

### 2.2 Second Stage
There are two steps as the second stage refinement. Here is a simple example. Please refer to [threefiner](https://github.com/3DTopia/threefiner) for more detailed usage.
```bash
# step 1
threefiner sd --mesh results/default/stage1/a_robot_0_0.ply --prompt "a robot" --text_dir --front_dir='-y' --outdir results/default/stage2/ --save a_robot_1_0_sd.glb
# step 2
threefiner if2 --mesh results/default/stage2/a_robot_0_0_sd.glb --prompt "a robot" --outdir results/default/stage2/ --save a_robot_1_0_if2.glb
```
The resulting mesh can be found at `results/default/stage2/a_robot_1_0_if2.glb`

## 3. Acknowledgement
We thank the community for building and open-sourcing the foundation of this work. Specifically, we want to thank [EG3D](https://github.com/NVlabs/eg3d), [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for their codes. We also want to thank [Objaverse](https://objaverse.allenai.org) for the wonderful dataset.
