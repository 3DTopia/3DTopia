# 3DTopia

## Requirements

### Install Environment for this Repository
See `environment.yml`
```
conda env create -f environment.yml
```

### Install Second Stage Refiner
See #TODO

### Download Checkpoints
#TODO
Please put the checkpoint `3dtopia_diffusion_state_dict.ckpt` under the folder `checkpoints`

## Inference

### First Stage
Execute the following command to sample `a robot` as the first stage. Results will be located under the folder `results`
```
python -u sample_stage1.py --text "a robot" --samples 1 --seed 0
```

Other arguments:
- `--test_folder` controls which subfolder to put all the results;
- `--seed` will fix random seeds; `--sampler` can be set to `ddim` for DDIM sampling (By default, we use 1000 steps DDPM sampling);
- `--steps` controls sampling steps only for DDIM;
- `--samples` controls number of samples;
- `--text` is the input text;
- `--no_video` and `--no_mcubes` surpress rendering multi-view videos and marching cubes, which are by-default enabled;
- `--mcubes_res` controls the resolution of the 3D volumn sampled for marching cubes; One can lower this resolution to save graphics memory;
- `--render_res` controls the resolution of the rendered video;

### Second Stage
```
threefiner if2 --mesh results/default/stage1/a_robot_0_0.ply --prompt "a robot"
```
See more examples at #TODO
