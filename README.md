# 3D AIGC Triplane Latent Diffusion

## Requirements
See `environment.yml`
```
conda activate /mnt/petrelfs/share_data/lvsizhe/pt2-ae3d-env
```

## Usage

First train latent VAE, then train diffusion.

### VAE
```
python -u main.py --config configs/triplane_vae_conv_groupconv_shapenet_sigma2_ch6_tvloss_fixdecoder_kl5e-4_res32ch8.yaml
```

### Diffusion
```
python -u main.py --config ddpm_configs/triplane_ddpm_crossattention_groupconv_shapenet_sigma2_ch6_tvloss_fixdecoder_kl1e-5_scale_std_x3.yaml
```

### Calculate FID
We use [pytorch-fid](https://github.com/mseitzer/pytorch-fid) to calculate FID. I should have installed it in our shared environment.
```
python -u main.py --config ${config yaml file} --mode ${path to ckpt} --test_mode fid --test_tag ${any tag} --gpu 1
cd log/${config file name}/lightning_logs/version_0/
python -m pytorch_fid --save-stats FID_${any tag} fid_${any tag}.npz
python -m pytorch_fid fid_${any tag}.npz /mnt/petrelfs/share_data/hongfangzhou.p/shapenet/fid/1view_100.npz
```
