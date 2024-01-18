import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# TriPlane Utils
class MipRayMarcher2(nn.Module):
    def __init__(self):
        super().__init__()

    def run_forward(self, colors, densities, depths, rendering_options):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2


        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        density_delta = densities_mid * deltas

        alpha = 1 - torch.exp(-density_delta)

        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        # composite_depth = torch.sum(weights * depths_mid, -2) / weight_total
        composite_depth = torch.sum(weights * depths_mid, -2)

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        # composite_depth = torch.nan_to_num(composite_depth, 0.)
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights

    def forward(self, colors, densities, depths, rendering_options):
        composite_rgb, composite_depth, weights = self.run_forward(colors, densities, depths, rendering_options)

        return composite_rgb, composite_depth, weights

def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def torch_dot(x: torch.Tensor, y: torch.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)

def get_ray_limits_box(rays_o: torch.Tensor, rays_d: torch.Tensor, box_side_length):
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)


    bb_min = [-1*(box_side_length/2), -1*(box_side_length/2), -1*(box_side_length/2)]
    bb_max = [1*(box_side_length/2), 1*(box_side_length/2), 1*(box_side_length/2)]
    bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)

    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()

    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]

    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]
    tymax = (bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)

    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]
    tzmax = (bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)

    # Mark invalid.
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -2

    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)

def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.
    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    
    # # ORIGINAL
    # N, M, C = coordinates.shape
    # xy_coords = coordinates[..., [0, 1]]
    # xz_coords = coordinates[..., [0, 2]]
    # zx_coords = coordinates[..., [2, 0]]
    # return torch.stack([xy_coords, xz_coords, zx_coords], dim=1).reshape(N*3, M, 2)

    # FIXED
    N, M, _ = coordinates.shape
    xy_coords = coordinates[..., [0, 1]]
    yz_coords = coordinates[..., [1, 2]]
    zx_coords = coordinates[..., [2, 0]]
    return torch.stack([xy_coords, yz_coords, zx_coords], dim=1).reshape(N*3, M, 2)

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)

    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

class FullyConnectedLayer(nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        # self.weight = torch.nn.Parameter(torch.full([out_features, in_features], np.float32(0)))
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


def positional_encoding(positions, freqs):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

# class TriPlane_Decoder(nn.Module):
#     def __init__(self, dim=12, width=128):
#         super().__init__()
#         self.net = torch.nn.Sequential(
#             FullyConnectedLayer(dim, width),
#             torch.nn.Softplus(),
#             FullyConnectedLayer(width, width),
#             torch.nn.Softplus(),
#             FullyConnectedLayer(width, 1 + 3)
#         )

#     def forward(self, sampled_features, viewdir):
#         sampled_features = sampled_features.mean(1)
#         x = sampled_features

#         N, M, C = x.shape
#         x = x.view(N*M, C)

#         x = self.net(x)
#         x = x.view(N, M, -1)
#         rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
#         sigma = x[..., 0:1]
#         return {'rgb': rgb, 'sigma': sigma}

class TriPlane_Decoder(nn.Module):
    def __init__(self, dim=12, width=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(dim, width),
            torch.nn.Softplus(),
            FullyConnectedLayer(width, width),
            torch.nn.Softplus(),
            FullyConnectedLayer(width, width),
            torch.nn.Softplus(),
            FullyConnectedLayer(width, width),
            torch.nn.Softplus(),
            FullyConnectedLayer(width, 1 + 3)
        )

    # def forward(self, sampled_features, viewdir):
    #     #ipdb.set_trace()
    #     sampled_features = sampled_features.mean(1)
    #     x = sampled_features

    #     N, M, C = x.shape
    #     x = x.view(N*M, C)

    #     x = self.net(x)
    #     x = x.view(N, M, -1)
    #     rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
    #     sigma = x[..., 0:1]
    #     return {'rgb': rgb, 'sigma': sigma}
    def forward(self, sampled_features, viewdir):
        M = sampled_features.shape[-2]
        batch_size = 256 * 256
        num_batches = M // batch_size
        if num_batches * batch_size < M:
            num_batches += 1
        res = {
            'rgb': [],
            'sigma': [],
        }

        for b in range(num_batches):
            p = b * batch_size
            b_sampled_features = sampled_features[:, :, p:p+batch_size]
            b_res = self._forward(b_sampled_features)
            res['rgb'].append(b_res['rgb'])
            res['sigma'].append(b_res['sigma'])
        res['rgb'] = torch.cat(res['rgb'], -2)
        res['sigma'] = torch.cat(res['sigma'], -2)

        return res

    def _forward(self, sampled_features):
        # N, _, M, C = sampled_features.shape
        sampled_features = sampled_features.mean(1)
        x = sampled_features
        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]

        # assert self.sigma_dim + self.c_dim == C
        # sigma_features = sampled_features[..., :self.sigma_dim]
        # rgb_features = sampled_features[..., -self.c_dim:]
        # sigma_features = sigma_features.permute(0, 2, 1, 3).reshape(N * M, self.sigma_dim * 3)
        # rgb_features = rgb_features.permute(0, 2, 1, 3).reshape(N * M, self.c_dim * 3)

        # x = torch.cat([self.sigmanet(sigma_features), self.rgbnet(rgb_features)], -1)
        # x = x.view(N, M, -1)
        # rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        # sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

class TriPlane_Decoder_PE(nn.Module):
    def __init__(self, dim=12, width=128, viewpe=2, feape=2):
        super().__init__()
        assert viewpe > 0 and feape > 0
        self.viewpe = viewpe
        self.feape = feape
        # self.densitynet = torch.nn.Sequential(
        #     FullyConnectedLayer(dim + 2*feape*dim, width),
        #     torch.nn.Softplus()
        # )
        # self.densityout = FullyConnectedLayer(width, 1)
        # self.rgbnet = torch.nn.Sequential(
        #     FullyConnectedLayer(width + 3 + 2 * viewpe * 3, width),
        #     torch.nn.Softplus(),
        #     FullyConnectedLayer(width, 3)
        # )
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(dim+2*feape*dim+3+2*viewpe*3, width),
            torch.nn.Softplus(),
            FullyConnectedLayer(width, width),
            torch.nn.Softplus(),
            FullyConnectedLayer(width, 1 + 3)
        )

    def forward(self, sampled_features,viewdir):
        sampled_features = sampled_features.mean(1)
        x = sampled_features
        N, M, C = x.shape
        x = x.view(N*M, C)
        viewdir = viewdir.view(N*M, 3)
        x_pe = positional_encoding(x, self.feape)
        viewdir_pe = positional_encoding(viewdir, self.viewpe)

        x = torch.cat([x, x_pe, viewdir, viewdir_pe], -1)
        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        # layer1 = self.densitynet(torch.cat([x, x_pe], -1))
        # sigma = self.densityout(layer1).view(N, M, 1)
        # rgb = self.rgbnet(torch.cat([layer1, viewdir, viewdir_pe], -1)).view(N, M, -1)
        # rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001
        return {'rgb': rgb, 'sigma': sigma}

class TriPlane_Decoder_Decompose(nn.Module):
    def __init__(self, sigma_dim=12, c_dim=12, width=128):
        super().__init__()
        self.rgbnet = torch.nn.Sequential(
            FullyConnectedLayer(c_dim * 3, width),
            torch.nn.Softplus(),
            FullyConnectedLayer(width, width),
            torch.nn.Softplus(),
            FullyConnectedLayer(width, 3)
        )
        self.sigmanet = torch.nn.Sequential(
            FullyConnectedLayer(sigma_dim * 3, width),
            torch.nn.Softplus(),
            FullyConnectedLayer(width, width),
            torch.nn.Softplus(),
            FullyConnectedLayer(width, 1)
        )
        self.sigma_dim = sigma_dim
        self.c_dim = c_dim

    def forward(self, sampled_features, viewdir):
        M = sampled_features.shape[-2]
        batch_size = 256 * 256
        num_batches = M // batch_size
        if num_batches * batch_size < M:
            num_batches += 1
        res = {
            'rgb': [],
            'sigma': [],
        }

        for b in range(num_batches):
            p = b * batch_size
            b_sampled_features = sampled_features[:, :, p:p+batch_size]
            b_res = self._forward(b_sampled_features)
            res['rgb'].append(b_res['rgb'])
            res['sigma'].append(b_res['sigma'])
        res['rgb'] = torch.cat(res['rgb'], -2)
        res['sigma'] = torch.cat(res['sigma'], -2)

        return res

    def _forward(self, sampled_features):
        N, _, M, C = sampled_features.shape
        assert self.sigma_dim + self.c_dim == C
        sigma_features = sampled_features[..., :self.sigma_dim]
        rgb_features = sampled_features[..., -self.c_dim:]
        sigma_features = sigma_features.permute(0, 2, 1, 3).reshape(N * M, self.sigma_dim * 3)
        rgb_features = rgb_features.permute(0, 2, 1, 3).reshape(N * M, self.c_dim * 3)

        x = torch.cat([self.sigmanet(sigma_features), self.rgbnet(rgb_features)], -1)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

class Renderer_TriPlane(nn.Module):
    # def __init__(self, rgbnet_dim=18, rgbnet_width=128, viewpe=0, feape=0):
    #     super(Renderer_TriPlane, self).__init__()
    #     if viewpe > 0 or feape > 0:
    #         self.decoder = TriPlane_Decoder_PE(dim=rgbnet_dim//3, width=rgbnet_width, viewpe=viewpe, feape=feape)
    #     else:
    #         self.decoder = TriPlane_Decoder(dim=rgbnet_dim//3, width=rgbnet_width)
    #     self.ray_marcher = MipRayMarcher2()
    #     self.plane_axes = generate_planes()
    
    def __init__(self, rgbnet_dim=18, rgbnet_width=128, viewpe=0, feape=0, sigma_dim=0, c_dim=0):
        super(Renderer_TriPlane, self).__init__()
        if viewpe > 0 and feape > 0:
            self.decoder = TriPlane_Decoder_PE(dim=rgbnet_dim//3, width=rgbnet_width, viewpe=viewpe, feape=feape)
        elif sigma_dim > 0 and c_dim > 0:
            self.decoder = TriPlane_Decoder_Decompose(sigma_dim=sigma_dim, c_dim=c_dim, width=rgbnet_width)
        else:
            self.decoder = TriPlane_Decoder(dim=rgbnet_dim, width=rgbnet_width)
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()

    def forward(self, planes, ray_origins, ray_directions, rendering_options, whole_img=False, tvloss=False):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        ray_start, ray_end = get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
        is_ray_valid = ray_end > ray_start
        if torch.any(is_ray_valid).item():
            ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
            ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'],
                                               rendering_options['det'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)


        out = self.run_model(planes, self.decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance, rendering_options['det'])

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, self.decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)


        if tvloss:
            initial_coordinates = torch.rand((batch_size, 1000, 3), device=planes.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * 0.004
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            projected_coordinates = project_onto_planes(self.plane_axes, all_coordinates).unsqueeze(1)
            N, n_planes, C, H, W = planes.shape
            _, M, _ = all_coordinates.shape
            planes = planes.view(N*n_planes, C, H, W)
            output_features = torch.nn.functional.grid_sample(planes, projected_coordinates.float(), mode='bilinear', padding_mode='zeros', align_corners=False).permute(0, 3, 2, 1).reshape(batch_size, n_planes, M, C)
            sigma = self.decoder(output_features)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]
            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed)
        else:
            TVloss = None

        # return rgb_final, depth_final, weights.sum(2)
        if whole_img:
            H = W = int(ray_origins.shape[1] ** 0.5)
            rgb_final = rgb_final.permute(0, 2, 1).reshape(-1, 3, H, W).contiguous()
            depth_final = depth_final.permute(0, 2, 1).reshape(-1, 1, H, W).contiguous()
            depth_final = (depth_final - depth_final.min()) / (depth_final.max() - depth_final.min())
            depth_final = depth_final.repeat(1, 3, 1, 1)
            # rgb_final = torch.clip(rgb_final, min=0, max=1)
            rgb_final = (rgb_final + 1) / 2.
            weights = weights.sum(2).reshape(rgb_final.shape[0], rgb_final.shape[2], rgb_final.shape[3])
            return {
                'rgb_marched': rgb_final,
                'depth_final': depth_final,
                'weights': weights,
                'tvloss': TVloss,
            }
        else:
            rgb_final = (rgb_final + 1) / 2.
            return {
                'rgb_marched': rgb_final,
                'depth_final': depth_final,
                'tvloss': TVloss,
            }

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])

        out = decoder(sampled_features, sample_directions)
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False, det=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                if det:
                    depths_coarse += 0.5 * depth_delta[..., None]
                else:
                    depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                if det:
                    depths_coarse += 0.5 * depth_delta
                else:
                    depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance, det=False):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance, det=det).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples
