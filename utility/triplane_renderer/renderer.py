import os, sys
import imageio
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn,label, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [inputs.shape[0],-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [inputs.shape[0],-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        #embedded = torch.cat([embedded, embedded_dirs], -1)

    input_all=torch.cat([inputs_flat,embedded_dirs],-1)
    outputs_flat =  fn(input_all,label)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs




import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
class Triplane(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        
        self.plane_axis=self.generate_planes()
    def generate_planes(self):
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
                                [0, 1, 0],
                                [1, 0, 0]]], dtype=torch.float32)

    def project_onto_planes(self,planes, coordinates):
        """
        Does a projection of a 3D point onto a batch of 2D planes,
        returning 2D plane coordinates.

        Takes plane axes of shape n_planes, 3, 3
        # Takes coordinates of shape N, M, 3
        # returns projections of shape N*n_planes, M, 2
        """
        N, M, C = coordinates.shape
        n_planes, _, _ = planes.shape
        coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
        inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3).to(device=coordinates.device)
        projections = torch.bmm(coordinates, inv_planes)
        return projections[..., :2]

    def sample_from_planes(self,plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
        assert padding_mode == 'zeros'
        N, n_planes, C, H, W = plane_features.shape
        
        _, M, _ = coordinates.shape
        plane_features = plane_features.view(N*n_planes, C, H, W)

        coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds
        #ipdb.set_trace()
        coordinates = self.project_onto_planes(plane_axes, coordinates).unsqueeze(1)
        
        output_features = torch.nn.functional.grid_sample(plane_features, coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)  # xy,xz,zy
        
        return output_features
        

    def forward(self, planes, sample_coordinates,box=1):

        #ipdb.set_trace()

        return self.sample_from_planes(self.plane_axis, planes, sample_coordinates, padding_mode='zeros', box_warp=box)

def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts
def exists(val):
    return val is not None
def resize_image_to(
    image,
    target_image_size,
    clamp_range = None,
    mode = 'nearest'
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode = mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, size=256,input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,num_instance=1):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch//3
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.hidden_dim=W

        self.triplane=Triplane()
        
        #ipdb.set_trace() 
        self.tri_planes = nn.Parameter(torch.randn(num_instance, input_ch, size, size))
        nn.init.normal_(self.tri_planes, mean=0, std=0.1)
        #self.weight=nn.Parameter(torch.ones(1,3,1,input_ch))
        #ipdb.set_trace() 
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W)])

        self.softplus=nn.Softplus()

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W, 3)
        # for m in self.children():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, std=0.01)
                
        

    def forward(self, x, label):
        #ipdb.set_trace() 
        

        input_pts, input_views = torch.split(x, [int(x.shape[-1]-self.input_ch_views), self.input_ch_views], dim=-1)
        B,N,M=input_views.shape
        #ipdb.set_trace()

        # eal=resize_image_to(self.tri_planes[label],256)
        # eal=resize_image_to(eal,256)
        norm=torch.abs(self.tri_planes[label]).max(2)[0].max(2)[0].unsqueeze(-1).unsqueeze(-1)
        sample_triplane=(self.tri_planes[label]/norm).view(1,3,self.tri_planes.shape[-3]//3,self.tri_planes.shape[-2],self.tri_planes.shape[-1]).repeat(B,1,1,1,1)
        #ipdb.set_trace() 
        input_pts=(self.triplane(sample_triplane,input_pts,2.5)).mean(1).view(-1,self.tri_planes.shape[-3]//3)
        #ipdb.set_trace()
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            #ipdb.set_trace() 
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
               # ipdb.set_trace() 
                h = torch.cat([input_pts, h], -1)

        
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views.view(B*N,M)], -1)
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb.view(B,N,3), alpha.view(B,N,1)], -1)
        
        #ipdb.set_trace()
        return outputs 

class NeRF11(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,num_instance=1):
        """ 
        """
        super(NeRF11, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch//3
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.hidden_dim=W

        self.triplane=Triplane()
        
        self.weight = nn.Parameter(torch.zeros(1, 1, 256))

        self.tri_planes = nn.Parameter(torch.randn(num_instance, input_ch, 256, 256))
        #nn.init.normal_(self.tri_planes, mean=0, std=0.1)
        #self.weight=nn.Parameter(torch.ones(1,3,1,input_ch))
        #ipdb.set_trace() 
        self.label_emb = nn.Embedding(num_instance, W)
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W)])

        self.softplus=nn.Softplus()

        #self.label_feature = nn.Linear(W, W)

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W, 3)
        

    def forward(self, x,label):
        #ipdb.set_trace() 
        

        input_pts, input_views = torch.split(x, [int(x.shape[-1]-self.input_ch_views), self.input_ch_views], dim=-1)
        B,N,M=input_views.shape
        sample_triplane=self.tri_planes[label].view(B,3,self.tri_planes.shape[-3]//3,self.tri_planes.shape[-2],self.tri_planes.shape[-1])
        
        input_pts=(self.triplane(sample_triplane,input_pts,4)).mean(1).view(B,-1,self.tri_planes.shape[-3]//3)
        #ipdb.set_trace()
        label_emb=(self.weight*self.label_emb(label).unsqueeze(1)).expand(-1,N,-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            #ipdb.set_trace() 
            h = self.pts_linears[i](h)
            h=h+label_emb
            h = F.relu(h)
            if i in self.skips:
               # ipdb.set_trace() 
                h = torch.cat([input_pts, h], -1)
        

        
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views.view(B,N,M)], -1)
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb.view(B,N,3), alpha.view(B,N,1)], -1)
        

        return outputs 


    
class NeRF0(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,num_instance=1):
        """ 
        """
        super(NeRF0, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch//3
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.hidden_dim=W

        self.triplane=Triplane()
        

        self.tri_planes = nn.Parameter(torch.randn(num_instance, input_ch, 256, 256))
        #nn.init.normal_(self.tri_planes, mean=0, std=0.1)
        #self.weight=nn.Parameter(torch.ones(1,3,1,input_ch))
        #ipdb.set_trace() 
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W)])

        self.softplus=nn.Softplus()

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W, 3)
        

    def forward(self, x,label):
        #ipdb.set_trace() 
        

        input_pts, input_views = torch.split(x, [int(x.shape[-1]-self.input_ch_views), self.input_ch_views], dim=-1)
        B,N,M=input_views.shape
        sample_triplane=self.tri_planes[label].view(B,3,self.tri_planes.shape[-3]//3,self.tri_planes.shape[-2],self.tri_planes.shape[-1])
        #ipdb.set_trace() 
        input_pts=(self.triplane(sample_triplane,input_pts,8)).mean(1).view(-1,self.tri_planes.shape[-3]//3)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            #ipdb.set_trace() 
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
               # ipdb.set_trace() 
                h = torch.cat([input_pts, h], -1)

        
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views.view(B*N,M)], -1)
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb.view(B,N,3), alpha.view(B,N,1)], -1)
        

        return outputs 
class NeRF1(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,num_instance=1):
        """ 
        """
        super(NeRF1, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch//3*9
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.hidden_dim=W

        self.triplane=Triplane()
        

        self.tri_planes = nn.Parameter(torch.randn(num_instance, input_ch, 256, 256))
        #nn.init.normal_(self.tri_planes, mean=0, std=0.1)
        #self.weight=nn.Parameter(torch.ones(1,3,1,input_ch))
        #ipdb.set_trace() 
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        self.softplus=nn.Softplus()

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)
        

    def forward(self, x,label):
        #ipdb.set_trace() 
        

        input_pts, input_views = torch.split(x, [int(x.shape[-1]-self.input_ch_views), self.input_ch_views], dim=-1)
        B,N,M=input_views.shape
        sample_triplane=self.tri_planes[label].view(B,3,self.tri_planes.shape[-3]//3,self.tri_planes.shape[-2],self.tri_planes.shape[-1])
        #ipdb.set_trace() 
        input_pts=(self.triplane(sample_triplane,input_pts,4)).mean(1).view(-1,self.tri_planes.shape[-3]//3)

        h = torch.cat((input_pts,positional_encoding(input_pts,4)),-1)
        for i, l in enumerate(self.pts_linears):
            #ipdb.set_trace() 
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
               # ipdb.set_trace() 
                h = torch.cat([input_pts, h], -1)

        
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views.view(B*N,M)], -1)
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb.view(B,N,3), alpha.view(B,N,1)], -1)
        

        return outputs 
            
class NeRF_dual(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,num_instance=1):
        """ 
        """
        super(NeRF_dual, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch//3*9+input_ch_views
        self.input_ch2= input_ch//3*5
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.hidden_dim=W

        self.triplane=Triplane()
        

        self.tri_planes1 = nn.Parameter(torch.randn(num_instance, input_ch, 256, 256))
        self.tri_planes2 = nn.Parameter(torch.randn(num_instance, input_ch, 256, 256))
        #nn.init.normal_(self.tri_planes, mean=0, std=0.1)
        #self.weight=nn.Parameter(torch.ones(1,3,1,input_ch))
        #ipdb.set_trace() 
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        self.pts_linears2 = nn.ModuleList(
            [nn.Linear(self.input_ch2, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(W, W//2)])

        self.softplus=nn.Softplus()

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)
        

    def forward(self, x,label):
        #ipdb.set_trace() 
        

        input_pts, input_views = torch.split(x, [int(x.shape[-1]-self.input_ch_views), self.input_ch_views], dim=-1)
        B,N,M=input_views.shape
        sample_triplane1=self.tri_planes1[label].view(B,3,self.tri_planes1.shape[-3]//3,self.tri_planes1.shape[-2],self.tri_planes1.shape[-1])
        #ipdb.set_trace() 
        input_pts1=(self.triplane(sample_triplane1,input_pts,8)).mean(1).view(B,-1,self.tri_planes1.shape[-3]//3)

        sample_triplane2=self.tri_planes2[label].view(B,3,self.tri_planes2.shape[-3]//3,self.tri_planes2.shape[-2],self.tri_planes2.shape[-1])
        #ipdb.set_trace() 
        input_pts2=(self.triplane(sample_triplane2,input_pts,8)).mean(1).view(B,-1,self.tri_planes2.shape[-3]//3)



        #ipdb.set_trace() 
        h = torch.cat((input_pts1,positional_encoding(input_pts1,4),input_views),-1)
        for i, l in enumerate(self.pts_linears):
            #ipdb.set_trace() 
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
               # ipdb.set_trace() 
                h = torch.cat([input_pts, h], -1)

        h = self.feature_linear(h)
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)


        h = torch.cat((input_pts2,positional_encoding(input_pts2,2)),-1)
        for i, l in enumerate(self.pts_linears2):
            #ipdb.set_trace() 
            h = self.pts_linears2[i](h)
            h = F.relu(h)

        alpha = self.alpha_linear(h)


        outputs = torch.cat([rgb.view(B,N,3), alpha.view(B,N,1)], -1)
        

        return outputs         

# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def render_path1(batch_rays, chunk, render_kwargs, gt_imgs=None, savedir=None,savedir1=None,near=None,far=None,label=None):
    rgbs = []
    disps = []

    t = time.time()
    

    #render(chunk=newargs.chunk, rays=batch_rays,near=near,far=far,label=label, retraw=True, **render_kwargs_train)

    rgbs, disps, acc, _ = render(chunk=chunk, rays=batch_rays,near=near,far=far, label=label,**render_kwargs)
    #ipdb.set_trace()

    reso=int(rgbs.shape[-2]**0.5)
    rgbs=rgbs.view(-1,reso,reso,3)
    disps=disps.view(-1,reso,reso,1)
    acc=acc.view(-1,reso,reso,1)
    #ipdb.set_trace()
    gt_imgs=gt_imgs.view(-1,reso,reso,3)
    mask=(gt_imgs.mean(-1)<0.9999)


    #ipdb.set_trace()
    if savedir is not None:
        for i in range(len(rgbs)):

            rgb8 =to8b(rgbs[i].cpu().numpy())#np.fliplr(np.rot90(to8b(rgbs[i]),-1))
            filename = os.path.join(savedir)
            imageio.imwrite(savedir, rgb8)
            imageio.imwrite(savedir1, np.uint8(gt_imgs[i].cpu().numpy()*255))
            #ipdb.set_trace()
            print('psnr:' ,mse2psnr(img2mse(torch.Tensor(rgb8/255).to(device=gt_imgs.device)[mask[i]],(gt_imgs[i])[mask[i]])))
            print('psnr_all:' ,mse2psnr(img2mse(torch.Tensor(rgb8/255).to(device=gt_imgs.device),(gt_imgs[i]))))

    psnr_list = []
    for i in range(len(rgbs)):
        rgb8 = to8b(rgbs[i].cpu().numpy())
        psnr = mse2psnr(img2mse(torch.Tensor(rgb8/255).to(device=gt_imgs.device),(gt_imgs[i])))
        psnr_list.append(psnr)

    #ipdb.set_trace()
    return rgbs, disps, acc, psnr_list

def render(chunk=1024*32, rays=None, c2w=None, ndc=True,label=None,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    #ipdb.set_trace()

    rays_o, rays_d = rays[:,0,...], rays[:,1,...]

    
    viewdirs = rays_d

    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [rays_d.shape[0],-1,3]).float()

    sh = rays_d.shape # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [sh[0],-1,3]).float()
    rays_d = torch.reshape(rays_d, [sh[0],-1,3]).float()
    #ipdb.set_trace()

    near, far = near[:,None,:] * torch.ones_like(rays_d[...,:1]), far[:,None,:] * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    #ipdb.set_trace()
    # Render and reshape
    all_ret = batchify_rays(rays, label,chunk,**kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[2:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def batchify_rays(rays_flat,label, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[1], chunk):
        #ipdb.set_trace()
        ret = render_rays(rays_flat[:,i:i+chunk],label=label, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    #ipdb.set_trace()
    all_ret = {k : torch.cat(all_ret[k], 1) for k in all_ret}
    return all_ret

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                label=None,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    B,N_rays,_ = ray_batch.shape
    
    rays_o, rays_d = ray_batch[:,:,0:3], ray_batch[:,:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,:,-3:] 
    bounds = torch.reshape(ray_batch[...,6:8], [B,-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(near.device)

    z_vals = near * (1.-t_vals) + far * (t_vals)
    

    #z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, label,network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    #ipdb.set_trace()
    act_ff=nn.Softplus()

    raw2alpha = lambda raw, dists, act_fn=act_ff: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(dists.device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1) 

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    #ipdb.set_trace()
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    
    #ipdb.set_trace()  
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0],alpha.shape[1], 1)).to(alpha.device), 1.-alpha + 1e-10], -1), -1)[:,:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d
