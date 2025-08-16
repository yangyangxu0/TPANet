import torch.nn as nn
import torch
from einops import rearrange
from torch.nn.modules.utils import _pair as to_2tuple
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from timm.models.layers import trunc_normal_
import math
from einops import rearrange
from timm.models.layers import DropPath

from . import utils_heads
from .base import BaseHead

Norm = nn.LayerNorm

class AdaptHead(BaseHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enc_dec_times=1
        self.head_endpoints = ['final']
        linear_dim = self.in_channels
        out_dim = 108
        self.out_dim = out_dim
        out_channels = self.in_channels // 4

        self.bottleneck = nn.ModuleDict({t: utils_heads.ConvBNReLU(out_dim,
                                                                   out_channels,
                                                                   kernel_size=3,
                                                                   norm_layer=nn.BatchNorm2d,
                                                                   activation_layer=nn.ReLU)
                                         for t in self.tasks})
        self.final_logits = nn.ModuleDict({t: nn.Conv2d(out_channels,
                                                        self.task_channel_mapping[t]['final'],
                                                        kernel_size=1,
                                                        bias=True)
                                           for t in self.tasks})

        self.specific_tasks = nn.ModuleList([nn.Conv2d(self.in_channels, out_dim, kernel_size=1, groups=2)  for t in range(len(self.tasks))])
        self.inject_prior = nn.Conv2d(self.in_channels, out_dim, kernel_size=1, groups=2)

        self.adapt_task_mixing = AdaptTaskMixing(channel_dim=out_dim*len(self.tasks), token_dim=14980)

        self.cross_atts = nn.ModuleList([nn.MultiheadAttention(out_dim, num_heads=2) for t in range(len(self.tasks))])
        self.mlp = Mlp(out_dim, hidden_features=out_dim*2, act_layer=nn.GELU, norm_layer=Norm)
        self.norm2 = Norm(out_dim)
        drop_path=0.1
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()


    def forward(self, inp, inp_shape, image, **kwargs):
        out=[]
        inp = self._transform_inputs(inp)
        _, C, H ,W = inp.shape
        out_ls=[]
        x_t=[]

        inp_inj = self.inject_prior(inp)
        
        for idx, specific_task in enumerate(self.specific_tasks):
            out_ls.append(specific_task(inp) + inp_inj)

        x = torch.cat(out_ls, dim=1)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x_ada = self.adapt_task_mixing(x)
        x_adapt = torch.split(x_ada, self.out_dim, dim=2)

        for i, cross_att in enumerate(self.cross_atts):
            x_q = rearrange(out_ls[i], 'b c h w -> b (h w) c')
            x_q = x_q + x_adapt[i] + cross_att(x_adapt[i], x_adapt[i], x_adapt[i])[0] + cross_att(x_q, x_adapt[i], x_adapt[i])[0]
            x_q = x_q + self.drop_path(self.mlp(self.norm2(x_q))) 
            x_t.append(rearrange(x_q, 'b (h w) c -> b c h w', h=H, w=W))

        inp_dict = {t: x_t[idx] for idx, t in enumerate(self.tasks)}


        task_specific_feats = {t: self.bottleneck[t](inp_dict[t]) for t in self.tasks}
        final_pred = {t: self.final_logits[t](task_specific_feats[t]) for t in self.tasks}
        final_pred = {t: nn.functional.interpolate(
            final_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.tasks}
        return {'final': final_pred}

        final_pred = {t: self.final_logits[t](task_specific_feats[t]) for t in self.tasks}
        final_pred = {t: nn.functional.interpolate(final_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.tasks}
        return {'final':final_pred}



class AdaptTaskMixing(nn.Module):
    def __init__(self, channel_dim, token_dim, drop_path=0.1, act=nn.GELU, has_ffn=True):
        super(AdaptTaskMixing, self).__init__()

        self.fc_s = Mlp(token_dim, hidden_features= token_dim*0.0005, act_layer=act, norm_layer=Norm)
        self.fc_c = Mlp(channel_dim, hidden_features=channel_dim, act_layer=act, norm_layer=Norm)

        self.drop_path = DropPath(drop_prob=0.1) if drop_path else nn.Identity()
        self.LN = nn.LayerNorm(channel_dim)

    def forward(self, x, parts=None, qpos=None, kpos=None, mask=None):
        """
        Args:
            feats: task list [b,c,h,w]
            kpos: [B, patch_num * patch_size, C]
            mask: [B, 1, patch_num, patch_size] if exists, else None
        Returns:
            parts: [b,c,h,w]
        """
        y = self.LN(x).permute(0,2,1)

        y = self.fc_s(y).permute(0,2,1)
        x = x + y
        y = self.LN(x)
        x = x + self.fc_c(y)
        return x



class SimpleReasoning(nn.Module):
    def __init__(self, np, dim):
        super(SimpleReasoning, self).__init__()
        self.norm = Norm(dim)
        self.linear = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        tokens = self.norm(x).permute(0,2,1)
        tokens = self.linear(tokens).permute(0,2,1)
        return x + tokens

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


