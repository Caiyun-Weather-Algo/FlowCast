# https://github.com/SwinTransformer/Video-Swin-Transformer/blob/db018fb8896251711791386bbd2127562fd8d6a6/mmaction/models/backbones/swin_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from src.models.components.utils import DropPath, trunc_normal_
from functools import reduce, lru_cache
from einops import rearrange
from src.models.components.window_attention_v2 import window_partition, window_reverse, WindowAttention3D, get_window_size
from src.models.components.patch_processer import TimestepEmbedder, PatchEmbed2DOverlap, PatchEmbed3DOverlap, \
    PatchMerging, Upsampling, PatchMerging3D, Upsampling3D, \
    PatchRecoveryLowRank, PatchRecovery2DConv, PatchRecovery3DConv, Padding2D, Padding3D


def modulate(x, shift, scale):
    if len(x.shape) == 4:
        return x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
    else:
        return x * (1 + scale[:, None, None, None, :]) + shift[:, None, None, None, :]


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, t_dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, earth_position=False,
                 window_partition_shape=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.t_dim = t_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.earth_position = earth_position

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # adaptive layer-norm zero
        rank = dim // 8
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.t_dim, rank, bias=False), 
            nn.Linear(rank, 6 * dim, bias=True) 
        )
        
        # adaptive layer-norm zero
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        # x = rearrange(x, 'b d h w c -> b c d h w')
        # (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back) = Padding3D(x, window_size)
        # x = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back))
        # x = rearrange(x, 'b c d h w -> b d h w c')
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        window_partition_shape = None
        if self.earth_position:
            window_partition_shape = (Dp // self.window_size[0],
                                      Hp // self.window_size[1],
                                      Wp // self.window_size[2],
                                      )
            # print("x_pad", shifted_x.shape)
            # print("window_partition_shape", window_partition_shape)
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
      #  x = x[:, padding_front: Dp-padding_back, padding_top: Hp-padding_bottom, padding_left: Wp-padding_right, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(x))

    def forward(self, x, mask_matrix, t):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)

        shortcut = x
        # attn
        x = modulate(self.norm1(x), shift_msa, scale_msa)  # scale and shift
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + gate_msa[:, None, None, None, :] * self.drop_path(x)  # scale

        # mlp
        shortcut_mlp = x
        x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        if self.use_checkpoint:
            x = shortcut_mlp + gate_mlp[:, None, None, None, :] * checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = shortcut_mlp + gate_mlp[:, None, None, None, :] * self.forward_part2(x)
        return x


# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            # for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
            for w in slice(-window_size[2]), slice(-window_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 t_dim,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 upsample=None,
                 earth_position=False,
                 window_partition_shape=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.downsample = downsample
        # self.t_downsample = nn.Identity()
        self.upsample = upsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim // 2, norm_layer=norm_layer)
            # self.t_downsample = nn.Sequential(
            #     nn.Linear(dim // 2, dim),
            #     nn.SiLU(),
            #     nn.Linear(dim, dim))
        if self.upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                t_dim=t_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                earth_position=earth_position,
                window_partition_shape=window_partition_shape,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

    def forward(self, x, t):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        if self.downsample is not None:
            x = rearrange(x, 'b c d h w -> b d h w c')
            x = self.downsample(x)
            x = rearrange(x, 'b d h w c -> b c d h w')
            # t = self.t_downsample(t)

        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask, t)

        x = x.view(B, D, H, W, -1)

        if self.upsample is not None:
            x = self.upsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


# @BACKBONES.register_module()
class SwinTransformer3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4, 4, 4),
                 in_chans_2d=5,
                 in_chans_3d=5,
                 in_chans_st=9,  # static and temporal channel
                 #  in_3d_levels=13,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 earth_position=False,
                 window_partition_shapes=[(4, 9, 11), (4, 5, 6), (4, 9, 11)],
                 use_checkpoint=False,
                 ):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.in_chans_2d = in_chans_2d
        self.in_chans_3d = in_chans_3d
        self.in_chans_st = in_chans_st
        # self.in_3d_levels = in_3d_levels

        # split image into non-overlapping patches
        # self.patch_embed3d = nn.ModuleList(PatchEmbed2D(
        #     patch_size=patch_size[1:], in_chans=in_chans_3d, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None) for _ in range(self.in_3d_levels))
        self.patch_embed3d = PatchEmbed3DOverlap(
            patch_size=patch_size, in_chans=in_chans_3d, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed2d = PatchEmbed2DOverlap(
            patch_size=patch_size[1:], in_chans=in_chans_2d + in_chans_st, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.t_embedder = TimestepEmbedder(embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        downsamples = [None, PatchMerging3D, None]
        upsamples = [None, Upsampling3D, None]
        embed_dims = [embed_dim, embed_dim * 2, embed_dim]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dims[i_layer],
                t_dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsamples[i_layer],
                upsample=upsamples[i_layer],
                earth_position=earth_position,
                window_partition_shape=window_partition_shapes[i_layer],
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.gelu = nn.GELU()

        # unembed
        self.unembed3d = PatchRecoveryLowRank(embed_dim, patch_size[0] * patch_size[1] * patch_size[2], in_chans_3d)
        self.unembed2d = PatchRecoveryLowRank(embed_dim, patch_size[1] * patch_size[2], in_chans_2d)

        # self.unembed3d = PatchRecovery3DConv(embed_dim, in_chans_3d)
        # self.unembed2d = PatchRecovery2DConv(embed_dim, in_chans_2d)
        # self.unembed3d = nn.ModuleList(
        #             PatchRecovery(embed_dim, patch_size[0] * patch_size[1] * patch_size[2], in_chans_3d) for _ in
        #             range(self.in_3d_levels))

        # self.layer_norm1 = norm_layer(embed_dim, elementwise_affine=False, eps=1e-6)
        # self.layer_norm2 = norm_layer(embed_dim, elementwise_affine=False, eps=1e-6)

        # rank = embed_dim // 8
        # self.layer1_adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(embed_dim, rank, bias=False), 
        #     nn.Linear(rank, 4 * embed_dim, bias=True) 
        # )
        # self.layer2_adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(embed_dim, rank, bias=False), 
        #     nn.Linear(rank, 4 * embed_dim, bias=True) 
        # )

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.
        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.
        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1,
                                                                                                          self.patch_size[
                                                                                                              0], 1,
                                                                                                          1) / \
                                                self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(2 * self.window_size[1] - 1, 2 * self.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2,
                                                                                                                   L2).permute(
                        1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained

    def forward(self, input, t, static):
        """Forward function."""
        x0 = input[:, 0:self.in_chans_2d]
        x1 = input[:, self.in_chans_2d:]
        x1 = rearrange(x1, 'n (c d) h w -> n c d h w', c=self.in_chans_3d)
        B, C, D, H, W = x1.shape
        x0 = torch.cat((x0, static), dim=1)
        x0 = self.patch_embed2d(x0)
        x1 = self.patch_embed3d(x1)
        # x1 = torch.cat([self.patch_embed3d[i](x1[:, :, i]).unsqueeze(2) for i in range(self.in_3d_levels)], dim=2)
        x = torch.cat((x0.unsqueeze(2), x1), dim=2)
        x = self.pos_drop(x)

        t = self.t_embedder(t)
        # net
        x0 = self.layers[0](x.contiguous(), t)
        x1 = self.layers[1](x0.contiguous(), t)
        x = x0 + x1
        x = self.layers[2](x.contiguous(), t)
        x = x0 + x  ## double residual

        # shift_msa, scale_msa, shift_mlp, scale_mlp = self.layer1_adaLN_modulation(t).chunk(4, dim=1)
        # x1 = rearrange(x1, 'n c d h w -> n d h w c')
        # x1 = modulate(self.layer_norm1(x1), shift_msa, scale_msa) 
        # x1 = rearrange(x1, 'n d h w c -> n c d h w')
        # x = x0 + x1 
        # x =  self.layers[2](x.contiguous(), t)

        # shift_msa, scale_msa, shift_mlp, scale_mlp = self.layer2_adaLN_modulation(t).chunk(4, dim=1)
        # x = rearrange(x, 'n c d h w -> n d h w c')
        # x = modulate(self.layer_norm2(x), shift_msa, scale_msa) 
        # x = rearrange(x, 'n d h w c -> n c d h w')
        # x = x0 + x 
        
        # unembed-linear
        x = rearrange(x, 'n c d h w -> n d h w c')  # 5
        x0 = x[:, 0]
        x0 = self.unembed2d(x0, t)
        x0 = rearrange(x0, 'n h w (h0 w0 c) -> n (h h0) (w w0) c', h0=self.patch_size[1], w0=self.patch_size[2])
        x0 = rearrange(x0, 'n h w c -> n c h w')
        x1 = self.unembed3d(x[:, 1:], t)
        # x1 = torch.cat([self.unembed3d[i](x[:, i+1], t).unsqueeze(1) for i in range(self.in_3d_levels)], dim=1)
        x1 = rearrange(x1, 'n d h w (d0 h0 w0 c) -> n (d d0) (h h0) (w w0) c', d0=self.patch_size[0],
                       h0=self.patch_size[1], w0=self.patch_size[2])
        x1 = x1[:, :-1]
        x1 = rearrange(x1, 'n d h w c -> n (c d) h w')

        # unembed-conv2d/3d
        # x0 = x[:, 0:1]
        # x0 = self.unembed2d(x0, t)
        # x0 = rearrange(x0, 'n c d h w -> n (c d) h w')
        # x1 = self.unembed3d(x[:, 1:], t)
        # x1 = x1[:, :, :-1]
        # x1 = rearrange(x1, 'n c d h w -> n (c d) h w')

        # unembed-conv
        # x = rearrange(x, 'n c d h w -> n d h w c')  # 5
        # x0 = x[:, 0]
        # x0 = self.unembed2d(x0, t)
        # x1 = self.unembed3d(x[:, 1:], t)
        # x1 = x1[:, :, :-1]
        # x1 = rearrange(x1, 'n c d h w-> n (c d) h w')

        # crop
        if self.patch_size[-1] == 4:
            x0 = x0[:, :, 1:-2, :]
            x1 = x1[..., 1:-2, :]
        else:
            x0 = x0[:, :, :-1, :]
            x1 = x1[..., :-1, :]

        return torch.cat((x0, x1), dim=1)  # [x0, x1]

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D, self).train(mode)
        self._freeze_stages()


if __name__ == '__main__':
    model = SwinTransformer3D(in_chans_2d=6,
                              in_chans_3d=5,
                              in_chans_st=9,  # static and temporal channel
                              # in_3d_levels=13,
                              patch_size=(2, 4, 4),
                              embed_dim=192,
                              window_size=(2, 7, 7),
                              depths=[2, 12, 2],
                              num_heads=[6, 12, 6, 24],
                              earth_position=False,
                              window_partition_shapes=[(4, 10, 10), (4, 5, 5), (4, 10, 10)]
                              )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x0 = torch.rand(1, 6, 181, 360).to(device)
    x1 = torch.rand(1, 5 * 13, 181, 360).to(device)
    x = torch.rand(1, 71, 181, 360).to(device)
    s = torch.rand(1, 9, 181, 360).to(device)
    # x0 = torch.rand(1, 6, 721, 1440).to(device)
    # x1 = torch.rand(1, 5 * 13, 721, 1440).to(device)
    # x = torch.rand(1, 71, 721, 1440).to(device)
    # s = torch.rand(1, 9, 721, 1440).to(device)

    # x = [x0, x1]
    t = torch.rand((1)).to(device)
    y = model(x, t, s)
    print('out', y.shape)
    # from torchinfo import summary

    # summary(model, input_data=[x, t, s], device='cuda')