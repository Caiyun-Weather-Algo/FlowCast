import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from src.models.components.td_norm import TimeDependentLayerNorm


def modulate(x, shift, scale):
    if len(x.shape) == 4:
        return x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
    else:
        return x * (1 + scale[:, None, None, None, :]) + shift[:, None, None, None, :]


def Padding3D(x, patch_size):
    _, _, D, H, W = x.size()
    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0

    if W % patch_size[2] != 0:
        w_pad = patch_size[2] - W % patch_size[2]
        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)
    if H % patch_size[1] != 0:
        h_pad = patch_size[1] - H % patch_size[1]
        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)
    if D % patch_size[0] != 0:
        l_pad = patch_size[0] - D % patch_size[0]
        padding_front = l_pad // 2
        padding_back = l_pad - padding_front
    return (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)


def Padding2D(x, patch_size):
    _, _, H, W = x.size()
    padding_left = padding_right = padding_top = padding_bottom = 0
    if W % patch_size[1] != 0:
        w_pad = patch_size[2] - W % patch_size[2]
        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)
    if H % patch_size[0] != 0:
        h_pad = patch_size[1] - H % patch_size[1]
        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)
    return (padding_left, padding_right, padding_top, padding_bottom)


class PatchEmbed2DOverlap(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.add_size = 3
        
        kernel = [patch_size[j] + self.add_size for j in range(len(patch_size))]
        pad = [kernel[j] // 2 for j in range(len(kernel))]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel, stride=patch_size)
        self.pad = torch.nn.CircularPad2d((self.add_size, 0, 0, 0))

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        # padding = Padding2D(x, self.patch_size)
        # x = F.pad(x, padding)
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        # padding for larger kernel
        x = self.pad(x)
        x = F.pad(x, (0, 0, self.add_size, 0))
        
        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class PatchEmbed3DOverlap(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.add_size = 3
        kernel = [patch_size[j] + self.add_size for j in range(len(patch_size))]
        pad = [kernel[j] // 2 for j in range(len(kernel))]
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=patch_size)
        self.pad = torch.nn.CircularPad3d((self.add_size, 0, 0, 0, 0, 0))

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        # padding = Padding3D(x, self.patch_size)
        # x = F.pad(x, padding)
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        # padding for larger kernel
        x = self.pad(x)
        x = F.pad(x, (0, 0, self.add_size, 0, self.add_size, 0))
        
        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x
    
    
class PatchEmbed3DClassic(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        padding = Padding3D(x, self.patch_size)
        x = F.pad(x, padding)

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        return x


class PatchEmbed2DClassic(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        padding = Padding2D(x, self.patch_size)
        x = F.pad(x, padding)

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x
    
    
class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        kernel = [patch_size[j] + 3 for j in range(len(patch_size))]
        pad = [kernel[j] // 2 for j in range(len(kernel))]
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=patch_size, padding=pad)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)

        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        return x


class PatchEmbed2D(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        kernel = [patch_size[j] + 3 for j in range(len(patch_size))]
        pad = [kernel[j] // 2 for j in range(len(kernel))]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel, stride=patch_size, padding=pad)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class PatchEmbed3DVarWise(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.convs = nn.ModuleList([nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size) for _ in range(in_chans)])

        # mlp for dimensionality reduction
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_chans, 1),
            nn.GELU(),
        )  
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        # x = self.proj(x)  # B C D Wh Ww
        x_split = rearrange(x, 'b (v 1) d h w ->v b 1 d h w')
        x_proj = [conv(x_i) for conv, x_i in zip(self.convs, x_split)]
        x = torch.stack(x_proj, dim=1)  # Concatenate along the channel dimension (B, in_chans, embed_dim, Wd, Wh, Ww)

        # apply MLP for embedding
        x = rearrange(x, 'b v c d h w ->b d h w c v')
        x = self.mlp(x)  # (B, Wd, Wh, Ww, C, 1)
        x = rearrange(x, 'b d h w c 1 ->b (c 1) d h w ')

        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x
    
    
class PatchEmbed2DVarWise(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # create a list of conv layers for each input channel
        self.convs = nn.ModuleList([nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size) for _ in range(in_chans)])

        # mlp for dimensionality reduction
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_chans, 1),
            nn.GELU(),
        )   
        
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        # x = self.proj(x)  # B C Wh Ww
        # apply each conv layer to the corresponding channel
        x_split = rearrange(x, 'b (v 1) h w ->v b 1 h w')
        x_proj = [conv(x_i) for conv, x_i in zip(self.convs, x_split)]
        x = torch.stack(x_proj, dim=1)  # Concatenate along the channel dimension (B, in_chans, embed_dim, Wh, Ww)

        # apply MLP for embedding
        x = rearrange(x, 'b v c h w ->b h w c v')
        x = self.mlp(x)  # (B, Wh, Ww, C, 1)
        x = rearrange(x, 'b h w c 1 ->b (c 1) h w ')


        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x
    

class PatchRecoveryVarWise(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_total_size, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linears =  nn.ModuleList([nn.Linear(hidden_size, patch_total_size, bias=True)
                                      for _ in range(out_channels)])
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
        self.init_weights()

    def init_weights(self):
        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        for linear in self.linears:
            nn.init.constant_(linear.weight, 0)
            nn.init.constant_(linear.bias, 0)

    def forward(self, x, t):
        # x[n d h w c]
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = torch.stack([self.linears[i](x) for i in range(self.out_channels)], dim=-1)
        return x
    
        
class PatchRecovery(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_total_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_total_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PatchRecoveryLowRank(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_total_size, out_channels, t_dim=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_total_size * out_channels, bias=True)

        t_dim = t_dim if t_dim is not None else hidden_size

        rank = hidden_size // 8
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, rank, bias=False), 
            nn.Linear(rank, 2 * hidden_size, bias=True)
        )

        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PatchRecoveryTD(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_total_size, out_channels):
        super().__init__()
        self.norm_final = TimeDependentLayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_total_size * out_channels, bias=True)

    def forward(self, x, t):
        x = self.norm_final(x, t)
        x = self.linear(x)
        return x


class PatchRecovery3DConvTranspose(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels, patch_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.conv = nn.ConvTranspose3d(hidden_size, out_channels, patch_size, patch_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = rearrange(x, 'n d h w c -> n c d h w')
        x = self.conv(x)
        return x


class PatchRecovery2DConvTranspose(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels, patch_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.conv = nn.ConvTranspose2d(hidden_size, out_channels, patch_size, patch_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = rearrange(x, 'n h w c -> n c h w')
        x = self.conv(x)
        return x
    
    
class PatchRecoveryNoT(nn.Module):
    """
    The final layer of SwinTranfomer.
    """
    def __init__(self, hidden_size, patch_total_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_total_size * out_channels, bias=True)
       
    def forward(self, x):
        x = self.linear(self.norm_final(x))
        return x
    
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.unreduction = nn.Linear(dim // 4, dim // 2, bias=False)
        self.norm = norm_layer(dim // 4)

    def forward(self, x):
        B, D, H, W, C = x.shape
        x = rearrange(x, 'n d h w (h0 w0 c) -> n d (h h0) (w w0) c', h0=2, w0=2)
        x = self.norm(x)
        x = self.unreduction(x)
        return x


class PatchMerging3D(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim            
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, 0::2, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, 0::2, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, 0::2, 1::2, 1::2, :]  # B D H/2 W/2 C
        x4 = x[:, 1::2, 0::2, 0::2, :]  # B D H/2 W/2 C
        x5 = x[:, 1::2, 1::2, 0::2, :]  # B D H/2 W/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B D H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3,x4,x5,x6,x7], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x


class Upsampling3D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim 
        self.unreduction = nn.Linear(dim // 8, dim // 2, bias=False)
        self.norm = norm_layer(dim // 8)

    def forward(self, x):
        B, D, H, W, C = x.shape 
        x = rearrange(x, 'n d h w (d0 h0 w0 c) -> n (d d0) (h h0) (w w0) c', d0=2, h0=2, w0=2)
        x = self.norm(x)
        x = self.unreduction(x)
        return x


class PatchRecovery2DConv(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.conv_before_upsample = nn.Sequential(nn.Conv3d(hidden_size, hidden_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
                                                  nn.LeakyReLU(inplace=True))
        self.conv_up1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_up2 = nn.Conv3d(hidden_size, hidden_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_hr = nn.Conv3d(hidden_size, hidden_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_last = nn.Conv3d(hidden_size, out_channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)

        self.lrelu = nn.LeakyReLU(inplace=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        for m in [self.conv_before_upsample[0], self.conv_up1, self.conv_up2, self.conv_hr, self.conv_last]:
            nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = rearrange(x, 'n d h w c -> n c d h w').contiguous()

        x = self.conv_before_upsample(x).contiguous()
        x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=[1,2,2], mode='nearest')))
        x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=[1,2,2], mode='nearest')))
        x = self.conv_last(self.lrelu(self.conv_hr(x)))
        return x


class PatchRecovery3DConv(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.conv_before_upsample = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True))
        self.conv_up1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_up2 = nn.Conv3d(hidden_size, hidden_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_hr = nn.Conv3d(hidden_size, hidden_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_last = nn.Conv3d(hidden_size, out_channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                   bias=False)
        self.lrelu = nn.LeakyReLU(inplace=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        for m in [self.conv_before_upsample[0], self.conv_up1, self.conv_up2, self.conv_hr, self.conv_last]:
            nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = rearrange(x, 'n d h w c -> n c d h w').contiguous()

        x = self.conv_before_upsample(x).contiguous()
        x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=[1, 2, 2], mode='nearest')))
        x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=[2, 2, 2], mode='nearest')))
        x = self.conv_last(self.lrelu(self.conv_hr(x)))
        return x


if __name__ == "__main__":
    patch_recovery_conv = PatchRecoveryConv2D(hidden_size=96, out_channels=6, tembed=False)
    x = torch.randn(1, 1, 60, 70, 96)
    y = patch_recovery_conv(x)
    print(y.shape)