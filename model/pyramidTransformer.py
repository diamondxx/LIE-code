
import torch.nn.functional as F
from model.DCNv2 import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from torchvision.transforms import Resize

##########################################################################
## Layer Norm


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(DeformableConv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(DeformableConv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


def resize(img, size):
    torch_resize = Resize([size, size])  # 定义Resize类对象
    return torch_resize(img)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None,
                 BN_momentum=0.1):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)
        return out


# with multi leven loss
##########################################################################
class pyramidTransformer(nn.Module):
    def __init__(self, input_c_frame=3, input_c_event=5, output_c=3):
        inp_rgb_channels = input_c_frame
        inp_event_channels = input_c_event
        out_channels = output_c
        dim = 48
        num_blocks_1 = [2, 4, 4, 4, 6]
        num_blocks_2 = [2, 4, 4, 4, 6]
        num_blocks_3 = [2, 2, 2, 2]
        num_refinement_blocks = 4
        heads_1 = [2, 4, 4, 4, 6]
        heads_2 = [2, 4, 4, 4, 6]
        heads_3 = [4, 4, 2, 2]
        heads_4 = [2, 2, 2, 2, 2]
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'WithBias'  #  Other option 'BiasFree'
        self.residual = True
        self.dual_pixel_task = False

        self.use_resblock = True

        super(pyramidTransformer, self).__init__()

        self.patch_embed_rgb = OverlapPatchEmbed(inp_rgb_channels, dim)    
        self.patch_embed_event = OverlapPatchEmbed(inp_event_channels, dim) 

        self.down1_2 = Downsample(dim) 
        self.down1_4 = Downsample(dim * 2) 
        self.down1_8 = Downsample(dim * 4) 
        self.down1_16 = Downsample(dim * 8) 

        self.encoder_1_1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads_1[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks_1[0])])
        self.encoder_1_2 = nn.Sequential(*[
            TransformerBlock(dim=dim * 2, num_heads=heads_1[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks_1[1])])
        self.encoder_1_3 = nn.Sequential(*[
            TransformerBlock(dim=dim * 4, num_heads=heads_1[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks_1[2])])
        self.encoder_1_4 = nn.Sequential(*[
            TransformerBlock(dim=dim * 8, num_heads=heads_1[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks_1[3])])
        self.encoder_1_5 = nn.Sequential(*[
            TransformerBlock(dim=dim * 16, num_heads=heads_1[4], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks_1[4])])
        # resConv
        if self.use_resblock:
            self.build_resblocks()
            self.relu = nn.ReLU(inplace=True)

        self.encoder_2_1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads_2[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks_2[0])])
        self.encoder_2_2 = nn.Sequential(*[
            TransformerBlock(dim=dim * 2, num_heads=heads_2[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks_2[1])])
        self.encoder_2_3 = nn.Sequential(*[
            TransformerBlock(dim=dim * 4, num_heads=heads_2[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks_2[2])])
        self.encoder_2_4 = nn.Sequential(*[
            TransformerBlock(dim=dim * 8, num_heads=heads_2[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks_2[3])])
        self.encoder_2_5 = nn.Sequential(*[
            TransformerBlock(dim=dim * 16, num_heads=heads_2[4], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks_2[4])])

        self.up_1_8 = Upsample(int(dim * 16))
        self.up_1_4 = Upsample(int(dim * 8))
        self.up_1_2 = Upsample(int(dim * 4))
        self.up_1 = Upsample(int(dim * 2))

        self.reduce_chan_1_8 = nn.Conv2d(int(dim * 16), int(dim * 8), kernel_size=1, bias=bias)
        self.reduce_chan_1_4 = nn.Conv2d(int(dim * 8), int(dim * 4), kernel_size=1, bias=bias)
        self.reduce_chan_1_2 = nn.Conv2d(int(dim * 4), int(dim * 2), kernel_size=1, bias=bias)
        self.reduce_chan_1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        self.decoder_1_8 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 8), num_heads=heads_3[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks_3[0])])
        self.decoder_1_4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads_3[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks_3[1])])
        self.decoder_1_2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads_3[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks_3[2])])
        self.decoder_1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads_3[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks_3[3])])

        self.refinement_1_16 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 16), num_heads=heads_4[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refinement_1_8 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 8), num_heads=heads_4[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refinement_1_4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads_4[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refinement_1_2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads_4[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refinement_1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads_4[4], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        # residual connect
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2), kernel_size=1, bias=bias)
        ###########################

        self.output_1_16 = nn.Conv2d(int(dim * 16), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_1_8 = nn.Conv2d(int(dim * 8), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_1_4 = nn.Conv2d(int(dim * 4), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_1_2 = nn.Conv2d(int(dim * 2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_1 = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.sigmoid = nn.Sigmoid()

    def build_resblocks(self):
        # 48 96 192 384 768
        # level 1
        self.resblocks_rgb_1_1 = ResidualBlock(48, 48, norm=None)
        self.resblocks_rgb_1_2 = ResidualBlock(96, 96, norm=None)
        self.resblocks_rgb_1_4 = ResidualBlock(192, 192, norm=None)
        self.resblocks_rgb_1_8 = ResidualBlock(384, 384, norm=None)
        self.resblocks_rgb_1_16 = ResidualBlock(768, 768, norm=None)

        self.resblocks_event_1_1 = ResidualBlock(48, 48, norm=None)
        self.resblocks_event_1_2 = ResidualBlock(96, 96, norm=None)
        self.resblocks_event_1_4 = ResidualBlock(192, 192, norm=None)
        self.resblocks_event_1_8 = ResidualBlock(384, 384, norm=None)
        self.resblocks_event_1_16 = ResidualBlock(768, 768, norm=None)
        # level 2
        self.resblocks_rgb_1_1_l2 = ResidualBlock(48, 48, norm=None)
        self.resblocks_rgb_1_2_l2 = ResidualBlock(96, 96, norm=None)
        self.resblocks_rgb_1_4_l2 = ResidualBlock(192, 192, norm=None)
        self.resblocks_rgb_1_8_l2 = ResidualBlock(384, 384, norm=None)
        self.resblocks_rgb_1_16_l2 = ResidualBlock(768, 768, norm=None)

        self.resblocks_event_1_1_l2 = ResidualBlock(48, 48, norm=None)
        self.resblocks_event_1_2_l2 = ResidualBlock(96, 96, norm=None)
        self.resblocks_event_1_4_l2 = ResidualBlock(192, 192, norm=None)
        self.resblocks_event_1_8_l2 = ResidualBlock(384, 384, norm=None)
        self.resblocks_event_1_16_l2 = ResidualBlock(768, 768, norm=None)


    def forward(self, inp_img, inp_event):

        outputs = []
        # encoder rgb
        inp_rgb_1 = self.patch_embed_rgb(inp_img)
        inp_rgb_1_2 = self.down1_2(inp_rgb_1)
        inp_rgb_1_4 = self.down1_4(inp_rgb_1_2)
        inp_rgb_1_8 = self.down1_8(inp_rgb_1_4)
        inp_rgb_1_16 = self.down1_16(inp_rgb_1_8)

        inp_enc_rgb_1 = self.encoder_1_1(inp_rgb_1)
        inp_enc_rgb_1_2 = self.encoder_1_2(inp_rgb_1_2)
        inp_enc_rgb_1_4 = self.encoder_1_3(inp_rgb_1_4)
        inp_enc_rgb_1_8 = self.encoder_1_4(inp_rgb_1_8)
        inp_enc_rgb_1_16 = self.encoder_1_5(inp_rgb_1_16)

        # encoder event
        inp_event_1 = self.patch_embed_event(inp_event)
        inp_event_1_2 = self.down1_2(inp_event_1)
        inp_event_1_4 = self.down1_4(inp_event_1_2)
        inp_event_1_8 = self.down1_8(inp_event_1_4)
        inp_event_1_16 = self.down1_16(inp_event_1_8)

        inp_enc_event_1 = self.encoder_1_1(inp_event_1)
        inp_enc_event_1_2 = self.encoder_1_2(inp_event_1_2)
        inp_enc_event_1_4 = self.encoder_1_3(inp_event_1_4)
        inp_enc_event_1_8 = self.encoder_1_4(inp_event_1_8)
        inp_enc_event_1_16 = self.encoder_1_5(inp_event_1_16)

        # another res
        if self.use_resblock:
            # level 1
            # event + rgb_res
            # 48 96 192 384 768
            inp_enc_event_1_l1 = self.relu(inp_enc_event_1 + self.resblocks_rgb_1_1(inp_enc_rgb_1))
            inp_enc_event_1_2_l1 = self.relu(inp_enc_event_1_2 + self.resblocks_rgb_1_2(inp_enc_rgb_1_2))
            inp_enc_event_1_4_l1 = self.relu(inp_enc_event_1_4 + self.resblocks_rgb_1_4(inp_enc_rgb_1_4))
            inp_enc_event_1_8_l1 = self.relu(inp_enc_event_1_8 + self.resblocks_rgb_1_8(inp_enc_rgb_1_8))
            inp_enc_event_1_16_l1 = self.relu(inp_enc_event_1_16 + self.resblocks_rgb_1_16(inp_enc_rgb_1_16))
            # rgb + event_res
            inp_enc_rgb_1_l1 = self.relu(inp_enc_rgb_1 + self.resblocks_event_1_1(inp_enc_event_1))
            inp_enc_rgb_1_2_l1 = self.relu(inp_enc_rgb_1_2 + self.resblocks_event_1_2(inp_enc_event_1_2))
            inp_enc_rgb_1_4_l1 = self.relu(inp_enc_rgb_1_4 + self.resblocks_event_1_4(inp_enc_event_1_4))
            inp_enc_rgb_1_8_l1 = self.relu(inp_enc_rgb_1_8 + self.resblocks_event_1_8(inp_enc_event_1_8))
            inp_enc_rgb_1_16_l1 = self.relu(inp_enc_rgb_1_16 + self.resblocks_event_1_16(inp_enc_event_1_16))

            # level 2
            inp_enc_event_1 = self.relu(inp_enc_event_1 + self.resblocks_rgb_1_1_l2(inp_enc_rgb_1_l1))
            inp_enc_event_1_2 = self.relu(inp_enc_event_1_2 + self.resblocks_rgb_1_2_l2(inp_enc_rgb_1_2_l1))
            inp_enc_event_1_4 = self.relu(inp_enc_event_1_4 + self.resblocks_rgb_1_4_l2(inp_enc_rgb_1_4_l1))
            inp_enc_event_1_8 = self.relu(inp_enc_event_1_8 + self.resblocks_rgb_1_8_l2(inp_enc_rgb_1_8_l1))
            inp_enc_event_1_16 = self.relu(inp_enc_event_1_16 + self.resblocks_rgb_1_16_l2(inp_enc_rgb_1_16_l1))

            inp_enc_rgb_1 = self.relu(inp_enc_rgb_1 + self.resblocks_event_1_1_l2(inp_enc_event_1_l1))
            inp_enc_rgb_1_2 = self.relu(inp_enc_rgb_1_2 + self.resblocks_event_1_2_l2(inp_enc_event_1_2_l1))
            inp_enc_rgb_1_4 = self.relu(inp_enc_rgb_1_4 + self.resblocks_event_1_4_l2(inp_enc_event_1_4_l1))
            inp_enc_rgb_1_8 = self.relu(inp_enc_rgb_1_8 + self.resblocks_event_1_8_l2(inp_enc_event_1_8_l1))
            inp_enc_rgb_1_16 = self.relu(inp_enc_rgb_1_16 + self.resblocks_event_1_16_l2(inp_enc_event_1_16_l1))

        # cat or add and then transformer stage 2
        inp_enc_1 = self.encoder_2_1(inp_enc_rgb_1 + inp_enc_event_1)
        inp_enc_1_2 = self.encoder_2_2(inp_enc_rgb_1_2 + inp_enc_event_1_2)
        inp_enc_1_4 = self.encoder_2_3(inp_enc_rgb_1_4 + inp_enc_event_1_4)
        inp_enc_1_8 = self.encoder_2_4(inp_enc_rgb_1_8 + inp_enc_event_1_8)
        inp_enc_1_16 = self.encoder_2_5(inp_enc_rgb_1_16 + inp_enc_event_1_16)

        # decoder
        # 1/16 to 1/8
        out_dec_1_16 = inp_enc_1_16
        out_dec_1_8 = self.up_1_8(out_dec_1_16)
        out_dec_1_8 = torch.cat([out_dec_1_8, inp_enc_1_8], 1)
        out_dec_1_8 = self.reduce_chan_1_8(out_dec_1_8)
        out_dec_1_8 = self.decoder_1_8(out_dec_1_8)
        # 1/8 to 1/4
        out_dec_1_4 = self.up_1_4(out_dec_1_8)
        out_dec_1_4 = torch.cat([out_dec_1_4, inp_enc_1_4], 1)
        out_dec_1_4 = self.reduce_chan_1_4(out_dec_1_4)
        out_dec_1_4 = self.decoder_1_4(out_dec_1_4)
        # 1/4 to 1/2
        out_dec_1_2 = self.up_1_2(out_dec_1_4)
        out_dec_1_2 = torch.cat([out_dec_1_2, inp_enc_1_2], 1)
        out_dec_1_2 = self.reduce_chan_1_2(out_dec_1_2)
        out_dec_1_2 = self.decoder_1_2(out_dec_1_2)
        # 1/2 to 1
        out_dec_1 = self.up_1(out_dec_1_2)
        out_dec_1 = torch.cat([out_dec_1, inp_enc_1], 1)
        out_dec_1 = self.reduce_chan_1(out_dec_1)
        out_dec_1 = self.decoder_1(out_dec_1)
        # refinement
        out_1_16 = self.refinement_1_16(out_dec_1_16)
        out_1_8 = self.refinement_1_8(out_dec_1_8)
        out_1_4 = self.refinement_1_4(out_dec_1_4)
        out_1_2 = self.refinement_1_2(out_dec_1_2)
        out = self.refinement_1(out_dec_1)
        # For Dual-Pixel Defocus Deblurring Task ####
        if self.residual:
            if self.dual_pixel_task:
                out = out + self.skip_conv(inp_rgb_1)
                out = self.output_1(out)
            ###########################
            else:
                # enter here
                out_1_16 = self.output_1_16(out_1_16) + resize(inp_img, 16)
                out_1_8 = self.output_1_8(out_1_8) + resize(inp_img, 32)
                out_1_4 = self.output_1_4(out_1_4) + resize(inp_img, 64)
                out_1_2 = self.output_1_2(out_1_2) + resize(inp_img, 128)
                out = self.output_1(out) + inp_img
        else:
            out_1_16 = self.output_1_16(out_1_16)
            out_1_8 = self.output_1_8(out_1_8)
            out_1_4 = self.output_1_4(out_1_4)
            out_1_2 = self.output_1_2(out_1_2)
            out = self.output_1(out)

        out_1_16 = self.sigmoid(out_1_16)
        out_1_8 = self.sigmoid(out_1_8)
        out_1_4 = self.sigmoid(out_1_4)
        out_1_2 = self.sigmoid(out_1_2)
        out = self.sigmoid(out)

        # multi-level loss calculate
        outputs.append(out_1_16)
        outputs.append(out_1_8)
        outputs.append(out_1_4)
        outputs.append(out_1_2)
        outputs.append(out)

        return outputs, outputs[4]