import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from timm.models.layers import trunc_normal_, DropPath

class LayerNorm(nn.Module):
    r""" LayerNorm适配NPU：解决浮点精度+广播+算子兼容问题 """
    def __init__(self, normalized_shape, eps=1e-4, data_format="channels_last"):  # 调大eps到1e-4
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps  # 增大eps，提升NPU下的数值稳定性
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        # 1. 强制NPU兼容：用float32计算，避免float16精度问题
        x = x.to(torch.float32).contiguous()
        
        if self.data_format == "channels_last":
            # ensure weight/bias are on the same device and dtype as input
            n_last = x.shape[-1]
            weight = self.weight.to(x.device, dtype=x.dtype)
            bias = self.bias.to(x.device, dtype=x.dtype)

            if weight.numel() != n_last:
                if weight.numel() == 1:
                    weight = weight.expand(n_last).contiguous()
                    bias = bias.expand(n_last).contiguous()
                else:
                    weight = torch.ones(n_last, device=x.device, dtype=x.dtype)
                    bias = torch.zeros(n_last, device=x.device, dtype=x.dtype)

            return F.layer_norm(x, (n_last,), weight, bias, self.eps)
        elif self.data_format == "channels_first":
            B, C, H, W = x.shape
            
            # 2. 计算均值和方差
            u = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
            s = (x - u).pow(2).mean(dim=1, keepdim=True)  # [B,1,H,W]
            
            # 3. 核心修复：数值保护
            s = torch.clamp(s, min=1e-8)
            u = u.expand(B, C, H, W).contiguous()
            s = s.expand(B, C, H, W).contiguous()
            
            # 4. NPU兼容的开平方
            denom = torch.sqrt(s + self.eps).contiguous()
            
            # 5. 标准化
            x = (x - u) / denom
            x = x.contiguous()
            
            # 6. 权重偏置
            weight = self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(x.device, dtype=x.dtype)
            bias = self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(x.device, dtype=x.dtype)
            x = weight * x + bias
            
            return x

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if transpose is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.act = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
            m.weight.data = m.weight.data.to(torch.float32)
            if m.bias is not None:
                m.bias.data = m.bias.data.to(torch.float32)

    def forward(self, x):
        x = x.to(torch.float32).contiguous()
        y = self.conv(x)
        y = y.contiguous()
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn

class ConvNeXt_block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(64, dim)
        )
        self.dwconv = LKA(dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, time_emb=None):
        input = x
        time_emb = self.mlp(time_emb)
        x = self.dwconv(x) +  rearrange(time_emb, 'b c -> b c 1 1')
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) 
        x = input + self.drop_path(x)
        return x

class ConvNeXt_bottle(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(64, dim)
        )
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x, time_emb=None):
        input = x
        time_emb = self.mlp(time_emb)
        x = self.dwconv(x) +  rearrange(time_emb, 'b c -> b c 1 1')
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) 
        x = self.res_conv(input) + self.drop_path(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = LayerNorm(planes, eps=1e-6, data_format="channels_first")
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = LayerNorm(planes, eps=1e-6, data_format="channels_first") 
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = LayerNorm( planes * self.expansion, eps=1e-6, data_format="channels_first")
        self.relu = nn.SiLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Learnable_Filter(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(640, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 =  LayerNorm(64, eps=1e-6, data_format="channels_first")
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = LayerNorm(64, eps=1e-6, data_format="channels_first")
        self.relu = nn.SiLU(True)
        self.residual = self._make_layer(Bottleneck, 64, 32, 2)
        self.seg_conv = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                LayerNorm(planes * block.expansion, eps=1e-6, data_format="channels_first"),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.residual(x)
        return self.seg_conv(x)

class PhysicalPriorNet(nn.Module):
    """
    Physically informed module to detect potential stress concentration areas in the Matrix.
    
    Inputs: 
        x: Binary image [0: Particle, 1: Matrix]
           (Now supports multi-channel input, assuming Channel 0 contains the geometry info)
    
    Features:
        1. Interfaces (Edges via Sobel)
        2. Local Particle Density (via Pooling)
        3. Structural Geometry
        
    Constraints:
        The output saliency map is forced to be 0 in Particle regions.
    """
    def __init__(self, in_channels, out_channels=1):
        super(PhysicalPriorNet, self).__init__()
        
        # 1. Edge Detection Branch (Fixed Physics - Interfaces)
        self.register_buffer('sobel_x', torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1,1,3,3))
        self.register_buffer('sobel_y', torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1,1,3,3))
        
        # 2. Interaction/Density Branch (Fixed Physics - Clustering)
        # Calculates local volume fraction.
        self.density_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        
        # 3. Learnable Branch
        # Input channels: Original (C) + Grad_X (C) + Grad_Y (C) + Density (C) = 4*C
        total_in_channels = in_channels * 4
        
        self.conv_in = BasicConv2d(total_in_channels, 32, kernel_size=3, stride=1, padding=1)
        
        self.process = nn.Sequential(
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1, act_norm=True),
            Attention(32), 
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1, act_norm=True)
        )
        
        self.conv_out = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        
        # --- Feature 1: Edges (Interfaces) ---
        # 处理多通道：将通道和Batch合并处理，然后还原
        x_reshaped = x.view(b*c, 1, h, w)
        grad_x = F.conv2d(x_reshaped, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.sobel_y, padding=1)
        grad_x = grad_x.view(b, c, h, w)
        grad_y = grad_y.view(b, c, h, w)
        
        # --- Feature 2: Local Density ---
        density = self.density_pool(x)
        
        # --- Concatenate ---
        feat = torch.cat([x, grad_x, grad_y, density], dim=1)
        
        # --- Predict Saliency ---
        feat = self.conv_in(feat)
        feat = self.process(feat)
        out = self.conv_out(feat) # [B, 1, H, W]
        
        # --- Apply Masking Constraint ---
        # x is 0 for particle, 1 for matrix.
        # Use Channel 0 of x as the mask source to keep output single-channel.
        # [Fix] Use slice x[:, 0:1, :, :] to maintain [B, 1, H, W] shape and avoid broadcasting to [B, C, H, W]
        return torch.sigmoid(out) * x[:, 0:1, :, :]
