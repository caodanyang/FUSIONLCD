import math
import torch
import torch._utils
import torch.nn as nn
from typing import Optional, Callable
from torchvision.models import resnet


class RIConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.use_bias = bias
        idx = torch.arange(kernel_size ** 2).view(-1, 1)
        row = torch.div(idx, kernel_size, rounding_mode='floor')
        col = torch.fmod(idx, kernel_size)
        idx = torch.cat([row, col], dim=1)
        dis = (idx - 0.5 * (kernel_size - 1)).norm(dim=1) + 0.5 * (kernel_size % 2 - 1)
        dis = dis.view(kernel_size, kernel_size)
        dis = torch.round(dis).long()
        dis[dis > 0.5 * (kernel_size - 1)] = -1
        self.mask = dis
        self.number = int(torch.max(dis).item() + 1)
        self.weight = torch.zeros([kernel_size, kernel_size, out_channel, in_channel])
        if bias:
            self.bias = torch.nn.Parameter(torch.rand([out_channel, ]))
        else:
            self.bias = None
        self.weight1 = torch.nn.Parameter(torch.rand([self.number, out_channel, in_channel]))

    def forward(self, x):
        weight = self.weight.to(self.weight1.device)
        for i in range(self.number):
            mask = self.mask == i
            weight[mask] = self.weight1[i]
        weight = weight.permute(2, 3, 0, 1)
        y = torch.nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding)
        return y

    def __repr__(self):
        return f"RIConv2d(in_channel={self.weight.shape[3]}, out_channel={self.weight.shape[2]}," \
               f" kernel_size={self.weight.shape[0]}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None})"


class RIMaxpool2d(nn.Module):
    def __init__(self, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        idx = torch.arange(kernel_size ** 2).view(-1, 1)
        row = torch.div(idx, kernel_size, rounding_mode='floor')
        col = torch.fmod(idx, kernel_size)
        idx = torch.cat([row, col], dim=1)
        dis = (idx - 0.5 * (kernel_size - 1)).norm(dim=1) + 0.5 * (kernel_size % 2 - 1)
        dis = dis.view(kernel_size, kernel_size)
        dis = torch.round(dis)
        dis[dis > 0.5 * (kernel_size - 1)] = -1
        self.mask = dis.view(-1, ) > -1

    def forward(self, x):
        B, C, H, W = x.shape
        h_out = math.floor((H + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
        w_out = math.floor((W + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
        unfold_x = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        y = unfold_x.view(B, C, self.kernel_size * self.kernel_size, h_out, w_out)
        y = y.permute(2, 0, 1, 3, 4)
        y1 = y[self.mask]
        y_max = torch.max(y1, dim=0, keepdim=False)[0]
        return y_max

    def __repr__(self):
        return f"RIMaxpool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class RIAvgpool2d(nn.Module):
    def __init__(self, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        idx = torch.arange(kernel_size ** 2).view(-1, 1)
        row = torch.div(idx, kernel_size, rounding_mode='floor')
        col = torch.fmod(idx, kernel_size)
        idx = torch.cat([row, col], dim=1)
        dis = (idx - 0.5 * (kernel_size - 1)).norm(dim=1) + 0.5 * (kernel_size % 2 - 1)
        dis = dis.view(kernel_size, kernel_size)
        dis = torch.round(dis)
        dis[dis > 0.5 * (kernel_size - 1)] = -1
        mask = dis > -1
        self.number = torch.sum(mask)
        self.weight = torch.zeros([kernel_size, kernel_size, 1, 1])
        self.weight[mask] = 1

    def forward(self, x):
        weight = self.weight.to(x.device)
        weight = weight.permute(2, 3, 0, 1)
        weight = weight.repeat(x.shape[1], 1, 1, 1)
        sum = torch.nn.functional.conv2d(x, weight, None, self.stride, self.padding, groups=x.shape[1])
        avg = sum / self.number
        return avg

    def __repr__(self):
        return f"RIAvgpool2d(kernel_size={self.weight.shape[0]}, stride={self.stride}, padding={self.padding})"


class MFConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size + 2, stride=stride, padding=padding + 1, bias=bias)
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size + 4, stride=stride, padding=padding + 2, bias=bias)
        self.dp = nn.Dropout(0.3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.dp(x1) + self.dp(x2) + self.dp(x3)
        return x4


class MFConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = MFConv(in_channel=in_channels, out_channel=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = MFConv(in_channel=out_channels, out_channel=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


class RIConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = RIConv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=5, padding=2, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = RIConv2d(in_channel=out_channels, out_channel=out_channels, kernel_size=5, padding=2, bias=False)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


class RIResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(RIResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = RIConv2d(in_channel=inplanes, out_channel=planes, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = RIConv2d(in_channel=planes, out_channel=planes, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class RICNN(nn.Module):
    def __init__(self, c1: int = 8, c2: int = 16, c3: int = 32, c4: int = 64, dim: int = 64
                 ):
        super().__init__()

        self.gate = nn.ReLU(inplace=True)
        self.pool2 = RIMaxpool2d(kernel_size=2, stride=2)
        self.pool4 = RIMaxpool2d(kernel_size=5, stride=4, padding=1)
        self.block1 = RIConvBlock(3, c1, self.gate, nn.BatchNorm2d)
        self.block2 = RIResBlock(inplanes=c1, planes=c2, stride=1,
                                 downsample=nn.Conv2d(c1, c2, 1),
                                 gate=self.gate,
                                 norm_layer=nn.BatchNorm2d)
        self.block3 = RIResBlock(inplanes=c2, planes=c3, stride=1,
                                 downsample=nn.Conv2d(c2, c3, 1),
                                 gate=self.gate,
                                 norm_layer=nn.BatchNorm2d)
        self.block4 = RIResBlock(inplanes=c3, planes=c4, stride=1,
                                 downsample=nn.Conv2d(c3, c4, 1),
                                 gate=self.gate,
                                 norm_layer=nn.BatchNorm2d)

        self.conv1 = resnet.conv1x1(c1, dim // 4)
        self.conv2 = resnet.conv1x1(c2, dim // 4)
        self.conv3 = resnet.conv1x1(c3, dim // 4)
        self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(dim, dim + 1, 1)

    def forward(self, image):
        x1 = self.block1(image)
        x2 = self.pool2(x1)
        x2 = self.block2(x2)
        x3 = self.pool4(x2)
        x3 = self.block3(x3)
        x4 = self.pool4(x3)
        x4 = self.block4(x4)
        x1 = self.gate(self.conv1(x1))
        x2 = self.gate(self.conv2(x2))
        x3 = self.gate(self.conv3(x3))
        x4 = self.gate(self.conv4(x4))
        x2_up = self.upsample2(x2)
        x3_up = self.upsample3(x3)
        x4_up = self.upsample4(x4)
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)
        y = self.out(x1234)
        descriptor_map = y[:, :-1, :, :]
        scores_map = torch.sigmoid(y[:, -1, :, :]).unsqueeze(1)
        return scores_map, descriptor_map

    def ri2maxpool(self, pool):
        stride = pool.stride
        pool_new = nn.MaxPool2d(stride)
        return pool_new

    def maxpool2ri(self, pool):
        kernel_size = stride = pool.stride
        ds = round((math.sqrt(2) - 1) / 2 * stride - 0.25 * (stride % 2 - 1))
        kernel_size = kernel_size + ds
        pool_new = RIMaxpool2d(kernel_size, stride, ds)
        return pool_new

    def ri2avgpool(self, pool):
        stride = pool.stride
        pool_new = nn.AvgPool2d(stride)
        return pool_new

    def avgpool2ri(self, pool):
        kernel_size = stride = pool.stride
        if stride > 3:
            kernel_size = kernel_size + 1
        pool_new = RIAvgpool2d(kernel_size, stride)
        return pool_new

    def ri2conv(self, conv):
        ri = conv
        weight = ri.weight
        device = ri.weight1.device
        bias = ri.bias
        use_bias = bias is not None
        weight_copy = weight.clone().to(device)
        for i in range(ri.number):
            mask = ri.mask == i
            weight_copy[mask] = ri.weight1[i]
        weight_copy = weight_copy.permute(2, 3, 0, 1)
        in_c = weight.shape[3]
        out_c = weight.shape[2]
        kz = weight.shape[0]
        sd = ri.stride
        pd = ri.padding
        conv_new = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kz, stride=sd, padding=pd, bias=use_bias)
        if use_bias:
            state_dict = {'weight': weight_copy, 'bias': bias}
        else:
            state_dict = {'weight': weight_copy}
        conv_new.load_state_dict(state_dict)
        return conv_new.to(device)

    def conv2ri(self, conv):
        weight = conv.weight
        bias = conv.bias
        device = weight.device
        in_c = weight.shape[1]
        out_c = weight.shape[0]
        kz = weight.shape[2]
        if kz < 3:
            return conv
        sd = conv.stride
        pd = conv.padding
        idx = torch.arange(kz ** 2).view(-1, 1)
        row = torch.div(idx, kz, rounding_mode='floor')
        col = torch.fmod(idx, kz)
        idx = torch.cat([row, col], dim=1)
        dis = (idx - 0.5 * (kz - 1)).norm(dim=1) + 0.5 * (kz % 2 - 1)
        dis = dis.view(kz, kz)
        dis = torch.round(dis).long()
        dis[dis > 0.5 * (kz - 1)] = -1
        mask = dis
        number = int(torch.max(dis).item() + 1)
        weight1 = torch.rand([number, out_c, in_c]).to(device)
        weight2 = weight.clone()
        weight2 = weight2.permute(2, 3, 0, 1)
        used_bias = bias is not None
        for i in range(number):
            mask1 = mask == i
            w = weight2[mask1]
            weight1[i] = torch.mean(w, dim=0)
        if used_bias:
            state_dict = {'weight1': weight1, 'bias': bias}
        else:
            state_dict = {'weight1': weight1}
        conv_new = RIConv2d(in_channel=in_c, out_channel=out_c, kernel_size=kz, stride=sd, padding=pd, bias=used_bias)
        conv_new.load_state_dict(state_dict)
        return conv_new.to(device)

    def disable_ri(self):
        modules = self.__dict__['_modules']
        for key, value in modules.items():
            if isinstance(value, RIMaxpool2d):
                setattr(self, key, self.ri2maxpool(value))
            if isinstance(value, RIAvgpool2d):
                setattr(self, key, self.ri2avgpool(value))
            if isinstance(value, RIConv2d):
                setattr(self, key, self.ri2conv(value))
            if 'block' in key:
                block = value
                block_modules = block.__dict__['_modules']
                for bkey, bvalue in block_modules.items():
                    if isinstance(bvalue, RIMaxpool2d):
                        setattr(block, bkey, self.ri2maxpool(bvalue))
                    if isinstance(bvalue, RIAvgpool2d):
                        setattr(block, bkey, self.ri2avgpool(bvalue))
                    if isinstance(bvalue, RIConv2d):
                        setattr(block, bkey, self.ri2conv(bvalue))
                modules[key] = block_modules
                setattr(self, key, block)

    def enable_ri(self):
        modules = self.__dict__['_modules']
        for key, value in modules.items():
            if isinstance(value, nn.MaxPool2d):
                setattr(self, key, self.maxpool2ri(value))
            if isinstance(value, nn.AvgPool2d):
                setattr(self, key, self.avgpool2ri(value))
            if isinstance(value, nn.Conv2d):
                setattr(self, key, self.conv2ri(value))
            if 'block' in key:
                block = value
                block_modules = block.__dict__['_modules']
                for bkey, bvalue in block_modules.items():
                    if isinstance(bvalue, nn.MaxPool2d):
                        setattr(block, bkey, self.maxpool2ri(bvalue))
                    if isinstance(bvalue, nn.AvgPool2d):
                        setattr(block, bkey, self.avgpool2ri(bvalue))
                    if isinstance(bvalue, nn.Conv2d):
                        setattr(block, bkey, self.conv2ri(bvalue))
                modules[key] = block_modules
                setattr(self, key, block)


class EncodePosition(nn.Module):
    def __init__(self, feature_size=128):
        super().__init__()
        self.bins = 16
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.bins, out_channels=feature_size//2, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm1d(feature_size//2), nn.ReLU(),
            nn.Conv1d(in_channels=feature_size//2, out_channels=feature_size//2, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm1d(feature_size//2), nn.ReLU(),
            nn.Conv1d(in_channels=feature_size//2, out_channels=feature_size, kernel_size=1, stride=1, padding=0, bias=True)
        )
        # self.conv2=(nn.Conv1d(in_channels=256,out_channels=128,kernel_size=1))

    def forward(self, x, fea):
        b, n, c = x.shape
        x1 = x.unsqueeze(1)
        x2 = x.unsqueeze(2)
        dx = x1 - x2
        distance = dx.norm(p=2, dim=3)
        hists = torch.zeros([b, n, self.bins]).to(x.device)
        for i in range(b):
            for j in range(n):
                dis = distance[i, j]
                hist = torch.histc(dis, bins=self.bins, min=1, max=80)
                hists[i, j] = hist
        hists = hists / torch.sum(hists, dim=2, keepdim=True)
        x3 = hists.permute(0, 2, 1)
        x4 = self.conv1(x3)
        if hasattr(self, 'conv2'):
            x5 = torch.cat([fea, x4], dim=1)
            y = self.conv2(x5)
        else:
            y = fea + x4
        return y
