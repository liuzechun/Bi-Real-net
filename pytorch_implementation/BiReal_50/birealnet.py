import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

stage_repeat = [3, 4, 6, 3]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def binaryconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def binaryconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

class Leakyclip(nn.Module):
    def __init__(self):
        super(Leakyclip, self).__init__()

    def forward(self, x):
         out = x
         mask1 = x < -1
         mask2 = x > 1
         out1 = (0.1 * x - 0.9) * mask1.type(torch.float32) + x * (1 - mask1.type(torch.float32))
         out2 = (0.1 * out1 + 0.9) * mask2.type(torch.float32) + out1 * (1 - mask2.type(torch.float32))
         out = out2
         return out

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(self.shape) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weight = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weight),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weight)
        cliped_weights = torch.clamp(real_weight, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, is_downsample=False, is_first_block=False):
        super(Bottleneck, self).__init__()
        expansion = 4
        midplanes = int(planes/expansion)
        norm_layer = nn.BatchNorm2d
        if is_first_block :
            self.conv1 = conv1x1(inplanes, midplanes)
        else:
            self.conv1 = binaryconv1x1(inplanes, midplanes)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = binaryconv3x3(midplanes, midplanes, stride)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = binaryconv1x1(midplanes, planes)
        self.bn3 = norm_layer(planes)
        self.leaky_clip = Leakyclip()
        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes
        self.midplanes = midplanes
        self.is_first_block = is_first_block
        self.is_downsample = is_downsample
        self.expansion = expansion
        if is_downsample:
            self.down_conv1x1 = conv1x1(int(inplanes/expansion), int(planes/expansion))
            self.pooling = nn.AvgPool2d(2,2)
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride),
                conv1x1(inplanes, planes),
                norm_layer(planes),
            )

    def forward(self, x1, x2):
        identity_first = x1
        identity_original = x2

        if self.is_first_block:
            out1 = self.leaky_clip(x2)
        else:
            out1 = self.binary_activation(x2)
        out1 = self.conv1(out1)
        out1 = self.bn1(out1)

        if not self.is_first_block:
            if self.is_downsample:
                identity_first = self.down_conv1x1(x1)
            out1 += identity_first

        out2 = self.binary_activation(out1)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)

        identity_mid = out1
        if (self.is_downsample) and (self.stride == 2):
            identity_mid = self.pooling(out1)
        out2 += identity_mid

        out3 = self.binary_activation(out2)
        out3 = self.conv3(out3)
        out3 = self.bn3(out3)

        if self.is_downsample:
            identity_original = self.downsample(x2)
        out3 += identity_original

        return out2, out3

class BiRealNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(BiRealNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList()

        #stage1
        self.layers.append(Bottleneck(64, 256, stride=1, is_downsample=True, is_first_block=True))
        for i in range(1, stage_repeat[0]):
            self.layers.append(Bottleneck(256, 256))

        #stage2
        self.layers.append(Bottleneck(256, 512, stride=2, is_downsample=True))
        for i in range(1, stage_repeat[1]):
            self.layers.append(Bottleneck(512, 512))

        #stage3
        self.layers.append(Bottleneck(512, 1024, stride=2, is_downsample=True))
        for i in range(1, stage_repeat[2]):
            self.layers.append(Bottleneck(1024, 1024))

        #stage4
        self.layers.append(Bottleneck(1024, 2048, stride=2, is_downsample=True))
        for i in range(1, stage_repeat[3]):
            self.layers.append(Bottleneck(2048, 2048))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.maxpool(x)
        x2 = self.maxpool(x)

        for i, block in enumerate(self.layers):
            x1, x2 = block(x1, x2)

        x = self.avgpool(x2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def birealnet50():
    model = BiRealNet()
    return model
