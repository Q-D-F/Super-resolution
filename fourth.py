import torch
import torch.nn as nn
import math


# 基础卷积层和反卷积层
class ConvBlock1(nn.Module):
    def __init__(self, dinput_size, doutput_size):
        super(ConvBlock1, self).__init__()
        self.conv = nn.Conv2d(dinput_size, doutput_size, kernel_size=4, stride=2, dilation=1, padding=1, bias=False)
        self.act = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        return self.act(out)


class ConvBlock3(nn.Module):
    def __init__(self, dinput_size, doutput_size):
        super(ConvBlock3, self).__init__()
        self.conv = nn.Conv2d(dinput_size, doutput_size, kernel_size=4, stride=2, dilation=3, padding=4, bias=False)
        self.act = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        return self.act(out)


class ConvBlock5(nn.Module):
    def __init__(self, dinput_size, doutput_size):
        super(ConvBlock5, self).__init__()
        self.conv = nn.Conv2d(dinput_size, doutput_size, kernel_size=4, stride=2, dilation=5, padding=7, bias=False)
        self.act = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        return self.act(out)


class DeconvBlock1(nn.Module):
    def __init__(self, uinput_size, uoutput_size):
        super(DeconvBlock1, self).__init__()
        self.deconv = nn.ConvTranspose2d(uinput_size, uoutput_size, kernel_size=4, stride=2, dilation=1, padding=1,
                                         bias=False)
        self.act = nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)
        return self.act(out)


class DeconvBlock3(nn.Module):
    def __init__(self, uinput_size, uoutput_size):
        super(DeconvBlock3, self).__init__()
        self.deconv = nn.ConvTranspose2d(uinput_size, uoutput_size, kernel_size=4, stride=2, dilation=3, padding=4,
                                         bias=False)
        self.act = nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)
        return self.act(out)


class DeconvBlock5(nn.Module):
    def __init__(self, uinput_size, uoutput_size):
        super(DeconvBlock5, self).__init__()
        self.deconv = nn.ConvTranspose2d(uinput_size, uoutput_size, kernel_size=4, stride=2, dilation=5, padding=7,
                                         bias=False)
        self.act = nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)
        return self.act(out)


# 基础上采样层和下采样层
class UpBlock1(nn.Module):
    def __init__(self):
        super(UpBlock1, self).__init__()
        self.conv1 = DeconvBlock1(uinput_size=64, uoutput_size=128)
        self.conv2 = ConvBlock1(dinput_size=128, doutput_size=64)
        self.conv3 = DeconvBlock1(uinput_size=64, uoutput_size=128)
        self.conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv0 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.act(self.conv(x))
        hr = self.conv1(x)
        hr1 = self.act(hr)
        hr2 = self.act(self.conv0(hr1))
        lr = self.conv2(hr1)
        lr1 = self.act(lr)
        h0 = self.conv3(lr1 - x1)
        h01 = self.act(h0)
        return h01 + hr2


class UpBlock3(nn.Module):
    def __init__(self):
        super(UpBlock3, self).__init__()
        self.conv1 = DeconvBlock3(uinput_size=64, uoutput_size=128)
        self.conv2 = ConvBlock3(dinput_size=128, doutput_size=64)
        self.conv3 = DeconvBlock3(uinput_size=64, uoutput_size=128)
        self.conv = nn.Conv2d(64, 64, kernel_size=1, stride=1,  padding=0, bias=False)
        self.act = nn.PReLU()
        self.conv0 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.act(self.conv(x))
        hr = self.conv1(x)
        hr1 = self.act(hr)
        hr2 = self.act(self.conv0(hr1))
        lr = self.conv2(hr1)
        lr1 = self.act(lr)
        h0 = self.conv3(lr1 - x1)
        h01 = self.act(h0)
        return h01 + hr2


class UpBlock5(nn.Module):
    def __init__(self):
        super(UpBlock5, self).__init__()
        self.conv1 = DeconvBlock5(uinput_size=64, uoutput_size=128)
        self.conv2 = ConvBlock5(dinput_size=128, doutput_size=64)
        self.conv3 = DeconvBlock5(uinput_size=64, uoutput_size=128)
        self.conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.PReLU()
        self.conv0 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.act(self.conv(x))
        hr = self.conv1(x)
        hr1 = self.act(hr)
        hr2 = self.act(self.conv0(hr1))
        lr = self.conv2(hr1)
        lr1 = self.act(lr)
        h0 = self.conv3(lr1 - x1)
        h01 = self.act(h0)
        return h01 + hr2


class DownBlock1(nn.Module):
    def __init__(self):
        super(DownBlock1, self).__init__()
        self.conv1 = ConvBlock1(dinput_size=128, doutput_size=64)
        self.conv2 = DeconvBlock1(uinput_size=64, uoutput_size=128)
        self.conv3 = ConvBlock1(dinput_size=128, doutput_size=64)
        self.conv = nn.Conv2d(64, 64, kernel_size=1, stride=1,  padding=0, bias=False)
        self.conv0 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.act(self.conv0(x))
        lr = self.conv1(x)
        lr1 = self.act(lr)
        lr2 = self.act(self.conv(lr1))
        hr = self.conv2(lr1)
        hr1 = self.act(hr)
        l0 = self.conv3(hr1 - x1)
        l01 = self.act(l0)
        return l01 + lr2


class DownBlock3(nn.Module):
    def __init__(self):
        super(DownBlock3, self).__init__()
        self.conv1 = ConvBlock3(dinput_size=128, doutput_size=64)
        self.conv2 = DeconvBlock3(uinput_size=64, uoutput_size=128)
        self.conv3 = ConvBlock3(dinput_size=128, doutput_size=64)
        self.conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv0 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.act(self.conv0(x))
        lr = self.conv1(x)
        lr1 = self.act(lr)
        lr2 = self.act(self.conv(lr1))
        hr = self.conv2(lr1)
        hr1 = self.act(hr)
        l0 = self.conv3(hr1 - x1)
        l01 = self.act(l0)
        return l01 + lr2


class DownBlock5(nn.Module):
    def __init__(self):
        super(DownBlock5, self).__init__()
        self.conv1 = ConvBlock5(dinput_size=128, doutput_size=64)
        self.conv2 = DeconvBlock5(uinput_size=64, uoutput_size=128)
        self.conv3 = ConvBlock5(dinput_size=128, doutput_size=64)
        self.conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv0 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.act(self.conv0(x))
        lr = self.conv1(x)
        lr1 = self.act(lr)
        lr2 = self.act(self.conv(lr1))
        hr = self.conv2(lr1)
        hr1 = self.act(hr)
        l0 = self.conv3(hr1 - x1)
        l01 = self.act(l0)
        return l01 + lr2


class ResnetBlock(nn.Module):
    def __init__(self, num_filter):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.act1 = nn.PReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(num_filter, 4, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(4, num_filter, kernel_size=1, padding=0, bias=False)
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = out * 0.1
        out = torch.add(out, x)

        res1 = out
        output = self.avg_pool(out)
        output = self.act1(self.conv2(output))
        output = self.conv3(output)
        output = self.Sig(output)
        output = output * res1
        return output
        return out


class Tiramisu1(nn.Module):
    def __init__(self):
        super(Tiramisu1, self).__init__()
        self.RB1 = ResnetBlock(num_filter=64)
        self.RB2 = ResnetBlock(num_filter=128)
        self.U1 = UpBlock1()
        self.D1 = DownBlock1()

    def forward(self, x):
        U1 = self.U1(x)
        RBU1 = self.RB2(U1)
        D1 = self.D1(RBU1)
        RBU2 = self.RB1(D1)
        return RBU2


class Tiramisu3(nn.Module):
    def __init__(self):
        super(Tiramisu3, self).__init__()
        self.RB1 = ResnetBlock(num_filter=64)
        self.RB2 = ResnetBlock(num_filter=128)
        self.U1 = UpBlock3()
        self.D1 = DownBlock3()

    def forward(self, x):
        U1 = self.U1(x)
        RBU1 = self.RB2(U1)
        D1 = self.D1(RBU1)
        RBU2 = self.RB1(D1)
        return RBU2


class Tiramisu5(nn.Module):
    def __init__(self):
        super(Tiramisu5, self).__init__()
        self.RB1 = ResnetBlock(num_filter=64)
        self.RB2 = ResnetBlock(num_filter=128)
        self.U1 = UpBlock5()
        self.D1 = DownBlock5()

    def forward(self, x):
        U1 = self.U1(x)
        RBU1 = self.RB2(U1)
        D1 = self.D1(RBU1)
        RBU2 = self.RB1(D1)
        return RBU2


class DB(nn.Module):
    def __init__(self):
        super(DB, self).__init__()
        self.Tiramisu1 = Tiramisu1()
        self.Tiramisu3 = Tiramisu3()
        self.Tiramisu5 = Tiramisu5()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        res = x
        output1 = self.Tiramisu1(x)
        output1 = output1 * 0.1
        output3 = self.Tiramisu3(x)
        output3 = output3 * 0.1
        output5 = self.Tiramisu5(x)
        output5 = output5 * 0.1
        output = torch.add(output3, output5)
        output = torch.add(output, output1)
        output = torch.add(output, res)
        output = self.conv(output)
        return output


class RB(nn.Module):
    def __init__(self):
        super(RB, self).__init__()
        # self.RB = self.make_layer(DB, 1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, padding=0, bias=False)
        self.Sig = nn.Sigmoid()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        # output = self.RB(x)
        resi = x
        output = self.conv1(x)
        output = self.relu(output)
        output = self.conv1(output)
        output = output * 0.1
        output = torch.add(output, resi)

        res1 = output
        output = self.avg_pool(output)
        output = self.relu(self.conv2(output))
        output = self.conv3(output)
        output = self.Sig(output)
        output = output * res1
        # output = output * 0.1
        # output = torch.add(output.resi)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual1 = self.make_layer(DB, 1)
        self.residual2 = self.make_layer(RB, 3)
        self.conv_mid1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            #nn.Conv2d(in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.PixelShuffle(2),
        )
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv_input(x)
        residua1 = out
        out = self.residual1(out)
        out = self.conv_mid1(self.residual2(out))
        out = torch.add(out, residua1)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out