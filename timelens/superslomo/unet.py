import torch as th
import torch.nn.functional as F
from torch import nn

from timelens.common import size_adapter


class Up(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(Up, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        x = x.to(self.conv1.weight.device)  # Assicura che l'input sia sullo stesso device dei pesi
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(th.cat((x, skpCn), 1)), negative_slope=0.1)
        return x


class Down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize):
        super(Down, self).__init__()
        self.conv1 = nn.Conv2d(
            inChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )
        self.conv2 = nn.Conv2d(
            outChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x


class UNet(nn.Module):
    """Modified version of Unet from SuperSloMo.
    
    Difference : 
    1) there is an option to skip ReLU after the last convolution.
    2) there is a size adapter module that makes sure that input of all sizes
       can be processed correctly. It is necessary because original
       UNet can process only inputs with spatial dimensions divisible by 32.
    """

    def __init__(self, inChannels, outChannels, ends_with_relu=True):
        super(UNet, self).__init__()
        self._ends_with_relu = ends_with_relu
        self._size_adapter = size_adapter.SizeAdapter(minimum_size=32)
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = Down(32, 64, 5)
        self.down2 = Down(64, 128, 3)
        self.down3 = Down(128, 256, 3)
        self.down4 = Down(256, 512, 3)
        self.down5 = Down(512, 512, 3)
        self.up1 = Up(512, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.up5 = Up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        # 🔹 Assicura che l'input sia sulla stessa device dei pesi
        device = next(self.parameters()).device
        x = x.to(device)

        # Size adapter spatially augments input to the size divisible by 32.
        x = self._size_adapter.pad(x)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)

        # Note that original code has relu at the end.
        if self._ends_with_relu:
            x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        else:
            x = self.conv3(x)

        # Size adapter crops the output to the original size.
        x = self._size_adapter.unpad(x)
        return x