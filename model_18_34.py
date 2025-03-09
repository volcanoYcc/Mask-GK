import torch
import torch.nn as nn
import torchvision

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Resnet18_34_Unet(nn.Module):
    DEPTH = 6

    def __init__(self,model_type='Resnet18'):
        super().__init__()
        if model_type == 'Resnet18':
            resnet = torchvision.models.resnet.resnet18(pretrained=True)
        elif model_type == 'Resnet34':
            resnet = torchvision.models.resnet.resnet34(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(512, 512)
        up_blocks.append(UpBlockForUNetWithResNet(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet(256, 128))
        up_blocks.append(UpBlockForUNetWithResNet(128, 64))
        up_blocks.append(UpBlockForUNetWithResNet(in_channels=64 + 32, out_channels=64,
                                                    up_conv_in_channels=64, up_conv_out_channels=32))
        up_blocks.append(UpBlockForUNetWithResNet(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=64, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        initialize_weights(self.bridge)
        initialize_weights(self.up_blocks)
        initialize_weights(self.out)

    def forward(self, x):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (Resnet18_34_Unet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{Resnet18_34_Unet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = self.out(x)
        return output_feature_map, pre_pools["layer_4"]
            
if __name__=='__main__':
    model = Resnet18_34_Unet().cuda()
    num = sum([param.nelement() for param in model.parameters()])
    num_require_grad = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    #print(model)
    print("Number of parameter: %.5fM" % (num / 1e6))
    print("Number of require grad parameter: %.5fM" % (num_require_grad / 1e6))
    
    inp = torch.rand((2, 3, 512, 512)).cuda()
    out,_ = model(inp)
    print(out.shape)
    