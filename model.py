"""model.py - U-Net by Aleksei Tiulpin (University of Oulu, 2017)"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# Helper modules
def _norm(name, ch):
    if name in (None, "", "none"):
        return None
    if name == "BN":
        return nn.BatchNorm2d(ch)
    if name == "IN":
        return nn.InstanceNorm2d(ch)
    raise ValueError(f"Unknown normalization: {name}")


def _act(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "selu":
        return nn.SELU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


def conv_block_3x3(in_ch, out_ch, activation="relu", normalization="BN"):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=normalization in (None, "", "none"))]
    n = _norm(normalization, out_ch)
    if n is not None:
        layers.append(n)
    layers.append(_act(activation))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Two 3x3 conv blocks, then 2x2 max-pool. Returns (convolved, pooled)."""

    def __init__(self, in_ch, out_ch, activation="relu", normalization="BN"):
        super().__init__()
        self.convs = nn.Sequential(
            conv_block_3x3(in_ch, out_ch, activation, normalization),
            conv_block_3x3(out_ch, out_ch, activation, normalization),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.convs(x)
        return out, self.pool(out)


class Decoder(nn.Module):
    """Upsample to the skip connection's size, concatenate, two 3x3 conv blocks.
    in_ch = skip channels + upsampled channels."""

    def __init__(self, in_ch, out_ch, activation="relu", normalization="BN"):
        super().__init__()
        self.convs = nn.Sequential(
            conv_block_3x3(in_ch, out_ch, activation, normalization),
            conv_block_3x3(out_ch, out_ch, activation, normalization),
        )

    def forward(self, skip, x):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convs(torch.cat([skip, x], dim=1))


# Professor's UNet
class UNet(nn.Module):
    """UNet architecture with 3x3 convolutions. Created dynamically based on the depth and width.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    Parameters
    ----------
    bw : int
        Basic width of the network, which is doubled at each layer.
    depth : int
        Number of layers
    center_depth : int
        Depth of the central block in UNet.
    n_inputs : int
        Number of input channels.
    n_classes : int
        Number of input classes
    activation : str
        Activation function. Can be ReLU, SeLU, or ELU. The latter has parameter 1 by default.
    norm_encoder : str
        Normalization to be used in the encoder.
    norm_decoder : str
        Normalization to be used in the decoder.
    norm_center : str
        Normalization to be used in the center of the network.
    """

    def __init__(self, bw=24, depth=6, center_depth=2, n_inputs=1, n_classes=1,
                 activation='relu', norm_encoder='BN', norm_decoder='BN', norm_center='BN'):
        super().__init__()
        # Preparing the modules dict
        modules = OrderedDict()
        modules['down1'] = Encoder(n_inputs, bw, activation=activation, normalization=norm_encoder)
        # Automatically creating the Encoder based on the depth and width

        mul_out = None
        for level in range(2, depth + 1):
            mul_in = 2 ** (level - 2)
            mul_out = 2 ** (level - 1)
            layer = Encoder(bw * mul_in, bw * mul_out, activation=activation, normalization=norm_encoder)
            modules['down' + str(level)] = layer

        if mul_out is None:
            raise ValueError('The depth parameter is wrong. Cannot determine the output size of the encoder')
            # Creating the center
        modules['center'] = nn.Sequential(
            *[conv_block_3x3(bw * mul_out, bw * mul_out, activation, norm_center) for _ in range(center_depth)]
        )
        # Automatically creating the decoder
        for level in reversed(range(2, depth + 1)):
            mul_in = 2 ** (level - 1)
            layer = Decoder(2 * bw * mul_in, bw * mul_in // 2, activation=activation, normalization=norm_decoder)
            modules['up' + str(level)] = layer

        modules['up1'] = Decoder(bw + bw, bw, activation=activation, normalization=norm_decoder)

        modules['mixer'] = nn.Conv2d(bw, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.__dict__['_modules'] = modules

    def encoder_parameters(self):
        """Added for fine-tuning (--freeze_encoder): parameters of the down blocks and center."""
        return [p for n, m in self._modules.items()
                if n.startswith('down') or n == 'center' for p in m.parameters()]

    def forward(self, x):
        encoded_results = {}
        out = x

        for name in self.__dict__['_modules']:
            if name.startswith('down'):
                layer = self.__dict__['_modules'][name]
                convolved, pooled = layer(out)
                encoded_results[name] = convolved
                out = pooled

        out = self.center(out)

        for name in self.__dict__['_modules']:
            if name.startswith('up'):
                layer = self.__dict__['_modules'][name]
                out = layer(encoded_results['down' + name[-1]], out)

        return self.mixer(out)
