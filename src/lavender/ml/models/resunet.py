import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RESIDUAL DOUBLE CONV
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

        # ----------------------------------------------------
        # Residual projection
        # ----------------------------------------------------

        if in_channels != out_channels:

            self.residual = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )

        else:

            self.residual = nn.Identity()

    def forward(self, x):

        residual = self.residual(x)

        x = self.block(x)

        return x + residual


# ============================================================
# RESUNET
# ============================================================

class ResUNet(nn.Module):

    def __init__(
        self,
        in_channels=12,
        out_channels=1,
        base_filters=32
    ):

        super().__init__()

        f = base_filters

        # ====================================================
        # ENCODER
        # ====================================================

        self.enc1 = DoubleConv(
            in_channels,
            f
        )

        self.enc2 = DoubleConv(
            f,
            f * 2
        )

        self.enc3 = DoubleConv(
            f * 2,
            f * 4
        )

        self.enc4 = DoubleConv(
            f * 4,
            f * 8
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.bottleneck = DoubleConv(
            f * 8,
            f * 16
        )

        # ====================================================
        # DECODER
        # ====================================================

        self.up4 = nn.ConvTranspose2d(
            f * 16,
            f * 8,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            f * 16,
            f * 8
        )

        self.up3 = nn.ConvTranspose2d(
            f * 8,
            f * 4,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            f * 8,
            f * 4
        )

        self.up2 = nn.ConvTranspose2d(
            f * 4,
            f * 2,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            f * 4,
            f * 2
        )

        self.up1 = nn.ConvTranspose2d(
            f * 2,
            f,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            f * 2,
            f
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        self.out = nn.Conv2d(
            f,
            out_channels,
            kernel_size=1
        )

    # ========================================================
    # SAFE CONCATENATION
    # ========================================================

    def _cat(
        self,
        x,
        skip
    ):

        if x.shape[-2:] != skip.shape[-2:]:

            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return torch.cat(
            [x, skip],
            dim=1
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        b = self.bottleneck(
            self.pool(e4)
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        d4 = self.up4(b)

        d4 = self._cat(
            d4,
            e4
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = self._cat(
            d3,
            e3
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = self._cat(
            d2,
            e2
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = self._cat(
            d1,
            e1
        )

        d1 = self.dec1(d1)

        # ----------------------------------------------------
        # Output logits
        # ----------------------------------------------------

        return self.out(d1)