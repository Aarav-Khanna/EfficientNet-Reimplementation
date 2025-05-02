import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            Swish(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.use_residual = in_channels == out_channels and stride == 1
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish()
            )
        else:
            self.expand_conv = None

        # Depthwise convolution phase
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride, 
                    padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish()
        )

        # Squeeze and excitation phase
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.se = SEBlock(expanded_channels, reduced_channels)

        # Output phase
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x

        if self.expand_conv is not None:
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)

        if self.use_residual:
            x = x + residual

        return x

class EfficientNet(nn.Module):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate=0.2, num_classes=200, input_size=224):
        super(EfficientNet, self).__init__()
        
        # Base configuration for EfficientNet-B0
        base_config = [
            # t, c, n, s, k
            [1, 16, 1, 1, 3],  # MBConv1_3x3
            [6, 24, 2, 2, 3],  # MBConv6_3x3
            [6, 40, 2, 2, 5],  # MBConv6_5x5
            [6, 80, 3, 2, 3],  # MBConv6_3x3
            [6, 112, 3, 1, 5], # MBConv6_5x5
            [6, 192, 4, 2, 5], # MBConv6_5x5
            [6, 320, 1, 1, 3]  # MBConv6_3x3
        ]

        # Scale width and depth
        scaled_config = []
        for t, c, n, s, k in base_config:
            scaled_c = self._round_filters(c, width_coefficient)
            scaled_n = self._round_repeats(n, depth_coefficient)
            scaled_config.append([t, scaled_c, scaled_n, s, k])

        # Adjust stride in first block for smaller input sizes
        if input_size < 224:
            scaled_config[0][3] = 1  # Change stride from 2 to 1 in first block

        # Stem
        out_channels = self._round_filters(32, width_coefficient)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=1 if input_size < 224 else 2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )

        # Blocks
        self.blocks = nn.ModuleList([])
        in_channels = out_channels
        for t, c, n, s, k in scaled_config:
            out_channels = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.append(MBConvBlock(in_channels, out_channels, k, stride, t, 0.25))
                in_channels = out_channels

        # Head
        final_channels = self._round_filters(1280, width_coefficient)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            Swish()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(final_channels, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _round_filters(self, filters, width_coefficient):
        if width_coefficient == 1.0:
            return filters
        divisor = 8
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(self, repeats, depth_coefficient):
        return int(math.ceil(depth_coefficient * repeats))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.classifier(x)
        return x

def efficientnet_b0(num_classes=200, input_size=224):
    return EfficientNet(1.0, 1.0, 0.2, num_classes, input_size)

def efficientnet_b1(num_classes=200, input_size=224):
    return EfficientNet(1.0, 1.1, 0.2, num_classes, input_size)

def efficientnet_b2(num_classes=200, input_size=224):
    return EfficientNet(1.1, 1.2, 0.3, num_classes, input_size)

def efficientnet_b3(num_classes=200, input_size=224):
    return EfficientNet(1.2, 1.4, 0.3, num_classes, input_size)

def efficientnet_b4(num_classes=200, input_size=224):
    return EfficientNet(1.4, 1.8, 0.4, num_classes, input_size)

def efficientnet_b5(num_classes=200, input_size=224):
    return EfficientNet(1.6, 2.2, 0.4, num_classes, input_size)

def efficientnet_b6(num_classes=200, input_size=224):
    return EfficientNet(1.8, 2.6, 0.5, num_classes, input_size)

def efficientnet_b7(num_classes=200, input_size=224):
    return EfficientNet(2.0, 3.1, 0.5, num_classes, input_size) 