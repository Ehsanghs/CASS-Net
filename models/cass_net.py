import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .layers import SEBlock, AttentionGate, DSConv

class CASSNet(nn.Module):
    def __init__(self, num_input_channels=4, backbone_name='efficientnet_lite0', dropout_rate=0.1):
        super().__init__()
        self.num_input_channels = num_input_channels
        
        # Encoder: EfficientNet-Lite0
        # Creating model with timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            in_chans=num_input_channels,
            out_indices=(1, 2, 3, 4)
        )
        
        # Hardcoded channels for EfficientNet-Lite0 to avoid runtime dummy input checks
        # Features at indices 1, 2, 3, 4
        self.skip_channels = [24, 40, 112] # e1, e2, e3
        self.deepest_channels = 320        # e4
        
        # Decoder Channels
        C_decode = [112, 40, 24, 16] # Adjusted to match logical progression or paper spec
        # Note: Your original code used [80, 40, 24, 16] but mapped from 320. 
        # Let's stick to your original code's logic for consistency with your results.
        C_decode = [80, 40, 24, 16]

        # --- Decoder Blocks ---
        
        # Block 1 (Deepest)
        self.dec1_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1_post_up_conv = nn.Conv2d(self.deepest_channels, C_decode[0], kernel_size=1, bias=False)
        self.dec1_ag = AttentionGate(C_decode[0], self.skip_channels[2], self.skip_channels[2] // 2)
        self.dec1_conv = DSConv(C_decode[0] + self.skip_channels[2], C_decode[0])
        self.dec1_se = SEBlock(C_decode[0])
        self.dec1_dropout = nn.Dropout2d(dropout_rate)

        # Block 2
        self.dec2_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2_post_up_conv = nn.Conv2d(C_decode[0], C_decode[1], kernel_size=1, bias=False)
        self.dec2_ag = AttentionGate(C_decode[1], self.skip_channels[1], self.skip_channels[1] // 2)
        self.dec2_conv = DSConv(C_decode[1] + self.skip_channels[1], C_decode[1])
        self.dec2_se = SEBlock(C_decode[1])
        self.dec2_dropout = nn.Dropout2d(dropout_rate)

        # Block 3
        self.dec3_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3_post_up_conv = nn.Conv2d(C_decode[1], C_decode[2], kernel_size=1, bias=False)
        self.dec3_ag = AttentionGate(C_decode[2], self.skip_channels[0], self.skip_channels[0] // 2)
        self.dec3_conv = DSConv(C_decode[2] + self.skip_channels[0], C_decode[2])
        self.dec3_se = SEBlock(C_decode[2])
        self.dec3_dropout = nn.Dropout2d(dropout_rate)

        # Block 4 (Final Resolution Restoration)
        self.dec4_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4_post_up_conv = nn.Conv2d(C_decode[2], C_decode[3], kernel_size=1, bias=False)
        self.dec4_conv = DSConv(C_decode[3], C_decode[3])
        self.dec4_se = SEBlock(C_decode[3])
        self.dec4_dropout = nn.Dropout2d(dropout_rate)

        # Final Head
        self.final_conv = nn.Sequential(
            nn.Conv2d(C_decode[3], 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Auxiliary Heads (Deep Supervision)
        self.aux1 = nn.Conv2d(C_decode[0], 1, 1)
        self.aux2 = nn.Conv2d(C_decode[1], 1, 1)
        self.aux3 = nn.Conv2d(C_decode[2], 1, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        feats = self.backbone(x)
        # feats[0] is usually stem/stride2, we use [1,2,3,4] based on indices
        skips = feats[:-1]     # [stride4, stride8, stride16] -> [24, 40, 112]
        center_deepest = feats[-1] # stride32 -> 320

        # Decoder 1
        d1_up = self.dec1_up(center_deepest)
        d1 = self.dec1_post_up_conv(d1_up)
        s2 = skips[2] 
        # Safe interpolation if sizes slightly mismatch due to odd input dims
        if s2.shape[2:] != d1.shape[2:]:
            s2 = F.interpolate(s2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        s2_att = self.dec1_ag(d1, s2)
        d1_cat = torch.cat([d1, s2_att], 1)
        d1_out = self.dec1_dropout(self.dec1_se(self.dec1_conv(d1_cat)))

        # Decoder 2
        d2_up = self.dec2_up(d1_out)
        d2 = self.dec2_post_up_conv(d2_up)
        s1 = skips[1]
        if s1.shape[2:] != d2.shape[2:]:
            s1 = F.interpolate(s1, size=d2.shape[2:], mode='bilinear', align_corners=False)
        s1_att = self.dec2_ag(d2, s1)
        d2_cat = torch.cat([d2, s1_att], 1)
        d2_out = self.dec2_dropout(self.dec2_se(self.dec2_conv(d2_cat)))

        # Decoder 3
        d3_up = self.dec3_up(d2_out)
        d3 = self.dec3_post_up_conv(d3_up)
        s0 = skips[0]
        if s0.shape[2:] != d3.shape[2:]:
            s0 = F.interpolate(s0, size=d3.shape[2:], mode='bilinear', align_corners=False)
        s0_att = self.dec3_ag(d3, s0)
        d3_cat = torch.cat([d3, s0_att], 1)
        d3_out = self.dec3_dropout(self.dec3_se(self.dec3_conv(d3_cat)))

        # Decoder 4
        d4_up = self.dec4_up(d3_out)
        d4 = self.dec4_post_up_conv(d4_up)
        d4_out = self.dec4_dropout(self.dec4_se(self.dec4_conv(d4)))

        # Final
        out_small = self.final_conv(d4_out)
        out = self.final_upsample(out_small)
        
        # Ensure exact output match
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        if self.training:
            # Deep Supervision
            aux1 = F.interpolate(self.aux1(d1_out), size=input_size, mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux2(d2_out), size=input_size, mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.aux3(d3_out), size=input_size, mode='bilinear', align_corners=False)
            return out, aux1, aux2, aux3
        
        return out