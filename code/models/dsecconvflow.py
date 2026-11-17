import torch
import torch.nn as nn
import torch.nn.functional as F
from models.convsnn import ConvSiLIFLayer


class SiLIFFlowNetScaled(nn.Module):

    def __init__(self, in_channels=2, channels=[32, 64, 128], dropout=0.1, v_threshold=0.5):
        super(SiLIFFlowNetScaled, self).__init__()
        c1, c2, c3 = channels
        vt = v_threshold

        # Encoder: 3 scales, 2 layers each
        self.enc1a = ConvSiLIFLayer(in_channels, c1, stride=1, dropout=dropout, v_threshold=vt)
        self.enc1b = ConvSiLIFLayer(c1, c1, stride=1, return_membrane=True, dropout=dropout)

        self.enc2a = ConvSiLIFLayer(c1, c2, stride=2, dropout=dropout, v_threshold=vt)
        self.enc2b = ConvSiLIFLayer(c2, c2, stride=1, return_membrane=True, dropout=dropout)

        self.enc3a = ConvSiLIFLayer(c2, c3, stride=2, dropout=dropout, v_threshold=vt)
        self.enc3b = ConvSiLIFLayer(c3, c3, stride=1, return_membrane=True, dropout=dropout)

        # Refinement (coarse-to-fine)
        self.refine2 = ConvSiLIFLayer(c2 + 2, c2, stride=1, return_membrane=True, dropout=dropout)
        self.refine1 = ConvSiLIFLayer(c1 + 2, c1, stride=1, return_membrane=True, dropout=dropout)

        # Flow heads: 1x1 conv, membrane -> (u,v), zero-initialized
        self.flow_head3 = nn.Conv2d(c3, 2, 1, bias=True)
        self.flow_head2 = nn.Conv2d(c2, 2, 1, bias=True)
        self.flow_head1 = nn.Conv2d(c1, 2, 1, bias=True)
        
        
        for head in [self.flow_head1, self.flow_head2, self.flow_head3]:
            nn.init.zeros_(head.bias)   # zero bias → near-zero initial flow

    def forward(self, x):
        """
        x: (B, T, 2, H, W)
        Returns: final_flow, coarse_flow, medium_flow
        """
        
        # TODO Important to check the representation in tid channels are merged on the value, not separated
        B, T, H, W = x.shape
        x = x.unsqueeze(2)  # (B, T, 1, H, W)

        s1 = self.enc1a(x)                             # (B,T,c1,H,W)
        _, s1_mem = self.enc1b(s1)                     # skip1: integrate s1 spikes → membrane

        s2 = self.enc2a(s1)                            # (B,T,c2,H/2,W/2) — feed s1, not enc1b's zeros
        _, s2_mem = self.enc2b(s2)                     # skip2: integrate s2 spikes → membrane

        s3 = self.enc3a(s2)                            # (B,T,c3,H/4,W/4) — feed s2, not enc2b's zeros
        _, s3_mem = self.enc3b(s3)                     # bottleneck membrane
            
        # === COARSE FLOW at H/4 ===
        coarse_flow = self.flow_head3(
            s3_mem.flatten(0, 1)
        ).view(B, T, 2, H//4, W//4)

        # === MEDIUM FLOW at H/2 ===
        coarse_up = F.interpolate(
            coarse_flow.flatten(0, 1), scale_factor=2,
            mode='bilinear', align_corners=False
        ).view(B, T, 2, H//2, W//2)

        refine2_in = torch.cat([s2_mem, coarse_up], dim=2)  # (B,T,c2+2,H/2,W/2)
        _, refine2_mem = self.refine2(refine2_in)

        medium_flow = self.flow_head2(
            refine2_mem.flatten(0, 1)
        ).view(B, T, 2, H//2, W//2)
        medium_flow = medium_flow + coarse_up                # RESIDUAL

        # === FINE FLOW at H ===
        medium_up = F.interpolate(
            medium_flow.flatten(0, 1), scale_factor=2,
            mode='bilinear', align_corners=False
        ).view(B, T, 2, H, W)

        refine1_in = torch.cat([s1_mem, medium_up], dim=2)  # (B,T,c1+2,H,W)
        _, refine1_mem = self.refine1(refine1_in)

        final_flow = self.flow_head1(
            refine1_mem.flatten(0, 1)
        ).view(B, T, 2, H, W)
        final_flow = final_flow + medium_up                  # RESIDUAL

        return final_flow, coarse_flow, medium_flow


if __name__ == "__main__":
    model = SiLIFFlowNetScaled()
    x = torch.randn(4, 10, 2, 256, 256)  # (batch, time, C, H, W)
    x2 = torch.randn(4, 10, 2, 256, 256)

    final_flow, coarse_flow, medium_flow = model(x)
    print("Final flow shape:", final_flow.shape)      # (4,10,2,256,256)
    print("Coarse flow shape:", coarse_flow.shape)    # (4,10,2,64,64)
    print("Medium flow shape:", medium_flow.shape)    # (4,10,2,128,128)
    
    final_flow2, coarse_flow2, medium_flow2 = model(x2)
    
    print(torch.equal(final_flow, final_flow2))  # Should be False (different inputs)