"""
model.py
SegNet with a VGG-19 (batchnorm) encoder initialized from ImageNet-pretrained
weights (transfer learning - no training from scratch).

Two variants are provided:

  SegNetVGG19Plain
      The original SegNet design (Badrinarayanan et al., 2015): encoder/decoder
      mirror each other, decoder unpools using stored max-pool indices only
      (no feature content crosses from encoder to decoder).

  SegNetVGG19Modified
      Implements three modifications described in polyp-segmentation SegNet
      literature (e.g. "Automatic Polyp Segmentation ... Modified Deep
      Convolutional Encoder-Decoder Architecture", PMC8402594):
        1. Skip connections at all 5 encoder/decoder depths (concat, not add),
           so decoder gets encoder feature *content*, not just pooling indices.
        2. An extra 5x5-conv-BN-ReLU block at the first two encoder depths
           (after their normal 3x3 blocks) and the last two decoder depths
           (before their normal 3x3 blocks), to suppress background noise
           picked up by small 3x3 filters in the shallow layers.
        3. A parallel dilated-convolution ("ASPP-style") module at the
           bottleneck (dilation rates 1, 6, 12, 18), enlarging the receptive
           field to recover context for small/flat polyps without losing
           resolution.

Default: use SegNetVGG19Modified - it is the stronger of the two for this task.
SegNetVGG19Plain is kept for anyone who wants a baseline to compare against.
"""
import torch
import torch.nn as nn
import torchvision.models as models


def _make_vgg_stage(pretrained_layers, n_convs):
    """Builds one encoder stage (n conv-bn-relu blocks) from a slice of
    pretrained VGG19-bn feature layers, keeping the pretrained conv/bn weights."""
    layers = []
    convs = [m for m in pretrained_layers if isinstance(m, nn.Conv2d)]
    bns = [m for m in pretrained_layers if isinstance(m, nn.BatchNorm2d)]
    assert len(convs) == n_convs and len(bns) == n_convs
    for i in range(n_convs):
        layers.append(convs[i])
        layers.append(bns[i])
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _conv_bn_relu(in_ch, out_ch, kernel_size, dilation=1):
    padding = dilation * (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _dec_block(in_ch, mid_ch, out_ch, n_convs):
    """Standard 3x3 decoder conv block: in_ch -> mid_ch (n_convs-1 times) -> out_ch."""
    layers = []
    ch = in_ch
    for i in range(n_convs):
        o = mid_ch if i < n_convs - 1 else out_ch
        layers += [nn.Conv2d(ch, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
        ch = o
    return nn.Sequential(*layers)


def _load_vgg19_bn_stages(pretrained):
    vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None)
    feats = list(vgg.features.children())
    # VGG19_bn feature layout (conv-bn-relu xN, then maxpool):
    # stage1: layers 0:6   (2 convs) -> pool at 6
    # stage2: layers 7:13  (2 convs) -> pool at 13
    # stage3: layers 14:26 (4 convs) -> pool at 26
    # stage4: layers 27:39 (4 convs) -> pool at 39
    # stage5: layers 40:52 (4 convs) -> pool at 52
    return {
        "enc1": _make_vgg_stage(feats[0:6], 2),
        "enc2": _make_vgg_stage(feats[7:13], 2),
        "enc3": _make_vgg_stage(feats[14:26], 4),
        "enc4": _make_vgg_stage(feats[27:39], 4),
        "enc5": _make_vgg_stage(feats[40:52], 4),
    }


# =====================================================================
# Variant 1: original / plain SegNet-VGG19 (kept as a baseline reference)
# =====================================================================
class SegNetVGG19Plain(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        stages = _load_vgg19_bn_stages(pretrained)
        self.enc1, self.enc2, self.enc3, self.enc4, self.enc5 = (
            stages["enc1"], stages["enc2"], stages["enc3"], stages["enc4"], stages["enc5"])

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

        self.dec5 = _dec_block(512, 512, 512, 4)
        self.dec4 = _dec_block(512, 512, 256, 4)
        self.dec3 = _dec_block(256, 256, 128, 4)
        self.dec2 = _dec_block(128, 128, 64, 2)
        self.dec1 = _dec_block(64, 64, 64, 2)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def set_encoder_trainable(self, trainable: bool):
        for m in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
            for p in m.parameters():
                p.requires_grad = trainable

    def forward(self, x):
        x = self.enc1(x); s0 = x.size(); x, idx1 = self.pool(x)
        x = self.enc2(x); s1 = x.size(); x, idx2 = self.pool(x)
        x = self.enc3(x); s2 = x.size(); x, idx3 = self.pool(x)
        x = self.enc4(x); s3 = x.size(); x, idx4 = self.pool(x)
        x = self.enc5(x); s4 = x.size(); x, idx5 = self.pool(x)

        x = self.unpool(x, idx5, output_size=s4); x = self.dec5(x)
        x = self.unpool(x, idx4, output_size=s3); x = self.dec4(x)
        x = self.unpool(x, idx3, output_size=s2); x = self.dec3(x)
        x = self.unpool(x, idx2, output_size=s1); x = self.dec2(x)
        x = self.unpool(x, idx1, output_size=s0); x = self.dec1(x)

        return self.final(x)


# =====================================================================
# Variant 2: modified SegNet-VGG19 with skip connections, 5x5 shallow-depth
# blocks, and a parallel-dilated-conv bottleneck (recommended default)
# =====================================================================
class ParallelDilatedBlock(nn.Module):
    """4 parallel dilated 3x3 convs (rates 1, 6, 12, 18) applied to the encoder
    bottleneck, concatenated, then projected back to the channel count the
    decoder's first unpool step expects (512, to match idx5/enc5)."""
    def __init__(self, in_ch=512, branch_ch=128, out_ch=512, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            _conv_bn_relu(in_ch, branch_ch, kernel_size=3, dilation=d) for d in dilations
        ])
        self.project = _conv_bn_relu(branch_ch * len(dilations), out_ch, kernel_size=1)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1)
        return self.project(x)


class SegNetVGG19Modified(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        stages = _load_vgg19_bn_stages(pretrained)
        self.enc1, self.enc2, self.enc3, self.enc4, self.enc5 = (
            stages["enc1"], stages["enc2"], stages["enc3"], stages["enc4"], stages["enc5"])

        # extra 5x5 conv block at the first two (shallow) encoder depths,
        # applied after the normal VGG 3x3 blocks, before pooling
        self.enc1_5x5 = _conv_bn_relu(64, 64, kernel_size=5)
        self.enc2_5x5 = _conv_bn_relu(128, 128, kernel_size=5)

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

        # bottleneck: parallel dilated convolutions (ASPP-style), rates 1/6/12/18
        self.aspp = ParallelDilatedBlock(in_ch=512, branch_ch=128, out_ch=512)

        # decoder blocks take CONCATENATED (unpooled + same-depth encoder skip) input
        self.dec5 = _dec_block(in_ch=512 + 512, mid_ch=512, out_ch=512, n_convs=4)   # + enc5 skip
        self.dec4 = _dec_block(in_ch=512 + 512, mid_ch=512, out_ch=256, n_convs=4)   # + enc4 skip
        self.dec3 = _dec_block(in_ch=256 + 256, mid_ch=256, out_ch=128, n_convs=4)   # + enc3 skip

        # last two decoder depths: extra 5x5 block BEFORE the normal 3x3 blocks
        self.dec2_5x5 = _conv_bn_relu(128 + 128, 128, kernel_size=5)                 # + enc2 skip
        self.dec2 = _dec_block(in_ch=128, mid_ch=128, out_ch=64, n_convs=2)

        self.dec1_5x5 = _conv_bn_relu(64 + 64, 64, kernel_size=5)                    # + enc1 skip
        self.dec1 = _dec_block(in_ch=64, mid_ch=64, out_ch=64, n_convs=2)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def set_encoder_trainable(self, trainable: bool):
        modules = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5,
                   self.enc1_5x5, self.enc2_5x5]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = trainable

    def forward(self, x):
        # ---- Encoder (save pre-pool feature maps as skip connections) ----
        e1 = self.enc1(x);  e1 = self.enc1_5x5(e1)
        p1, idx1 = self.pool(e1)

        e2 = self.enc2(p1); e2 = self.enc2_5x5(e2)
        p2, idx2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3, idx3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4, idx4 = self.pool(e4)

        e5 = self.enc5(p4)
        p5, idx5 = self.pool(e5)

        # ---- Bottleneck: multi-scale context via parallel dilated convs ----
        b = self.aspp(p5)

        # ---- Decoder (unpool with stored indices, fuse with encoder skip) ----
        d = self.unpool(b, idx5, output_size=e5.size())
        d = self.dec5(torch.cat([d, e5], dim=1))

        d = self.unpool(d, idx4, output_size=e4.size())
        d = self.dec4(torch.cat([d, e4], dim=1))

        d = self.unpool(d, idx3, output_size=e3.size())
        d = self.dec3(torch.cat([d, e3], dim=1))

        d = self.unpool(d, idx2, output_size=e2.size())
        d = self.dec2_5x5(torch.cat([d, e2], dim=1))
        d = self.dec2(d)

        d = self.unpool(d, idx1, output_size=e1.size())
        d = self.dec1_5x5(torch.cat([d, e1], dim=1))
        d = self.dec1(d)

        return self.final(d)  # raw logits; apply sigmoid outside (BCEWithLogits)


# =====================================================================
# Variant 3: SegFormer (Xie et al., 2021) via HuggingFace `transformers`.
# Requires: pip install transformers
#
# Architecturally unrelated to the SegNet variants above - hierarchical
# ViT encoder (MiT-b0..b5) + lightweight all-MLP decoder that fuses
# multi-scale features directly, so there's no max-pool-index plumbing
# to carry over. Logits come out at 1/4 input resolution; forward()
# bilinearly upsamples back to full size before returning, so it's a
# drop-in replacement for the SegNet variants (same call signature,
# same raw-logits output for BCEWithLogitsLoss).
#
# Pretrained checkpoints used here (`nvidia/segformer-b{size}-finetuned-
# ade-512-512`) come with a decode head already trained for (ADE20K)
# semantic segmentation, not just an ImageNet-pretrained encoder - a
# stronger starting point for dense prediction than VGG. The final
# classifier layer is reinitialized for num_classes via
# ignore_mismatched_sizes=True.
# =====================================================================
class SegFormerWrapper(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, size="b0"):
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
        except ImportError as e:
            raise ImportError(
                "SegFormer variant requires the `transformers` package. "
                "Install it with: pip install transformers"
            ) from e

        model_name = f"nvidia/segformer-{size}-finetuned-ade-512-512"
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                model_name, num_labels=num_classes, ignore_mismatched_sizes=True
            )
        else:
            config = SegformerConfig.from_pretrained(model_name, num_labels=num_classes)
            self.net = SegformerForSemanticSegmentation(config)

    def set_encoder_trainable(self, trainable: bool):
        # mirrors the SegNet variants' encoder-freeze/unfreeze API used by train.py;
        # here "encoder" is the MiT transformer backbone, decode head is left trainable
        for p in self.net.segformer.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        logits = self.net(pixel_values=x).logits  # (B, num_classes, H/4, W/4)
        return nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


def build_model(variant: str, num_classes: int = 1, pretrained: bool = True, segformer_size: str = "b0"):
    variant = variant.lower()
    if variant == "modified":
        return SegNetVGG19Modified(num_classes=num_classes, pretrained=pretrained)
    elif variant == "plain":
        return SegNetVGG19Plain(num_classes=num_classes, pretrained=pretrained)
    elif variant == "segformer":
        return SegFormerWrapper(num_classes=num_classes, pretrained=pretrained, size=segformer_size)
    else:
        raise ValueError(
            f"Unknown model variant '{variant}', expected 'modified', 'plain', or 'segformer'"
        )


if __name__ == "__main__":
    x = torch.randn(2, 3, 384, 384)
    for variant in ["plain", "modified", "segformer"]:
        m = build_model(variant, num_classes=1, pretrained=False)
        y = m(x)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"[{variant}] output shape: {tuple(y.shape)}, params: {n_params/1e6:.1f}M")
