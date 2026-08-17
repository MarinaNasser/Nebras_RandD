"""
model.py
RetinaNet (torchvision) with a ResNet-50 FPN backbone, pretrained on COCO,
fine-tuned for single-class polyp detection on REAL-Colon.

RetinaNet is a single-stage, anchor-based detector, chosen here specifically
because its focal loss down-weights the very large number of easy background
anchors at training time - which suits colonoscopy frames well, since most of
each frame is textureless mucosa and polyps are frequently small or flat
against that background. That's exactly the foreground/background imbalance
problem focal loss was designed to fix, and it tends to make RetinaNet more
sensitive to small/subtle objects than two-stage detectors at a given speed
budget.
"""
import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


NUM_CLASSES = 1  # single foreground class: "polyp". torchvision's RetinaNet does
                  # NOT count background as a class (unlike Faster R-CNN) - it uses
                  # independent per-class sigmoid focal loss, so no explicit
                  # "background" logit is needed.


def build_model(pretrained: bool = True, score_thresh: float = 0.3,
                 nms_thresh: float = 0.4, detections_per_img: int = 50):
    """
    Loads a COCO-pretrained RetinaNet (ResNet-50 FPN v2 backbone - transfer
    learning, not trained from scratch) and swaps in a fresh classification
    head sized for our 1-class problem, keeping the pretrained backbone/FPN
    weights intact.
    """
    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = retinanet_resnet50_fpn_v2(
        weights=weights,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
    )

    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES,
    )
    return model


def set_backbone_trainable(model, trainable: bool):
    """Freezes/unfreezes the ResNet-50 + FPN backbone. Used for the same
    freeze-then-unfreeze fine-tuning schedule as the SegNet project: a short
    warm-up with the pretrained backbone frozen so the new (randomly
    initialized) classification/regression heads catch up first, then
    unfreeze for full fine-tuning at a lower learning rate."""
    for p in model.backbone.parameters():
        p.requires_grad = trainable


if __name__ == "__main__":
    import torch
    m = build_model(pretrained=False)
    m.eval()
    x = [torch.randn(3, 512, 512)]
    with torch.no_grad():
        out = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params/1e6:.1f}M")
    print(f"output keys: {list(out[0].keys())}")
