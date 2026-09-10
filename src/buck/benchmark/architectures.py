"""Architecture registry for the benchmark.

The previous sweep declared 33 architectures but silently ran 13: the
transformer families require a fixed 224x224 input, the sweep fed them
600x600, and the resulting exception was swallowed by a bare
``except Exception: continue``. Every ViT, Swin, MaxViT, ConvNeXt, DenseNet,
MobileNet and RegNet result was therefore missing from a comparison that
appeared complete.

Two things prevent a repeat. Each entry declares its own input size, and
``fixed_input`` marks the models whose patch embedding or relative-position
tables cannot be resized. Construction failures raise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


# ``size`` is the resolution the model is evaluated at. For fixed_input models
# it is mandatory and cannot be overridden. For the rest it is a sensible
# default that the CLI may raise or lower.
#
# ``batch`` targets roughly 24 GB of VRAM at the listed size; scale with
# --batch-scale on smaller cards.
REGISTRY = {
    # --- EfficientNet: sizes are the resolutions each variant was trained at.
    "efficientnet_b0": dict(fn=tvm.efficientnet_b0, size=224, batch=64, freeze=3),
    "efficientnet_b1": dict(fn=tvm.efficientnet_b1, size=240, batch=48, freeze=3),
    "efficientnet_b2": dict(fn=tvm.efficientnet_b2, size=260, batch=40, freeze=3),
    "efficientnet_b3": dict(fn=tvm.efficientnet_b3, size=300, batch=32, freeze=3),
    "efficientnet_b4": dict(fn=tvm.efficientnet_b4, size=380, batch=24, freeze=3),
    "efficientnet_b5": dict(fn=tvm.efficientnet_b5, size=456, batch=16, freeze=3),
    "efficientnet_b6": dict(fn=tvm.efficientnet_b6, size=528, batch=12, freeze=3),
    "efficientnet_b7": dict(fn=tvm.efficientnet_b7, size=600, batch=8, freeze=3),
    "efficientnet_v2_s": dict(fn=tvm.efficientnet_v2_s, size=384, batch=24, freeze=3),
    "efficientnet_v2_m": dict(fn=tvm.efficientnet_v2_m, size=480, batch=12, freeze=3),
    # --- ResNet
    "resnet18": dict(fn=tvm.resnet18, size=224, batch=128, freeze=2),
    "resnet34": dict(fn=tvm.resnet34, size=224, batch=96, freeze=2),
    "resnet50": dict(fn=tvm.resnet50, size=224, batch=64, freeze=2),
    "resnet101": dict(fn=tvm.resnet101, size=224, batch=48, freeze=2),
    "resnet152": dict(fn=tvm.resnet152, size=224, batch=32, freeze=2),
    # --- DenseNet
    "densenet121": dict(fn=tvm.densenet121, size=224, batch=48, freeze=2),
    "densenet169": dict(fn=tvm.densenet169, size=224, batch=40, freeze=2),
    "densenet201": dict(fn=tvm.densenet201, size=224, batch=32, freeze=2),
    # --- MobileNet
    "mobilenet_v2": dict(fn=tvm.mobilenet_v2, size=224, batch=96, freeze=3),
    "mobilenet_v3_small": dict(fn=tvm.mobilenet_v3_small, size=224, batch=128, freeze=3),
    "mobilenet_v3_large": dict(fn=tvm.mobilenet_v3_large, size=224, batch=96, freeze=3),
    # --- Sub-10MB backbones designed for on-device inference. These are the
    # candidates when download size dominates the decision.
    "shufflenet_v2_x0_5": dict(fn=tvm.shufflenet_v2_x0_5, size=224, batch=128, freeze=2),
    "shufflenet_v2_x1_0": dict(fn=tvm.shufflenet_v2_x1_0, size=224, batch=128, freeze=2),
    "shufflenet_v2_x1_5": dict(fn=tvm.shufflenet_v2_x1_5, size=224, batch=96, freeze=2),
    "shufflenet_v2_x2_0": dict(fn=tvm.shufflenet_v2_x2_0, size=224, batch=80, freeze=2),
    "mnasnet0_5": dict(fn=tvm.mnasnet0_5, size=224, batch=128, freeze=3),
    "mnasnet0_75": dict(fn=tvm.mnasnet0_75, size=224, batch=112, freeze=3),
    "mnasnet1_0": dict(fn=tvm.mnasnet1_0, size=224, batch=96, freeze=3),
    # --- RegNet
    "regnet_y_400mf": dict(fn=tvm.regnet_y_400mf, size=224, batch=96, freeze=2),
    "regnet_y_800mf": dict(fn=tvm.regnet_y_800mf, size=224, batch=80, freeze=2),
    "regnet_y_1_6gf": dict(fn=tvm.regnet_y_1_6gf, size=224, batch=64, freeze=2),
    "regnet_y_3_2gf": dict(fn=tvm.regnet_y_3_2gf, size=224, batch=48, freeze=2),
    # --- ConvNeXt
    "convnext_tiny": dict(fn=tvm.convnext_tiny, size=224, batch=48, freeze=2),
    "convnext_small": dict(fn=tvm.convnext_small, size=224, batch=40, freeze=2),
    "convnext_base": dict(fn=tvm.convnext_base, size=224, batch=32, freeze=2),
    # --- Transformers. These never ran in the previous sweep.
    "swin_t": dict(fn=tvm.swin_t, size=224, batch=48, freeze=2, fixed_input=True),
    "swin_s": dict(fn=tvm.swin_s, size=224, batch=32, freeze=2, fixed_input=True),
    "swin_b": dict(fn=tvm.swin_b, size=224, batch=24, freeze=2, fixed_input=True),
    "swin_v2_t": dict(fn=tvm.swin_v2_t, size=256, batch=40, freeze=2, fixed_input=True),
    "vit_b_16": dict(fn=tvm.vit_b_16, size=224, batch=48, freeze=6, fixed_input=True),
    "vit_b_32": dict(fn=tvm.vit_b_32, size=224, batch=64, freeze=6, fixed_input=True),
    "maxvit_t": dict(fn=tvm.maxvit_t, size=224, batch=24, freeze=2, fixed_input=True),
}

# A spread of families and capacities that runs in reasonable time. Good
# default when you want a comparison rather than an exhaustive sweep.
DEFAULT_SUITE = [
    "efficientnet_b0",
    "efficientnet_b3",
    "efficientnet_v2_s",
    "resnet18",
    "resnet50",
    "densenet121",
    "mobilenet_v3_large",
    "regnet_y_1_6gf",
    "convnext_tiny",
    "swin_t",
    "vit_b_16",
    "maxvit_t",
]

# Candidates when the model has to ship to a browser or a modest web server.
# Every entry here is a modern, ImageNet-competitive backbone that stays small;
# the point of benchmarking them together is to find where BUCK's accuracy
# actually starts to fall off as capacity drops.
EFFICIENT_SUITE = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x2_0",
    "mnasnet1_0",
    "efficientnet_b0",
    "efficientnet_v2_s",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "resnet18",
    "convnext_tiny",
]


# --- Self-supervised backbones, built through timm rather than torchvision.
#
# Every entry above is ImageNet *supervised*, so the 12-model suite spanned a
# 4x parameter range but only one pretraining regime. DINOv3 is a different
# regime rather than another point on the same axis: self-supervised on
# LVD-1689M. ``convnext_tiny_dinov3`` is deliberately the same architecture,
# parameter count and 224px input as the torchvision ``convnext_tiny`` entry,
# so running the two against each other isolates pretraining as the only
# variable -- and it ships the same browser artefact either way.
#
# Their pretrained configs declare ImageNet mean/std at 224px, identical to
# IMAGENET_MEAN/IMAGENET_STD in data.py, so no preprocessing change is needed.
# That is checked at build time rather than assumed: a backbone wanting
# different normalisation would be silently degraded by the fixed constants in
# ``_to_tensor()`` and would look like a failed experiment. See
# ``_check_normalisation``.
#
# The DINOv2 ViTs are deliberately absent. They are patch-14 at a native 518px
# and would need position-embedding interpolation to run at 224, which is a
# real change to the model rather than a registry entry.
REGISTRY.update({
    "convnext_tiny_dinov3": dict(timm="convnext_tiny.dinov3_lvd1689m",
                                 size=224, batch=48, freeze=2),
    "convnext_small_dinov3": dict(timm="convnext_small.dinov3_lvd1689m",
                                  size=224, batch=40, freeze=2),
})

# The controlled pair: identical architecture and cost, different pretraining.
DINO_SUITE = [
    "convnext_tiny",
    "convnext_tiny_dinov3",
]


# --- The wide field: modern architectures torchvision does not ship.
#
# torchvision's 42 entries are one vendor's slice of 2015-2022, all pretrained
# the same way. These 41 add the families that came after, plus -- more
# importantly -- pretraining regimes the torchvision set cannot express:
# masked-autoencoder (MAE, FCMAE, BEiT-v2), image-text contrastive (CLIP,
# SigLIP), self-distillation (DINOv3), 21k-class supervision, and SSLD/USI
# distillation. Pretraining is the axis the original 12-model suite held
# constant while varying capacity 4x, and capacity turned out not to matter.
#
# Several of these do NOT use ImageNet normalisation -- see normalisation().
REGISTRY.update({
    # -- modern convnets
    "convnextv2_nano": dict(timm="convnextv2_nano.fcmae_ft_in22k_in1k", size=224, batch=48, freeze=2),
    "convnextv2_tiny": dict(timm="convnextv2_tiny.fcmae_ft_in22k_in1k", size=224, batch=48, freeze=2),
    "resnext50_32x4d": dict(timm="resnext50_32x4d.a1h_in1k", size=224, batch=48, freeze=2),
    "seresnext50_32x4d": dict(timm="seresnext50_32x4d.racm_in1k", size=224, batch=48, freeze=2),
    "res2net50_26w_4s": dict(timm="res2net50_26w_4s.in1k", size=224, batch=48, freeze=2),
    "resnetrs50": dict(timm="resnetrs50.tf_in1k", size=160, batch=48, freeze=2),
    "regnetz_d8": dict(timm="regnetz_d8.ra3_in1k", size=256, batch=38, freeze=2),
    "dpn68b": dict(timm="dpn68b.ra_in1k", size=224, batch=48, freeze=2),
    "hgnetv2_b4": dict(timm="hgnetv2_b4.ssld_stage2_ft_in1k", size=224, batch=48, freeze=2),
    "mixnet_l": dict(timm="mixnet_l.ft_in1k", size=224, batch=48, freeze=2),
    "ghostnetv2_130": dict(timm="ghostnetv2_130.in1k", size=224, batch=48, freeze=2),
    "inception_v4": dict(timm="inception_v4.tf_in1k", size=299, batch=19, freeze=2),
    "xception41": dict(timm="xception41.tf_in1k", size=299, batch=28, freeze=2),

    # -- efficient / mobile-class, mostly distilled
    "repvit_m1_5": dict(timm="repvit_m1_5.dist_300e_in1k", size=224, batch=48, freeze=2, fixed_input=True),
    "edgenext_small": dict(timm="edgenext_small.usi_in1k", size=256, batch=38, freeze=2, fixed_input=True),
    "efficientformerv2_s2": dict(timm="efficientformerv2_s2.snap_dist_in1k", size=224, batch=48, freeze=2, fixed_input=True),
    "fastvit_sa24": dict(timm="fastvit_sa24.apple_in1k", size=256, batch=38, freeze=2, fixed_input=True),
    "mobilevitv2_150": dict(timm="mobilevitv2_150.cvnets_in22k_ft_in1k", size=256, batch=38, freeze=2, fixed_input=True),
    "levit_256": dict(timm="levit_256.fb_dist_in1k", size=224, batch=48, freeze=2, fixed_input=True),
    "tiny_vit_21m": dict(timm="tiny_vit_21m_224.dist_in22k_ft_in1k", size=224, batch=48, freeze=2, fixed_input=True),

    # -- transformers and hybrids
    "deit3_small": dict(timm="deit3_small_patch16_224.fb_in22k_ft_in1k", size=224, batch=48, freeze=2, fixed_input=True),
    "pit_s": dict(timm="pit_s_224.in1k", size=224, batch=48, freeze=2, fixed_input=True),
    "crossvit_15": dict(timm="crossvit_15_240.in1k", size=240, batch=38, freeze=2, fixed_input=True),
    "twins_svt_small": dict(timm="twins_svt_small.in1k", size=224, batch=48, freeze=2, fixed_input=True),
    "pvt_v2_b2": dict(timm="pvt_v2_b2.in1k", size=224, batch=48, freeze=2, fixed_input=True),
    "visformer_small": dict(timm="visformer_small.in1k", size=224, batch=48, freeze=2, fixed_input=True),
    "davit_tiny": dict(timm="davit_tiny.msft_in1k", size=224, batch=48, freeze=2, fixed_input=True),
    "gcvit_tiny": dict(timm="gcvit_tiny.in1k", size=224, batch=24, freeze=2, fixed_input=True),
    "focalnet_tiny_srf": dict(timm="focalnet_tiny_srf.ms_in1k", size=224, batch=48, freeze=2, fixed_input=True),
    "mvitv2_tiny": dict(timm="mvitv2_tiny.fb_in1k", size=224, batch=24, freeze=2, fixed_input=True),
    "nextvit_small": dict(timm="nextvit_small.bd_ssld_6m_in1k", size=224, batch=48, freeze=2, fixed_input=True),
    "swinv2_tiny_w8": dict(timm="swinv2_tiny_window8_256.ms_in1k", size=256, batch=38, freeze=2, fixed_input=True),
    "maxvit_tiny_tf": dict(timm="maxvit_tiny_tf_224.in1k", size=224, batch=24, freeze=2, fixed_input=True),
    "coatnet_0": dict(timm="coatnet_0_rw_224.sw_in1k", size=224, batch=24, freeze=2, fixed_input=True),
    "caformer_s18": dict(timm="caformer_s18.sail_in22k_ft_in1k", size=224, batch=48, freeze=2, fixed_input=True),
    "poolformerv2_s24": dict(timm="poolformerv2_s24.sail_in1k", size=224, batch=48, freeze=2, fixed_input=True),

    # -- alternative pretraining objectives on an identical ViT-B/16 body.
    # These four differ from each other ONLY in how they were pretrained,
    # which makes them the cleanest read on whether pretraining is the axis
    # that matters.
    "vit_b_16_mae": dict(timm="vit_base_patch16_224.mae", size=224, batch=32, freeze=6, fixed_input=True),
    "beitv2_base": dict(timm="beitv2_base_patch16_224.in1k_ft_in22k_in1k", size=224, batch=32, freeze=6, fixed_input=True),
    "vit_b_16_clip": dict(timm="vit_base_patch16_clip_224.laion2b_ft_in1k", size=224, batch=32, freeze=6, fixed_input=True),
    "vit_b_16_siglip": dict(timm="vit_base_patch16_siglip_224.webli", size=224, batch=24, freeze=6, fixed_input=True),
    "eva02_small": dict(timm="eva02_small_patch14_336.mim_in22k_ft_in1k", size=336, batch=21, freeze=6, fixed_input=True),
})

# Everything in the registry, alphabetical. Callers that care about coverage
# under truncation should pass an explicit cost-ordered list instead.
MEGA_SUITE = sorted(REGISTRY)


def input_size(name, override=None):
    """Resolution to feed ``name``, honouring an override only when legal."""
    spec = REGISTRY[name]
    if override is None:
        return spec["size"]
    if spec.get("fixed_input"):
        if override != spec["size"]:
            print(
                f"[arch] {name} has a fixed {spec['size']}px input; ignoring "
                f"--image-size {override} for this model"
            )
        return spec["size"]
    return override


def _first_linear_in_features(module):
    """Input width of the first Linear inside a classifier block.

    When the whole classifier is replaced, the head must match what the
    *backbone* emits, which is the first Linear's input width. Reading the last
    Linear instead silently mis-sizes any model with a multi-layer classifier:
    MobileNetV3 pools to 576 but ends 1024-wide, so a head built from the last
    layer fails the first forward pass.
    """
    if isinstance(module, nn.Linear):
        return module.in_features
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            return layer.in_features
    raise RuntimeError("no Linear layer found in classifier block")


def _head(in_features, num_classes, dropout):
    """Shared classifier head so architectures differ only in their backbone."""
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.5),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.25),
        nn.Linear(256, num_classes),
    )


# Marker attribute stamped on whatever module ``build_model`` newly attached.
# The trainer needs to tell freshly-initialised parameters from transferred
# ones so it can give them different learning rates, and the module object is
# the only reliable way to know: parameter *names* do not carry the
# distinction. torchvision's squeeze-excitation blocks are named ``fc1``/
# ``fc2``, so a name test for "fc" pulls 64 of EfficientNet-B0's 70
# head-group tensors, 108 of RegNet-Y-1.6GF's 114 and 32 of MobileNetV3-Large's
# 38 out of the backbone and trains them at the head rate. That silently gave
# every SE-based family a different optimisation regime from ConvNeXt, Swin,
# ViT, ResNet and DenseNet, which is most of EFFICIENT_SUITE -- the very table
# used to choose a model to ship.
HEAD_MARKER = "_buck_head"


def _mark(module):
    """Tag ``module`` as a newly-initialised head and return it."""
    setattr(module, HEAD_MARKER, True)
    return module


def head_parameter_ids(model):
    """``id()`` of every parameter inside a head attached by ``build_model``.

    Identity rather than name, so no backbone tensor can be captured by a
    coincidental substring.

    Raises:
        RuntimeError: no marked head found, meaning the model did not come
            from :func:`build_model`. Raised rather than falling back to a
            name test, which is the failure this replaces.
    """
    ids = set()
    for module in model.modules():
        if getattr(module, HEAD_MARKER, False):
            ids.update(id(p) for p in module.parameters())
    if not ids:
        raise RuntimeError(
            "no head marker found on this model; parameter groups cannot be "
            "split safely. Build it with buck.benchmark.architectures.build_model()."
        )
    return ids


def split_parameters(model):
    """Partition trainable parameters into (backbone, head) lists."""
    head_ids = head_parameter_ids(model)
    backbone, head = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (head if id(param) in head_ids else backbone).append(param)
    return backbone, head


def build_model(name, num_classes, dropout=0.3, pretrained=True):
    """Instantiate ``name`` with an ImageNet backbone and a fresh head.

    Raises:
        KeyError: unknown architecture.
        RuntimeError: the head could not be attached, meaning torchvision
            changed the module layout. Raised rather than skipped so the model
            cannot vanish from the leaderboard unnoticed.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"unknown architecture {name!r}; known: {sorted(REGISTRY)}"
        )

    spec = REGISTRY[name]
    freeze = spec.get("freeze", 2)

    # Self-supervised entries carry no classifier to replace and their module
    # layout does not match the name-prefix rules below, so they take their
    # own construction path.
    if "timm" in spec:
        return _build_timm(name, spec, num_classes, dropout, pretrained, freeze)

    model = spec["fn"](weights="DEFAULT" if pretrained else None)

    # --- Attach the head, dispatching on the actual module layout.
    if name.startswith(("vit_",)):
        in_features = model.heads.head.in_features
        model.heads.head = _mark(_head(in_features, num_classes, dropout))
    elif name.startswith(("swin", "maxvit")):
        # Swin exposes .head; MaxViT ends its classifier Sequential with Linear.
        if isinstance(getattr(model, "head", None), nn.Linear):
            model.head = _mark(_head(model.head.in_features, num_classes, dropout))
        elif isinstance(getattr(model, "classifier", None), nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = _mark(_head(in_features, num_classes, dropout))
        else:
            raise RuntimeError(f"cannot locate classifier head on {name}")
    elif name.startswith("convnext"):
        in_features = model.classifier[2].in_features
        # The whole block is marked, not just ``_head``: this LayerNorm
        # replaces ConvNeXt's own and starts from default init, so it is a
        # fresh parameter too and belongs at the head learning rate.
        model.classifier = _mark(nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            _head(in_features, num_classes, dropout),
        ))
    elif name.startswith(("resnet", "regnet", "shufflenet")):
        model.fc = _mark(_head(model.fc.in_features, num_classes, dropout))
    elif name.startswith(("efficientnet", "mobilenet", "mnasnet")):
        in_features = _first_linear_in_features(model.classifier)
        model.classifier = _mark(_head(in_features, num_classes, dropout))
    elif name.startswith("densenet"):
        model.classifier = _mark(_head(model.classifier.in_features, num_classes, dropout))
    else:
        raise RuntimeError(f"no head-attachment rule for {name}")

    # Freezing only makes sense over transferred weights. Without pretraining
    # the early blocks are random projections, and freezing them would starve
    # the model of its own stem rather than preserve anything.
    if pretrained:
        _freeze_stem(model, name, freeze)
    return model


def _freeze_stem(model, name, freeze):
    """Freeze the earliest ``freeze`` blocks; generic edge features transfer."""
    if freeze <= 0:
        return

    if name.startswith(("resnet",)):
        blocks = [model.conv1, model.bn1, model.layer1, model.layer2]
    elif name.startswith("shufflenet"):
        blocks = [model.conv1, model.stage2, model.stage3, model.stage4]
    elif name.startswith("mnasnet"):
        blocks = list(model.layers.children())
    elif name.startswith("regnet"):
        blocks = [model.stem]
        trunk = getattr(model, "trunk_output", None)
        if trunk is not None:
            blocks += list(trunk.children())
    elif name.startswith("vit_"):
        blocks = [model.conv_proj] + list(model.encoder.layers.children())
    elif name.startswith("maxvit"):
        blocks = [model.stem] + list(model.blocks.children())
    elif hasattr(model, "features"):
        blocks = list(model.features.children())
    else:
        # Unknown layout: leave everything trainable rather than freeze the
        # wrong thing silently.
        print(f"[arch] {name}: no freeze rule, training all layers")
        return

    for block in blocks[:freeze]:
        for param in block.parameters():
            param.requires_grad = False


# data.py normalises every image with these fixed constants, so a backbone
# trained under different ones cannot be fed correctly by this pipeline.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def normalisation(name):
    """The (mean, std) this backbone was pretrained under, as 3-tuples.

    The benchmark feeds these to ``TrainDataset``/``EvalDataset`` instead of
    assuming ImageNet everywhere. torchvision entries are all ImageNet, but
    timm's CLIP and SigLIP weights, and the TF-ported Inception/Xception
    families, use different constants. Feeding those ImageNet values does not
    raise -- the model trains, converges and produces a leaderboard row that
    understates it. Getting this wrong is invisible, so it is read from the
    checkpoint's own config rather than hard-coded.
    """
    spec = REGISTRY[name]
    if "timm" not in spec:
        return _IMAGENET_MEAN, _IMAGENET_STD

    import timm

    cfg = timm.get_pretrained_cfg(spec["timm"]).to_dict()
    mean = tuple(cfg.get("mean") or _IMAGENET_MEAN)
    std = tuple(cfg.get("std") or _IMAGENET_STD)
    return mean, std


def _check_normalisation(name, backbone):
    """Report when a backbone uses constants other than ImageNet's.

    Informational, not a warning: :func:`normalisation` plumbs the correct
    values through to the datasets. Printed so a non-ImageNet backbone is
    visible in the run log rather than silently different.
    """
    cfg = getattr(backbone, "pretrained_cfg", None) or {}
    mean, std = cfg.get("mean"), cfg.get("std")

    def differs(actual, expected):
        return actual is not None and any(
            abs(a - e) > 1e-3 for a, e in zip(actual, expected)
        )

    if differs(mean, _IMAGENET_MEAN) or differs(std, _IMAGENET_STD):
        print(f"[arch] {name}: non-ImageNet normalisation mean={mean} "
              f"std={std} (plumbed through to the datasets)")


class TimmClassifier(nn.Module):
    """A timm feature extractor plus the benchmark's shared head.

    The DINO checkpoints ship with ``num_classes=0``: they are backbones with
    no classifier at all, so there is nothing for the name-prefix dispatch in
    :func:`build_model` to replace. The backbone is therefore built pooled
    (output ``(N, C)``) and the same :func:`_head` every torchvision entry
    uses is attached on top, behind a fresh LayerNorm mirroring what the
    ``convnext`` branch does -- so the two arms differ in backbone weights and
    nothing else.

    Only ``classifier`` carries the head marker, so
    :func:`split_parameters` puts the backbone on the backbone learning rate
    exactly as it does elsewhere.
    """

    def __init__(self, backbone, num_classes, dropout, width):
        super().__init__()
        self.backbone = backbone
        self.classifier = _mark(nn.Sequential(
            nn.LayerNorm(width),
            _head(width, num_classes, dropout),
        ))

    def forward(self, x):
        return self.classifier(self.backbone(x))


# Head-side module names to exclude when walking a backbone for its trunk.
_TIMM_HEAD_NAMES = {
    "head", "classifier", "fc", "global_pool", "norm", "norm_pre",
    "fc_norm", "head_drop", "head_dist", "pre_logits", "flatten",
}


def _feature_width(backbone, size):
    """Width the backbone actually emits, measured rather than declared.

    ``num_features`` is not always the pooled output width -- GhostNet-V2
    declares 1248 and emits 1280 -- and a mismatch only surfaces as a shape
    error on the first forward pass, i.e. hours into a sweep. One dummy
    forward at build time costs nothing and removes the class of failure.
    """
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        out = backbone(torch.zeros(1, 3, size, size))
    if was_training:
        backbone.train()
    return int(out.shape[1])


def _build_timm(name, spec, num_classes, dropout, pretrained, freeze):
    """Construct a timm-backed entry with the shared head attached."""
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"{name} is a timm-backed entry and timm is not installed "
            f"(pip install timm). The torchvision entries do not need it."
        ) from exc

    backbone = timm.create_model(
        spec["timm"], pretrained=pretrained, num_classes=0
    )
    if pretrained:
        _check_normalisation(name, backbone)
    model = TimmClassifier(
        backbone, num_classes, dropout, _feature_width(backbone, spec["size"])
    )
    if pretrained:
        _freeze_timm_stem(backbone, name, freeze)
    return model


def _freeze_timm_stem(backbone, name, freeze):
    """Freeze the earliest ``freeze`` blocks of a timm backbone.

    timm's ConvNeXt exposes ``stem`` plus a four-stage ``stages``, so
    ``freeze=2`` freezes the stem and the first stage -- the same two blocks
    torchvision's ``features[:2]`` covers, keeping the arms comparable.
    """
    if freeze <= 0:
        return

    if hasattr(backbone, "stem") and hasattr(backbone, "stages"):
        blocks = [backbone.stem] + list(backbone.stages.children())
    elif hasattr(backbone, "patch_embed") and hasattr(backbone, "blocks"):
        blocks = [backbone.patch_embed] + list(backbone.blocks.children())
    else:
        # Generic fallback: walk the trunk in definition order. Without this,
        # a dozen timm families trained every layer while every torchvision
        # entry froze two blocks -- different optimisation regimes across the
        # same leaderboard, which is the defect HEAD_MARKER exists to prevent.
        trunk = [
            module
            for child_name, module in backbone.named_children()
            if child_name not in _TIMM_HEAD_NAMES
            and any(True for _ in module.parameters())
        ]
        if not trunk:
            print(f"[arch] {name}: no freezable trunk found, training all layers")
            return
        # A single Sequential trunk (``features``) is a stack of blocks, not
        # one block; expand it so ``freeze`` means the same thing everywhere.
        if len(trunk) == 1 and isinstance(trunk[0], nn.Sequential):
            trunk = list(trunk[0].children())
        blocks = trunk

    for block in blocks[:freeze]:
        for param in block.parameters():
            param.requires_grad = False


def parameter_counts(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen