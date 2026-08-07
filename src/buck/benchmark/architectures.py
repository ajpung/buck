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
    model = spec["fn"](weights="DEFAULT" if pretrained else None)
    freeze = spec.get("freeze", 2)

    # --- Attach the head, dispatching on the actual module layout.
    if name.startswith(("vit_",)):
        in_features = model.heads.head.in_features
        model.heads.head = _head(in_features, num_classes, dropout)
    elif name.startswith(("swin", "maxvit")):
        # Swin exposes .head; MaxViT ends its classifier Sequential with Linear.
        if isinstance(getattr(model, "head", None), nn.Linear):
            model.head = _head(model.head.in_features, num_classes, dropout)
        elif isinstance(getattr(model, "classifier", None), nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = _head(in_features, num_classes, dropout)
        else:
            raise RuntimeError(f"cannot locate classifier head on {name}")
    elif name.startswith("convnext"):
        in_features = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            _head(in_features, num_classes, dropout),
        )
    elif name.startswith(("resnet", "regnet", "shufflenet")):
        model.fc = _head(model.fc.in_features, num_classes, dropout)
    elif name.startswith(("efficientnet", "mobilenet", "mnasnet")):
        in_features = _first_linear_in_features(model.classifier)
        model.classifier = _head(in_features, num_classes, dropout)
    elif name.startswith("densenet"):
        model.classifier = _head(model.classifier.in_features, num_classes, dropout)
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


def parameter_counts(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen