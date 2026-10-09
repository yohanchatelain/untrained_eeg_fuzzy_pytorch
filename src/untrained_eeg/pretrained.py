"""Pretrained checkpoints loaded into our plain-PyTorch models (no braindecode needed).

EEGNetv4: 12 motor-imagery checkpoints by Pierre Guetschel (MIT), one per MOABB dataset,
https://huggingface.co/PierreGtch/EEGNetv4. Each was trained on 3 channels (C3, Cz, C4) at
128 Hz. They have the same architecture as our `EEGNet`, so only the tensor names differ.
Download them with scripts/00_download_pretrained.sh. tests/test_pretrained.py checks that
our loaded copy gives exactly the same features as braindecode 1.8.1.
"""

from pathlib import Path

import torch

from .models import EEGNet

REPO = Path(__file__).resolve().parents[2]
PRETRAINED_DIR = REPO / "data" / "pretrained"  # data/ is never committed

EEGNETV4_DATASETS = ("AlexMI", "BNCI2014001", "BNCI2014004", "BNCI2015001", "BNCI2015004", "Cho2017",
                     "Lee2019_MI", "Ofner2017", "PhysionetMI", "Schirrmeister2017", "Weibo2014", "Zhou2016")
EEGNETV4_CHANNELS = ("C3", "Cz", "C4")
EEGNETV4_SFREQ = 128

# Checkpoint layer -> position in our EEGNet.features. The classifier is dropped: we only
# use the network as a feature extractor.
_EEGNETV4_LAYERS = {"conv_temporal": 0, "bnorm_temporal": 1, "conv_spatial": 2, "bnorm_1": 3,
                    "conv_separable_depth": 7, "conv_separable_point": 8, "bnorm_2": 9}


def eegnetv4_state_dict(state):
    """Rename an EEGNetv4 checkpoint's tensors to our EEGNet's names, without the classifier."""
    out = {}
    for key, value in state.items():
        layer, _, rest = key.partition(".")
        if layer == "conv_classifier":
            continue
        out[f"features.{_EEGNETV4_LAYERS[layer]}.{rest}"] = value
    return out


def load_eegnetv4(dataset, n_times, weights_dir=PRETRAINED_DIR):
    """EEGNet with the weights pretrained on `dataset`, in eval mode. Input: (batch, 3, n_times)."""
    if dataset not in EEGNETV4_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}, choose from {EEGNETV4_DATASETS}")
    path = Path(weights_dir) / "eegnetv4" / dataset / "model-params.pkl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: bash scripts/00_download_pretrained.sh")
    state = torch.load(path, map_location="cpu", weights_only=True)  # tensors only, no code
    model = EEGNet(len(EEGNETV4_CHANNELS), n_times)
    model.load_state_dict(eegnetv4_state_dict(state), strict=True)
    return model.eval()
