#!/bin/bash
# Download the pretrained checkpoints into data/pretrained/ (never committed).
#   bash scripts/00_download_pretrained.sh
# EEGNetv4: 12 motor-imagery checkpoints, MIT license, https://huggingface.co/PierreGtch/EEGNetv4
# We only fetch the weights (model-params.pkl, read with torch.load(weights_only=True)),
# not kwargs.pkl: unpickling a downloaded file can run arbitrary code.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://huggingface.co/PierreGtch/EEGNetv4/resolve/main
for ds in AlexMI BNCI2014001 BNCI2014004 BNCI2015001 BNCI2015004 Cho2017 \
          Lee2019_MI Ofner2017 PhysionetMI Schirrmeister2017 Weibo2014 Zhou2016; do
    dest=data/pretrained/eegnetv4/$ds
    mkdir -p "$dest"
    curl -sfL "$URL/EEGNetv4_$ds/model-params.pkl" -o "$dest/model-params.pkl"
    echo "downloaded $dest"
done
