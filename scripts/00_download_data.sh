#!/bin/bash
# Download OpenNeuro ds003478 v1.1.0, run-01 files only (8.2 GB, no account needed).
#
# You usually do not need this: the scripts download the data into data/raw/ds003478
# themselves the first time. Use it to put one shared copy elsewhere, e.g. on Rorqual in a
# /project folder (docs/rorqual.md), then point the scripts to it:
#
#   bash scripts/00_download_data.sh /path/to/ds003478
#   export UNTRAINED_EEG_DATA=/path/to/ds003478
#
# Interrupted? Run it again: files already downloaded are skipped.
set -euo pipefail
DEST=${1:?usage: bash scripts/00_download_data.sh /path/to/ds003478}
cd "$(dirname "$0")/.."
python3 - "$DEST" <<'EOF'
import sys
sys.path.insert(0, "src")
from untrained_eeg.data import load_config
from untrained_eeg.download import download_openneuro

src = load_config()["openneuro"]
download_openneuro(src["id"], sys.argv[1], src["include"], src["version"])
EOF
echo "now run: export UNTRAINED_EEG_DATA=$DEST"
