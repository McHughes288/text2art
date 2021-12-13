#!/bin/bash
set -euo pipefail

SRC_DIR=$(dirname $(realpath $0))
CODE_DIR=$(realpath $SRC_DIR/..)

model_name=VQGAN


. ./scripts/parse_options.sh || exit 1;

if [ $model_name = "VQGAN" ]; then
    echo "Downloading VQGAN model..."
    model_dir=$CODE_DIR/models/VQGAN
    mkdir -p $model_dir
    curl -L 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' > $model_dir/vqgan_imagenet_f16_1024.yaml
    curl -L 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1' > $model_dir/vqgan_imagenet_f16_1024.ckpt
    curl -L 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' > $model_dir/vqgan_imagenet_f16_16384.yaml
    curl -L 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' > $model_dir/vqgan_imagenet_f16_16384.ckpt
fi