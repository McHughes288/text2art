#!/bin/bash
set -euo pipefail

SRC_DIR=$(dirname $(realpath $0))
CODE_DIR=$(realpath $SRC_DIR/..)

. ./scripts/parse_options.sh || exit 1;

model_dir=$CODE_DIR/models/VQGAN
if [ ! -f $model_dir/done ]; then
    echo "Downloading VQGAN model..."
    mkdir -p $model_dir
    curl -L 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' > $model_dir/vqgan_imagenet_f16_1024.yaml
    curl -L 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1' > $model_dir/vqgan_imagenet_f16_1024.ckpt
    curl -L 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' > $model_dir/vqgan_imagenet_f16_16384.yaml
    curl -L 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' > $model_dir/vqgan_imagenet_f16_16384.ckpt
    touch $model_dir/done
fi

model_dir=$CODE_DIR/models/ESRGAN
if [ ! -f $model_dir/done ]; then
    mkdir -p $model_dir
    pushd $model_dir
    gdown --id 1pJ_T-V1dpb1ewoEra1TGSWl5e6H7M4NN
    gdown --id 1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene
    gdown --id 1mSJ6Z40weL-dnPvi390xDd3uZBCFMeqr
    gdown --id 1MJFgqXJrMkPdKtiuy7C6xfsU1QIbXEb-
    touch $model_dir/done
    popd
fi

if [ ! -f $CODE_DIR/ESRGAN/done ]; then
    git clone https://github.com/xinntao/ESRGAN $CODE_DIR/ESRGAN
    rm -rf $CODE_DIR/ESRGAN/models
    rm -f $CODE_DIR/ESRGAN/results/*.png
    rm -f $CODE_DIR/ESRGAN/LR/*.png
    ln -s $CODE_DIR/models/ESRGAN $CODE_DIR/ESRGAN/models
    touch $CODE_DIR/ESRGAN/done
fi
