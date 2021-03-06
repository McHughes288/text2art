#!/bin/bash -eu

SRC_DIR=$(dirname $(realpath $0))
CODE_DIR=$(realpath $SRC_DIR/../..)

EXPNAME=$(basename $BASH_SOURCE)
JOB_NAME=text2art
JOB_REASON="text2art"
JOB_QUEUE="aml-gpu.q@gpu-aa,aml-gpu.q@gpu-ab,aml-gpu.q@b3,aml-gpu.q@b2"
WORK_ROOT=
WORK_DIR=

model_name=VQGAN
prompt=             # e.g. "A zombie walking through a rainforest #artstation"
image_name=               # e.g. zombie
seed=1
steps=500
# Size (w768 h432 works for widescreen)
width=480
height=480
clip_model=ViT-B/32
lr=0.05
init_image=
init_weight=0.01        # only used in conjunction with initial image
log_every=50

. ./scripts/parse_options.sh || exit 1;

[ -z "$prompt" ] && echo "please provide a prompt" && exit 1
[ -z "$image_name" ] && echo "please provide a name" && exit 1
[ ! -d "$CODE_DIR/models/$model_name" ] && echo "please download the generative model" && exit 1
[ ! -d "$CODE_DIR/models/ESRGAN" ] && echo "please download the enhancement model" && exit 1

clip_model_stripped=$(echo $clip_model | tr -d '/')
WORK_ROOT=${WORK_ROOT:-/exp/$(whoami)/text2art/$(date +"%Y%m%d")}
WORK_DIR=${WORK_DIR:-${WORK_ROOT}/model_${model_name}/${image_name}_${clip_model_stripped}_${width}x${height}_lr${lr}}
image_dir=$WORK_DIR/images
VENV=$CODE_DIR/venv

vqgan_config="$CODE_DIR/models/VQGAN/vqgan_imagenet_f16_1024.yaml"
vqgan_checkpoint="$CODE_DIR/models/VQGAN/vqgan_imagenet_f16_1024.ckpt"

mkdir -p "$WORK_DIR" "$image_dir"
rsync --quiet -avhz --exclude-from "${CODE_DIR}/.gitignore" "$CODE_DIR"/* "$WORK_DIR"/code
( cd $CODE_DIR && echo "$(date -u) $(git describe --always --abbrev=40 --dirty)")>> "${WORK_DIR}"/code/git_sha

# Setup the super resolution code
esrgan_dir="$WORK_DIR"/code/ESRGAN
if [ ! -d "$esrgan_dir" ]; then
    git clone --depth=1 https://github.com/xinntao/ESRGAN $esrgan_dir
    rm -rf $esrgan_dir/models $esrgan_dir/figures $esrgan_dir/.git
    ln -sf $CODE_DIR/models/ESRGAN $esrgan_dir/models
fi
rm -f $esrgan_dir/results/*.png $esrgan_dir/LR/*.png

init_image_arg=""
if [ -f "$init_image" ]; then
    cp $init_image $image_dir/init.png
    init_image_arg="--init_image $image_dir/init.png --init_weight $init_weight"
fi

cat <<EOF >"${WORK_DIR}"/run.qsh
#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q ${JOB_QUEUE}
#$ -N "${JOB_NAME}"
#$ -terse
#$ -w e
#$ -wd $WORK_DIR/code
#$ -l gpu=1
#$ -pe smp 1
#$ -notify
#$ -p 200
#$ -o ${WORK_DIR}/run.log
#$ -sync n
set -e pipefail;

# job info
hostname && date
echo
echo "sge_job_id:  \${JOB_ID}"
echo "sge_queue:   \${JOB_QUEUE}"
echo "user:        \${USER}"
echo "sge_tmp_dir: \${TMPDIR}"
echo "sge_request: \${REQUEST}"
echo "reason:      ${JOB_REASON}"
echo "sge_wd:      \$(pwd)"
echo "pstree_pid:  \$\$"
echo

source $VENV/bin/activate

if [[ ! -f ${WORK_DIR}/code/done_train ]]; then
    python3 -m text2art.vqgan.run \
        --prompts "$prompt" \
        --image_dir "$image_dir" \
        --vqgan_config "$vqgan_config" \
        --vqgan_checkpoint "$vqgan_checkpoint" \
        --clip_model $clip_model \
        --size "$width" "$height" \
        --steps "$steps" \
        --seed "$seed" \
        --lr "$lr" \
        --log_every "$log_every" \
        $init_image_arg
    cp -f ${image_dir}/final.png ${WORK_DIR}/${image_name}.png

    touch ${WORK_DIR}/code/done_train
fi

if [[ ! -f ${WORK_DIR}/code/done_enhance ]]; then
    cp -f $image_dir/final.png $esrgan_dir/LR
    (cd $esrgan_dir && python3 $esrgan_dir/test.py)
    mv $esrgan_dir/results/final_rlt.png $WORK_DIR/${image_name}_enhanced.png
    touch ${WORK_DIR}/code/done_enhance
fi

echo "Done"

EOF
chmod +x "${WORK_DIR}"/run.qsh
qsub "${WORK_DIR}/run.qsh"

echo "All queued up"
