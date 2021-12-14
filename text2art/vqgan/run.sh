#!/bin/bash -eux

SRC_DIR=$(dirname $(realpath $0))
CODE_DIR=$(realpath $SRC_DIR/../..)

EXPNAME=$(basename $BASH_SOURCE)
JOB_NAME=text2art
JOB_REASON="text2art"
JOB_QUEUE="aml-gpu.q@gpu-aa,aml-gpu.q@gpu-ab"
WORK_ROOT=/exp/$(whoami)/text2art

model_name=VQGAN
prompt=             # e.g. "A zombie walking through a rainforest #artstation"
name=               # e.g. zombie
seed=1
steps=500
# Size (w768 h432 works for widescreen)
width=480
height=480

. ./scripts/parse_options.sh || exit 1;

[ -z $prompt ] && echo "please provide a prompt" && exit 1
[ -z $name ] && echo "please provide a name" && exit 1

WORK_DIR=${WORK_ROOT}/model_${model_name}/${name}
VENV=$CODE_DIR/venv

vqgan_config="$CODE_DIR/models/VQGAN/vqgan_imagenet_f16_1024.yaml"
vqgan_checkpoint="$CODE_DIR/models/VQGAN/vqgan_imagenet_f16_1024.ckpt"

mkdir -p "$WORK_DIR"
( cd $CODE_DIR && echo "$(date -u) $(git describe --always --abbrev=40 --dirty)")>> "${WORK_DIR}"/git_sha
rsync --quiet -avhz --exclude-from "${CODE_DIR}/.gitignore" "$CODE_DIR"/* "$WORK_DIR"/code

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

echo "\$(date -u) starting \${JOB_ID}" >> ${WORK_DIR}/sge_job_id
source $VENV/bin/activate

if [[ ! -f ${WORK_DIR}/done ]]; then
    python3 -m text2art.vqgan.run \
        --prompts "$prompt" \
        --work_dir "$WORK_DIR" \
        --vqgan_config "$vqgan_config" \
        --vqgan_checkpoint "$vqgan_checkpoint" \
        --size "$width" "$height" \
        --steps "$steps" \
        --seed "$seed" 

    touch ${WORK_DIR}/done
fi

EOF
chmod +x "${WORK_DIR}"/run.qsh
qsub "${WORK_DIR}/run.qsh"

echo "All queued up"
