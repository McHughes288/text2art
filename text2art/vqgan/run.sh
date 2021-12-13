#!/bin/bash -eux

SRC_DIR=$(dirname $(realpath $0))
CODE_DIR=$(realpath $SRC_DIR/../..)

EXPNAME=$(basename $BASH_SOURCE)
JOB_NAME=text2art
JOB_REASON="text2art"
JOB_QUEUE="aml-gpu.q@gpu-aa,aml-gpu.q@gpu-ab"
WORK_ROOT=/exp/$(whoami)/text2art

msg="text2art"
seed=1

model_name=VQGAN

. ./scripts/parse_options.sh || exit 1;

if [[ "$model_name" == "VQGAN" ]]; then
    echo "VQGAN"
else
    echo "Model name called '$dataset' not supported"
    exit 1;
fi


WORK_DIR=${WORK_ROOT}/$(date +"%Y%m%d_%H%M%S")_model_${model_name}
VENV=$CODE_DIR/venv

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

if [[ ! -f ${WORK_DIR}/done ]]; then
    python3 -m text2art.vqgan.run
    touch ${WORK_DIR}/done
    esac
fi

EOF
chmod +x "${WORK_DIR}"/run.qsh
qsub "${WORK_DIR}/run.qsh"

echo "All queued up"
