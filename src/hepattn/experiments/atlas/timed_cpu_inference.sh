#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${HOME}/projects/hepattn/src/hepattn/experiments/atlas"
REPO_ROOT="${HOME}/projects/hepattn"

shopt -s nullglob
# ckpts=("${REPO_ROOT}"/results/glow_baseline/epoch=027-val_loss=5.33018.ckpt)
# ckpts=("${JZ_PATH_RAW}/../results/glow_baseline/epoch=027-val_loss=5.33018.ckpt")
ckpts=("${CKPT_PATH_TUNING}/jz1234_v0_nopart_reproduce/atlas_jz1234_v0_nopart_reproduce_20260703-T235340/ckpts/epoch=039-val_loss=5.25750.ckpt")
shopt -u nullglob

CKPT_PATH="${ckpts[0]}"
# DATA_PATH="${JZ_PATH_RAW}/user.edreyer.801168.Py8EG_A14NNPDF23LO_jj_JZ3.recon.AOD.e8481_s4383_r17563_v12_mltree.root/user.edreyer.50520198._000249.mltree.root"
# DATA_PATH="/fast_scratch_4/atlas/pflow/0pileup_JZ1-4_Run4_AOD_samples/partitions/GLOW/data/dijet_JZ1234_June2026_2000_performance_test/JZ1234_performance_test.root"
DATA_PATH="${JZ_PATH_PARTITIONS}/JZ1234_val.root"

cd "${SCRIPT_DIR}"

echo "Checkpoint: ${CKPT_PATH}"
echo "Input ROOT: ${DATA_PATH}"
echo "Output directory: $(dirname "${CKPT_PATH}")"
echo "Starting CPU inference at $(date -Is)"

start_seconds=$(date +%s)

# Keep comments outside the continued command; otherwise the config flag is not passed.
"${REPO_ROOT}/.hepattn/bin/python" -u main.py test \
    -c configs/glow_cpu_inference_override.yaml \
    --ckpt_path "${CKPT_PATH}" \
    --data.test_path "${DATA_PATH}" \
    --data.test_suff baseline \
    # # --data.num_test 10 \
    # --data.batch_size 8

end_seconds=$(date +%s)
duration=$((end_seconds - start_seconds))

printf 'Finished CPU inference at %s\n' "$(date -Is)"
printf 'Elapsed wall time: %02d:%02d:%02d (%d seconds)\n' \
    $((duration / 3600)) \
    $(((duration % 3600) / 60)) \
    $((duration % 60)) \
    "${duration}"
