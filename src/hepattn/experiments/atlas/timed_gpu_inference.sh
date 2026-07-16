#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${HOME}/projects/hepattn/src/hepattn/experiments/atlas"
REPO_ROOT="${HOME}/projects/hepattn"
NUM_GPUS="${1:-${NUM_GPUS:-2}}"

shopt -s nullglob
# ckpts=("${JZ_PATH_PARTITIONS}/../../results/glow_baseline/atlas_run4_jz1234_v1/epoch=027-val_loss=5.33018.ckpt")
# cfg=("${SCRIPT_DIR}/configs/glow_cpu_inference_override.yaml")
# ckpts=("${JZ_PATH_PARTITIONS}/../../results/glow_baseline/atlas_v1_nopart/epoch=199-val_loss=5.25962-43799.ckpt")
# cfg=("${SCRIPT_DIR}/configs/glow_gpu_inference_override.yaml")
# ckpts=("${JZ_PATH_PARTITIONS}/../../results/jz1234_v0_nopart_reproduce/atlas_jz1234_v10_nopart_reproduce_20260630-T221535/ckpts/epoch=016-val_loss=5.32814.ckpt")
# ckpts=("${CKPT_PATH_TUNING}/jz1234_v0_nopart_reproduce/atlas_jz1234_v0_nopart_reproduce_20260703-T235340/ckpts/epoch=049-val_loss=5.21033.ckpt")
# ckpts=("${CKPT_PATH_TUNING}/jz1234_v0_nopart_reproduce/atlas_jz1234_v0_nopart_reproduce_20260708-T180846/ckpts/epoch=065-val_loss=5.18897.ckpt")
# ckpts=("${CKPT_PATH_TUNING}/jz1234_v0_nopart_reproduce/atlas_jz1234_v0_nopart_reproduce_20260708-T180846/ckpts/epoch=035-val_loss=5.20390.ckpt") # this seemed best from Comet web UI plot
# ckpts=("${CKPT_PATH_TUNING}/MDN_jz1234_v0_nopart_reproduce/atlas_MDN_jz1234_v0_nopart_reproduce_20260714-T172348/ckpts/epoch=068-val_loss=4.95623.ckpt") 
ckpts=("${CKPT_PATH_TUNING}/MDN_jz1234_v0_nopart_reproduce/atlas_MDN_jz1234_v0_nopart_reproduce_20260714-T232307/ckpts/epoch=153-val_loss=0.74976.ckpt")
# cfg=("${SCRIPT_DIR}/configs/glow_gpu_inference_override.yaml")
cfg=("${SCRIPT_DIR}/configs/base_MDN_jz1234.yaml")
shopt -u nullglob

CKPT_PATH="${ckpts[0]}"
CFG_PATH="${cfg[0]}"
# DATA_PATH="${JZ_PATH_RAW}/user.edreyer.801168.Py8EG_A14NNPDF23LO_jj_JZ3.recon.AOD.e8481_s4383_r17563_v12_mltree.root/user.edreyer.50520198._000249.mltree.root"
# DATA_PATH="/fast_scratch_4/atlas/pflow/0pileup_JZ1-4_Run4_AOD_samples/partitions/GLOW/data/dijet_JZ1234_June2026_2000_performance_test/JZ1234_performance_test.root"
DATA_PATH="${JZ_PATH_PARTITIONS}/JZ1234_val.root"

cd "${SCRIPT_DIR}"

echo "Checkpoint: ${CKPT_PATH}"
echo "Config: ${CFG_PATH}"
echo "Input ROOT: ${DATA_PATH}"
echo "Output directory: $(dirname "${CKPT_PATH}")"
echo "GPU devices: ${NUM_GPUS}"
echo "Starting GPU inference at $(date -Is)"

start_seconds=$(date +%s)

# Keep comments outside the continued command; otherwise the config flag is not passed.
# Optional output suffix for tuned-model evaluation:
# --data.test_suff tuning
"${REPO_ROOT}/.hepattn/bin/python" -u main.py test \
    -c "${CFG_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --data.test_path "${DATA_PATH}" \
    --data.is_inference true
    # --trainer.devices "${NUM_GPUS}" \
    # # --data.num_test 10 \
    # --data.batch_size 8

end_seconds=$(date +%s)
duration=$((end_seconds - start_seconds))

printf 'Finished GPU inference at %s\n' "$(date -Is)"
printf 'Elapsed wall time: %02d:%02d:%02d (%d seconds)\n' \
    $((duration / 3600)) \
    $(((duration % 3600) / 60)) \
    $((duration % 60)) \
    "${duration}"
