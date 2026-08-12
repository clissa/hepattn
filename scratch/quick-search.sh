#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/lclissa/projects/hepattn
ATLAS_DIR="${REPO_ROOT}/src/hepattn/experiments/atlas"
# Update this identifier and the experiment list below for each new round.
ROUND=round_3
CONFIG_DIR="${ATLAS_DIR}/configs/configs_queue/${ROUND}"
LOG_DIR="${REPO_ROOT}/logs/quick-search/${ROUND}"
PYTHON="${REPO_ROOT}/.hepattn/bin/python"
BASE_PORT="${QS_BASE_PORT:-29500}"

EXPERIMENTS=(
  qs19_null04_incaux
  qs20_null035_incaux
  qs21_incaux_detw30
)
GPU_PAIRS=("0,1" "2,3" "4,5")

dry_run=false
overwrite_logs=false
for argument in "$@"; do
  case "${argument}" in
    --dry-run) dry_run=true ;;
    --overwrite-logs) overwrite_logs=true ;;
    *)
      echo "ERROR: unknown argument: ${argument}" >&2
      echo "usage: $0 [--dry-run] [--overwrite-logs]" >&2
      exit 2
      ;;
  esac
done

if [[ ! "${BASE_PORT}" =~ ^[0-9]+$ ]] || ((BASE_PORT < 1 || BASE_PORT > 65533)); then
  echo "ERROR: QS_BASE_PORT must be an integer from 1 through 65533" >&2
  exit 2
fi

declare -A seen_names=()
declare -A seen_roots=()
declare -A seen_ports=()
for slot in "${!GPU_PAIRS[@]}"; do
  port=$((BASE_PORT + slot))
  if [[ -n "${seen_ports[${port}]:-}" ]]; then
    echo "ERROR: duplicate master port: ${port}" >&2
    exit 1
  fi
  seen_ports["${port}"]=1
done

for experiment_id in "${EXPERIMENTS[@]}"; do
  config="${CONFIG_DIR}/${experiment_id}.yaml"
  log="${LOG_DIR}/${experiment_id}.log"
  if [[ ! -f "${config}" ]]; then
    echo "ERROR: missing config: ${config}" >&2
    exit 1
  fi

  config_name=$(awk '$1 == "name:" {print $2; exit}' "${config}")
  output_root=$(awk '$1 == "default_root_dir:" {print $2; exit}' "${config}")
  if [[ "${config_name}" != "${experiment_id}" ]]; then
    echo "ERROR: config name mismatch in ${config}: ${config_name}" >&2
    exit 1
  fi
  if [[ "${output_root##*/}" != "${experiment_id}" ]]; then
    echo "ERROR: output basename mismatch in ${config}: ${output_root}" >&2
    exit 1
  fi
  if [[ -n "${seen_names[${config_name}]:-}" ]]; then
    echo "ERROR: duplicate config name: ${config_name}" >&2
    exit 1
  fi
  if [[ -n "${seen_roots[${output_root}]:-}" ]]; then
    echo "ERROR: duplicate output directory: ${output_root}" >&2
    exit 1
  fi
  seen_names["${config_name}"]=1
  seen_roots["${output_root}"]=1

  if ! ${dry_run}; then
    if [[ -e "${output_root}" ]]; then
      echo "ERROR: output directory already exists; refusing to overwrite checkpoints: ${output_root}" >&2
      exit 1
    fi
    if [[ -e "${log}" ]] && ! ${overwrite_logs}; then
      echo "ERROR: log already exists (pass --overwrite-logs to replace it): ${log}" >&2
      exit 1
    fi
  fi
done

if ! ${dry_run}; then
  if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: repository Python is missing or not executable: ${PYTHON}" >&2
    exit 1
  fi
  command -v setsid >/dev/null || { echo "ERROR: setsid is required" >&2; exit 1; }
  command -v tee >/dev/null || { echo "ERROR: tee is required" >&2; exit 1; }
  mkdir -p "${LOG_DIR}"
fi

active_pids=()
active_ids=()

terminate_children() {
  local signal_name=$1
  local exit_status=$2
  trap - INT TERM
  echo "Received ${signal_name}; terminating active process groups" >&2
  for pid in "${active_pids[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  for pid in "${active_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  exit "${exit_status}"
}

trap 'terminate_children SIGINT 130' INT
trap 'terminate_children SIGTERM 143' TERM

cd "${ATLAS_DIR}"
wave=0
for ((wave_start = 0; wave_start < ${#EXPERIMENTS[@]}; wave_start += 3)); do
  ((wave += 1))
  wave_end=$((wave_start + 3))
  if ((wave_end > ${#EXPERIMENTS[@]})); then
    wave_end=${#EXPERIMENTS[@]}
  fi
  echo "=== wave ${wave}: jobs $((wave_start + 1))-${wave_end} of ${#EXPERIMENTS[@]} ==="

  active_pids=()
  active_ids=()
  for ((index = wave_start; index < wave_end; index += 1)); do
    slot=$((index - wave_start))
    experiment_id=${EXPERIMENTS[index]}
    gpu_pair=${GPU_PAIRS[slot]}
    port=$((BASE_PORT + slot))
    relative_config="configs/configs_queue/${ROUND}/${experiment_id}.yaml"
    log="${LOG_DIR}/${experiment_id}.log"
    started_at=$(date --iso-8601=seconds)

    printf 'start=%s id=%s config=%s gpus=%s port=%s log=%s\n' \
      "${started_at}" "${experiment_id}" "${relative_config}" "${gpu_pair}" "${port}" "${log}"
    if ${dry_run}; then
      printf '  MASTER_ADDR=127.0.0.1 MASTER_PORT=%q CUDA_VISIBLE_DEVICES=%q %q -u main.py fit -c %q 2>&1 | tee %q\n' \
        "${port}" "${gpu_pair}" "${PYTHON}" "${relative_config}" "${log}"
      continue
    fi

    setsid bash -c '
      set -o pipefail
      MASTER_ADDR=127.0.0.1 MASTER_PORT="$1" CUDA_VISIBLE_DEVICES="$2" \
        "$3" -u main.py fit -c "$4" 2>&1 | tee "$5"
    ' quick-search-job "${port}" "${gpu_pair}" "${PYTHON}" "${relative_config}" "${log}" &
    active_pids+=("$!")
    active_ids+=("${experiment_id}")
  done

  if ${dry_run}; then
    continue
  fi

  failures=()
  for job_index in "${!active_pids[@]}"; do
    pid=${active_pids[job_index]}
    experiment_id=${active_ids[job_index]}
    if wait "${pid}"; then
      echo "completed id=${experiment_id} status=0"
    else
      status=$?
      failures+=("${experiment_id}:${status}")
      echo "FAILED id=${experiment_id} status=${status}" >&2
    fi
  done
  active_pids=()
  active_ids=()

  if ((${#failures[@]} > 0)); then
    echo "ERROR: wave ${wave} failed; later waves were not started: ${failures[*]}" >&2
    exit 1
  fi
done

if ${dry_run}; then
  echo "Dry run complete: ${#EXPERIMENTS[@]} configs in ${wave} waves; no training processes started."
else
  echo "Quick search complete: all ${#EXPERIMENTS[@]} jobs succeeded."
fi
