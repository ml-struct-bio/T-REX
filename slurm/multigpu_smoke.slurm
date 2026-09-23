#!/usr/bin/env bash
#SBATCH --job-name=trex_multigpu_smoke
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Allocation-only smoke for T-REX worker scaling. Override --gres and
# TREX_WORKER_GPUS at submission to test a larger node. This does not launch an
# LLM or a molecular-design backend and therefore works with any CUDA GPU type.

set -euo pipefail

REPO="${TREX_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
PYTHON="${TREX_TEST_PYTHON:-python}"
TREX_WORKER_GPUS="${TREX_WORKER_GPUS:-1,2,3}"

[[ -r "${REPO}/trex/controller.py" ]] || {
  echo "[FATAL] submit from the T-REX root or set TREX_REPO_ROOT" >&2
  exit 2
}
[[ -x "${PYTHON}" ]] || {
  echo "[FATAL] test Python is not executable: ${PYTHON}" >&2
  exit 2
}
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "[FATAL] nvidia-smi is unavailable" >&2
  exit 2
}
if [[ ! "${TREX_WORKER_GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "[FATAL] TREX_WORKER_GPUS must be a comma-separated GPU index list" >&2
  exit 2
fi

IFS=, read -r -a WORKER_GPUS <<< "${TREX_WORKER_GPUS}"
VISIBLE_GPU_COUNT="$(nvidia-smi --list-gpus | wc -l)"
[[ "${VISIBLE_GPU_COUNT}" -gt 0 ]] || {
  echo "[FATAL] Slurm allocation exposes no GPUs" >&2
  exit 2
}
TREX_EXPECTED_WORKER_COUNT="${TREX_EXPECTED_WORKER_COUNT:-$((VISIBLE_GPU_COUNT - 1))}"
if [[ ! "${TREX_EXPECTED_WORKER_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[FATAL] TREX_EXPECTED_WORKER_COUNT must be a positive integer" >&2
  exit 2
fi
if (( ${#WORKER_GPUS[@]} != TREX_EXPECTED_WORKER_COUNT )); then
  echo "[FATAL] expected ${TREX_EXPECTED_WORKER_COUNT} worker GPUs for ${VISIBLE_GPU_COUNT} visible GPUs; got ${#WORKER_GPUS[@]}" >&2
  echo "[FATAL] export comma-separated TREX_WORKER_GPUS in the shell, then submit with --export=ALL" >&2
  echo "[FATAL] set TREX_EXPECTED_WORKER_COUNT to validate an intentional worker subset" >&2
  exit 2
fi

seen_indices=","
seen_uuids=","
for gpu in "${WORKER_GPUS[@]}"; do
  if [[ "${gpu}" == "0" ]]; then
    echo "[FATAL] GPU 0 is reserved for the local LLM in the production layout" >&2
    exit 2
  fi
  if [[ "${seen_indices}" == *,"${gpu}",* ]]; then
    echo "[FATAL] duplicate worker GPU index: ${gpu}" >&2
    exit 2
  fi
  if (( gpu >= VISIBLE_GPU_COUNT )); then
    echo "[FATAL] worker GPU ${gpu} is outside ${VISIBLE_GPU_COUNT} visible GPUs" >&2
    exit 2
  fi
  uuid="$(
    nvidia-smi --id="${gpu}" \
      --query-gpu=uuid --format=csv,noheader,nounits
  )"
  [[ "$(printf '%s\n' "${uuid}" | wc -l)" -eq 1 && -n "${uuid}" ]] || {
    echo "[FATAL] GPU ${gpu} did not resolve to exactly one device" >&2
    exit 2
  }
  if [[ "${seen_uuids}" == *,"${uuid}",* ]]; then
    echo "[FATAL] GPU ${gpu} resolved to duplicate UUID ${uuid}" >&2
    exit 2
  fi
  seen_indices+="${gpu},"
  seen_uuids+="${uuid},"
  echo "[trex-multigpu-smoke] worker_gpu=${gpu} uuid=${uuid}"
done

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export TREX_WORKER_GPUS
"${PYTHON}" -m pytest -q \
  "${REPO}/trex/tests/test_live_tick.py::test_worker_wall_gpu_count_scales_beyond_reference_pool" \
  "${REPO}/trex/tests/test_event_driven_controller.py" \
  "${REPO}/trex/tests/test_output_identity.py" \
  "${REPO}/trex/tests/test_output_identity_boundaries.py" \
  "${REPO}/trex/tests/test_selector.py::test_batch_diversity_caps_family_repetition"

echo "T-REX multi-GPU smoke passed: visible=${VISIBLE_GPU_COUNT} workers=${#WORKER_GPUS[@]}"
