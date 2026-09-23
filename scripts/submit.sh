#!/usr/bin/env bash
# Derive the GPU allocation and time request from .env, validate, then submit.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${TREX_ENV_FILE:-${REPO}/.env}"
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/submit.sh [--check] [sbatch options]"
  echo "Compatibility launcher; prefer: trex check/submit campaign.yaml"
  echo "Reads .env (or TREX_ENV_FILE). --check validates without submitting."
  echo "Set TARGET, TREX_NUM_GPUS, TREX_MAX_WALL_H and TREX_ARCHIVE_BASE in .env."
  echo "Pass site options such as --account, --partition and --qos to sbatch."
  exit 0
fi
CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then CHECK_ONLY=1; shift; fi
if [[ "${CHECK_ONLY}" == 1 && "$#" -ne 0 ]]; then
  echo "--check does not accept sbatch options" >&2; exit 2
fi
export TREX_REPO_ROOT="${REPO}"
if [[ "${TREX_SKIP_ENV_FILE:-0}" != "1" ]]; then
  [[ -r "${ENV_FILE}" ]] || { echo "Missing ${ENV_FILE}; copy .env.example and configure it." >&2; exit 2; }
  set -a
  source "${ENV_FILE}"
  set +a
fi
: "${TARGET:?set TARGET in .env}"
: "${TREX_CONTROLLER_PYTHON:?set TREX_CONTROLLER_PYTHON in .env}"
: "${TREX_ENABLED_FAMILIES:?set TREX_ENABLED_FAMILIES in .env}"
: "${TREX_ARCHIVE_BASE:?set TREX_ARCHIVE_BASE in .env}"
TREX_NUM_GPUS="${TREX_NUM_GPUS:-4}"
TREX_MAX_WALL_H="${TREX_MAX_WALL_H:-48.0}"
TREX_GPU_TYPE="${TREX_GPU_TYPE:-h100}"
if [[ ! "${TREX_NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || (( TREX_NUM_GPUS < 2 )); then
  echo "TREX_NUM_GPUS must be an integer >= 2: one LLM GPU plus worker GPUs." >&2
  exit 2
fi
if [[ ! "${TREX_GPU_TYPE}" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
  echo "TREX_GPU_TYPE must be a Slurm GPU type, for example h100." >&2; exit 2
fi
HEADROOM_MINUTES=$((30 + 10 * (TREX_NUM_GPUS - 1)))
if (( HEADROOM_MINUTES < 60 )); then HEADROOM_MINUTES=60; fi
if [[ ! "${TREX_MAX_WALL_H}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
   ! SLURM_MINUTES="$(awk -v hours="${TREX_MAX_WALL_H}" -v headroom="${HEADROOM_MINUTES}" 'BEGIN {
       minutes = hours * 60;
       if (minutes <= 0 || minutes > 2147483647 - headroom) exit 1;
       rounded = int(minutes);
       if (minutes > rounded) rounded++;
       printf "%.0f\n", rounded + headroom;
     }')"; then
  echo "TREX_MAX_WALL_H must be a positive campaign duration in hours." >&2; exit 2
fi
for option in "$@"; do
  case "${option}" in
    --gres|--gres=*|--gpus*|-G*|--time*|-t*|--nodes*|-N*|--export*|--wrap*)
      echo "Set GPU count/type and campaign hours in .env; the wrapper manages one-node allocation and export (received ${option})." >&2
      exit 2 ;;
  esac
done
WORKER_GPUS=()
for ((gpu=1; gpu<TREX_NUM_GPUS; gpu++)); do WORKER_GPUS+=("${gpu}"); done
export TREX_WORKER_GPUS="$(IFS=,; echo "${WORKER_GPUS[*]}")"
export TREX_CHARGED_GPUS="${TREX_NUM_GPUS}"
export TREX_REQUIRE_THREE_WORKERS=0
export TREX_MAX_WALL_H
[[ -x "${TREX_CONTROLLER_PYTHON}" ]] || { echo "Missing controller Python: ${TREX_CONTROLLER_PYTHON}" >&2; exit 2; }
cd "${REPO}"
export TREX_REPO_ROOT="${REPO}"
echo "Target: ${TARGET}; GPUs: ${TREX_NUM_GPUS} (1 LLM + ${#WORKER_GPUS[@]} workers)"
echo "Campaign hours: ${TREX_MAX_WALL_H}; Slurm minutes: ${SLURM_MINUTES} (includes ${HEADROOM_MINUTES} minutes for startup/shutdown)"
echo "Output directory: ${TREX_ARCHIVE_BASE}"
VALIDATION_ARGS=(--target "${TARGET}" --enabled-families "${TREX_ENABLED_FAMILIES}"
  --require-backends --require-model --verify-backend-revisions)
if [[ -n "${TREX_TARGET_CONFIG:-}" ]]; then VALIDATION_ARGS+=(--target-config "${TREX_TARGET_CONFIG}"); fi
if [[ -n "${TREX_TARGET_PDB:-}" ]]; then VALIDATION_ARGS+=(--target-pdb "${TREX_TARGET_PDB}"); fi
"${TREX_CONTROLLER_PYTHON}" -m trex.validation "${VALIDATION_ARGS[@]}"
if [[ "${CHECK_ONLY}" == 1 ]]; then exit 0; fi
mkdir -p slurm_logs
exec sbatch --export=ALL --nodes=1 --gres="gpu:${TREX_GPU_TYPE}:${TREX_NUM_GPUS}" \
  --time="${SLURM_MINUTES}" "$@" "${REPO}/slurm/T-REX.slurm"
