#!/usr/bin/env bash
#SBATCH --job-name=trex_target
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=49:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G

# Submit with `trex submit campaign.yaml`. The CLI derives resources from YAML.
# Direct sbatch defaults above use GPU 0 for the LLM and GPUs 1-3 as workers.
# scripts/submit.sh remains the compatibility implementation used by the CLI.

set -euo pipefail
: "${TARGET:?set TARGET (for example: cd45 or betv1)}"

REPO="${TREX_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO}" || ! -r "${REPO}/trex/controller.py" ]]; then
  echo "[FATAL] submit from the T-REX root or set TREX_REPO_ROOT" >&2
  exit 2
fi

: "${TREX_ARCHIVE_BASE:?set TREX_ARCHIVE_BASE}"
TREX_COMPLEXA_REPO="${TREX_COMPLEXA_REPO:-${REPO}/external/Proteina-Complexa}"
TREX_BINDCRAFT_REPO="${TREX_BINDCRAFT_REPO:-${REPO}/external/BindCraft}"
TREX_BINDCRAFT_ENV="${TREX_BINDCRAFT_ENV:-${TREX_BINDCRAFT_REPO}/.venv}"
TREX_BOLTZGEN_REPO="${TREX_BOLTZGEN_REPO:-${REPO}/external/BoltzGen}"
TREX_BOLTZGEN_BIN="${TREX_BOLTZGEN_BIN:-${TREX_BOLTZGEN_REPO}/.venv/bin/boltzgen}"
: "${TREX_VLLM_VENV:?set TREX_VLLM_VENV}"
: "${TREX_QWEN_MODEL_PATH:?set TREX_QWEN_MODEL_PATH}"

TREX_LEGACY_COMPLEXA_REPO="${TREX_LEGACY_COMPLEXA_REPO:-${TREX_COMPLEXA_REPO}}"
TREX_COMPLEXA_PYTHON="${TREX_COMPLEXA_PYTHON:-${TREX_COMPLEXA_REPO}/.venv/bin/python}"
TREX_CONTROLLER_PYTHON="${TREX_CONTROLLER_PYTHON:-${TREX_VLLM_VENV}/bin/python}"
TREX_MODEL_MANIFEST="${TREX_MODEL_MANIFEST:-${REPO}/config/trex/qwen3_6_27b_fp8_model_manifest.json}"
TREX_SERVED_MODEL_NAME="${TREX_SERVED_MODEL_NAME:-Qwen/Qwen3.6-27B-FP8}"
TREX_LLM_MODEL="${TREX_LLM_MODEL:-vllm/${TREX_SERVED_MODEL_NAME}}"
TREX_VLLM_PORT="${TREX_VLLM_PORT:-$((12000 + (${SLURM_JOB_ID:-0} % 1000)))}"
# Cumulative controller limit; allocation includes startup/shutdown headroom.
TREX_MAX_WALL_H="${TREX_MAX_WALL_H:-48.0}"
TREX_SEED="${TREX_SEED:-0}"
TREX_WORKER_GPUS="${TREX_WORKER_GPUS:-1,2,3}"
TREX_CHARGED_GPUS="${TREX_CHARGED_GPUS:-}"
TREX_ENABLED_FAMILIES="${TREX_ENABLED_FAMILIES:-bindcraft,boltzgen,complexa_beam,complexa_best_of_n,complexa_fk_steering,complexa_mcts,proteinmpnn_redesign,structure_refilter}"

if [[ "${TREX_ENABLED_FAMILIES}" != *,* && "${TREX_ALLOW_SINGLE_FAMILY:-0}" != "1" ]]; then
  echo "[FATAL] enabled-family list contains no comma; export it in the shell and submit with --export=ALL" >&2
  exit 3
fi

if [[ ! "${TREX_WORKER_GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "[FATAL] TREX_WORKER_GPUS must be a comma-separated GPU index list" >&2
  exit 3
fi
IFS=, read -r -a TREX_WORKER_GPU_ARRAY <<< "${TREX_WORKER_GPUS}"
if [[ "${TREX_REQUIRE_THREE_WORKERS:-1}" == "1" && "${#TREX_WORKER_GPU_ARRAY[@]}" -ne 3 ]]; then
  echo "[FATAL] publication template requires exactly three worker GPUs" >&2
  exit 3
fi


TREX_WORKER_GPU_SEEN=","
for gpu in "${TREX_WORKER_GPU_ARRAY[@]}"; do
  if [[ "${gpu}" == "0" || "${TREX_WORKER_GPU_SEEN}" == *,"${gpu}",* ]]; then
    echo "[FATAL] worker GPU indices must be unique and must not include LLM GPU 0" >&2
    exit 3
  fi
  TREX_WORKER_GPU_SEEN+="${gpu},"
done
if [[ -z "${TREX_CHARGED_GPUS}" ]]; then
  TREX_CHARGED_GPUS="$((1 + ${#TREX_WORKER_GPU_ARRAY[@]}))"
fi


[[ -x "${TREX_CONTROLLER_PYTHON}" ]] || { echo "[FATAL] missing controller Python" >&2; exit 5; }
[[ -r "${TREX_MODEL_MANIFEST}" ]] || { echo "[FATAL] missing model manifest" >&2; exit 5; }

# Resolve both registered and custom targets through the Python registry. This
# is the only target lookup table used by the launcher. A custom target needs
# TREX_TARGET_CONFIG and TREX_TARGET_PDB; a registered target normally needs
# TREX_TARGET_ASSET_ROOT and may optionally override either concrete path.
TARGET_RESOLVE_ARGS=(
  --repo-root "${REPO}" resolve "${TARGET}" --format lines
)
if [[ -n "${TREX_TARGET_ASSET_ROOT:-}" ]]; then
  TARGET_RESOLVE_ARGS+=(--asset-root "${TREX_TARGET_ASSET_ROOT}")
fi
if [[ -n "${TREX_TARGET_CONFIG:-}" ]]; then
  TARGET_RESOLVE_ARGS+=(--target-config "${TREX_TARGET_CONFIG}")
fi
if [[ -n "${TREX_TARGET_PDB:-}" ]]; then
  TARGET_RESOLVE_ARGS+=(--target-pdb "${TREX_TARGET_PDB}")
fi
TARGET_RESOLVED_TEXT="$(
  PYTHONPATH="${REPO}" "${TREX_CONTROLLER_PYTHON}" -m trex.targets \
    "${TARGET_RESOLVE_ARGS[@]}"
)" || { echo "[FATAL] could not resolve TARGET=${TARGET}" >&2; exit 4; }
mapfile -t TARGET_RESOLVED <<< "${TARGET_RESOLVED_TEXT}"
[[ "${#TARGET_RESOLVED[@]}" -eq 4 ]] || {
  echo "[FATAL] target resolver returned ${#TARGET_RESOLVED[@]} fields" >&2
  exit 4
}
TARGET_ID_VALUE="${TARGET_RESOLVED[0]}"
TARGET_CONFIG="${TARGET_RESOLVED[1]}"
TARGET_PDB="${TARGET_RESOLVED[2]}"
TARGET_SOURCE="${TARGET_RESOLVED[3]}"

# Some environment-module hooks read PS1 even in a noninteractive Slurm shell.
export PS1="${PS1:-}"
if [[ -n "${TREX_FOLDSEEK_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
  module load "${TREX_FOLDSEEK_MODULE}"
fi
if [[ -n "${TREX_MMSEQS_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
  module load "${TREX_MMSEQS_MODULE}"
fi
if [[ -n "${TREX_FOLDSEEK_BIN:-}" ]]; then
  [[ -x "${TREX_FOLDSEEK_BIN}" ]] || { echo "[FATAL] invalid TREX_FOLDSEEK_BIN" >&2; exit 5; }
  export PATH="$(dirname "${TREX_FOLDSEEK_BIN}"):${PATH}"
fi
if [[ -n "${TREX_MMSEQS_BIN:-}" ]]; then
  [[ -x "${TREX_MMSEQS_BIN}" ]] || { echo "[FATAL] invalid TREX_MMSEQS_BIN" >&2; exit 5; }
  export PATH="$(dirname "${TREX_MMSEQS_BIN}"):${PATH}"
fi

export TREX_REPO_ROOT="${REPO}"
export TREX_COMPLEXA_REPO="${TREX_COMPLEXA_REPO}"
export TREX_LEGACY_COMPLEXA_REPO="${TREX_LEGACY_COMPLEXA_REPO}"
export TREX_COMPLEXA_PYTHON="${TREX_COMPLEXA_PYTHON}"
export TREX_EXTERNAL_ROOT="${TREX_EXTERNAL_ROOT:-${REPO}/external}"
export TREX_BINDCRAFT_REPO="${TREX_BINDCRAFT_REPO}"
export TREX_BINDCRAFT_ENV="${TREX_BINDCRAFT_ENV}"
export TREX_BOLTZGEN_REPO="${TREX_BOLTZGEN_REPO}"
export TREX_BOLTZGEN_BIN="${TREX_BOLTZGEN_BIN}"
export TREX_BOLTZGEN_CACHE="${TREX_BOLTZGEN_CACHE:-${TREX_EXTERNAL_ROOT:-${REPO}/external}/checkpoints/boltzgen}"
export TREX_WORKER_GPUS="${TREX_WORKER_GPUS}"
export TREX_CHARGED_GPUS="${TREX_CHARGED_GPUS}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
PREFLIGHT_ARGS=(--target "${TARGET}" --target-config "${TARGET_CONFIG}" --target-pdb "${TARGET_PDB}" --enabled-families "${TREX_ENABLED_FAMILIES}" --require-backends --require-model)
if [[ "${TREX_VERIFY_BACKEND_REVISIONS:-1}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--verify-backend-revisions)
fi
PYTHONPATH="${REPO}" "${TREX_CONTROLLER_PYTHON}" -m trex.validation "${PREFLIGHT_ARGS[@]}"

if [[ -n "${TREX_RESUME_ARCHIVE:-}" ]]; then
  ARCHIVE="${TREX_RESUME_ARCHIVE}"
else
  RUN_LABEL="trex_${TARGET}_s${TREX_SEED}_$(date +%Y%m%d_%H%M)"
  ARCHIVE="${TREX_ARCHIVE_BASE}/${RUN_LABEL}/${SLURM_JOB_ID}"
fi
mkdir -p "${ARCHIVE}"
echo "[trex] archive=${ARCHIVE}"
if [[ -n "${TREX_CAMPAIGN_SOURCE_B64:-}" || -n "${TREX_CAMPAIGN_RESOLVED_B64:-}" ]]; then
  : "${TREX_CAMPAIGN_SOURCE_SHA256:?missing submitted campaign SHA256}"
  [[ "${TREX_CAMPAIGN_SOURCE_SHA256}" =~ ^[a-f0-9]{64}$ ]] || {
    echo "[FATAL] invalid submitted campaign SHA256" >&2; exit 4;
  }
  INPUT_COPY="${ARCHIVE}/campaign_input_${TREX_CAMPAIGN_SOURCE_SHA256}.yaml"
  RESOLVED_COPY="${ARCHIVE}/campaign_resolved_${TREX_CAMPAIGN_SOURCE_SHA256}.json"
  printf '%s' "${TREX_CAMPAIGN_SOURCE_B64}" | base64 --decode > "${INPUT_COPY}"
  printf '%s' "${TREX_CAMPAIGN_RESOLVED_B64}" | base64 --decode > "${RESOLVED_COPY}"
  OBSERVED_CAMPAIGN_SHA256="$(sha256sum "${INPUT_COPY}" | awk '{print $1}')"
  [[ "${OBSERVED_CAMPAIGN_SHA256}" == "${TREX_CAMPAIGN_SOURCE_SHA256}" ]] || {
    echo "[FATAL] submitted campaign content failed SHA256 verification" >&2; exit 4;
  }
  unset TREX_CAMPAIGN_SOURCE_B64 TREX_CAMPAIGN_RESOLVED_B64
fi
export TREX_PROVENANCE_LOCK_DIR="${TREX_ARCHIVE_BASE}/.provenance_locks"

MODEL_DIGEST="$(PYTHONPATH="${REPO}" "${TREX_CONTROLLER_PYTHON}" -m trex.provenance model-digest --model-path "${TREX_QWEN_MODEL_PATH}" --manifest "${TREX_MODEL_MANIFEST}")"
export TREX_MODEL_DIGEST="${MODEL_DIGEST}"
PROVENANCE_OUT="${ARCHIVE}/run_provenance.json"
if [[ -e "${PROVENANCE_OUT}" ]]; then
  PROVENANCE_OUT="${ARCHIVE}/run_provenance_resume_${SLURM_JOB_ID}.json"
fi

PYTHONPATH="${REPO}" "${TREX_CONTROLLER_PYTHON}" -m trex.provenance capture \
  --out "${PROVENANCE_OUT}" --source-root "${REPO}" \
  --model-path "${TREX_QWEN_MODEL_PATH}" --model-manifest "${TREX_MODEL_MANIFEST}" \
  --served-model "${TREX_SERVED_MODEL_NAME}" --llm-model "${TREX_LLM_MODEL}" \
  --llm-base-url "http://127.0.0.1:${TREX_VLLM_PORT}/v1" --target "${TARGET}" \
  --target-id "${TARGET_ID_VALUE}" --target-pdb "${TARGET_PDB}" \
  --target-config "${TARGET_CONFIG}" --max-wall-h "${TREX_MAX_WALL_H}" \
  --foldseek-su-tm-score "${TREX_FOLDSEEK_SU_TM_SCORE:-0.60}" \
  --foldseek-collapse-tm-score "${TREX_FOLDSEEK_COLLAPSE_TM_SCORE:-0.60}" \
  --enabled-families "${TREX_ENABLED_FAMILIES}" --worker-gpus "${TREX_WORKER_GPUS}" \
  --charged-gpus "${TREX_CHARGED_GPUS}" --seed "${TREX_SEED}" \
  --critic-enabled "${TREX_ENABLE_CRITIC:-1}" \
  --evidence-skip-enabled "${TREX_ENABLE_EVIDENCE_SKIP:-0}"

echo "[trex] target=${TARGET} target_id=${TARGET_ID_VALUE} source=${TARGET_SOURCE}"
echo "[trex] target_config=${TARGET_CONFIG} target_pdb=${TARGET_PDB}"

VLLM_PID=""
cleanup() {
  local rc=$?
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  exit "${rc}"
}
trap cleanup EXIT TERM INT

(
  export CUDA_VISIBLE_DEVICES=0
  source "${TREX_VLLM_VENV}/bin/activate"
  exec python -m vllm.entrypoints.openai.api_server   \
  --model "${TREX_QWEN_MODEL_PATH}" --served-model-name "${TREX_SERVED_MODEL_NAME}"   \
  --host 127.0.0.1 --port "${TREX_VLLM_PORT}" --max-model-len 65536   \
  --gpu-memory-utilization 0.90   \
  --default-chat-template-kwargs '{"enable_thinking": false}' --trust-remote-code
) >"${ARCHIVE}/vllm.out" 2>"${ARCHIVE}/vllm.err" &
VLLM_PID=$!

READY=0
DEADLINE=$(($(date +%s) + 1500))
while [[ $(date +%s) -lt ${DEADLINE} ]]; do
  kill -0 "${VLLM_PID}" 2>/dev/null || { echo "[FATAL] vLLM exited" >&2; exit 6; }
  if curl -sf -m 3 "http://127.0.0.1:${TREX_VLLM_PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 10
done
[[ "${READY}" == 1 ]] || { echo "[FATAL] vLLM did not become healthy" >&2; exit 6; }

cd "${REPO}"
PYTHONPATH="${REPO}" "${TREX_CONTROLLER_PYTHON}" -m trex.controller \
  --archive-root "${ARCHIVE}" --target-constraint "${TARGET_CONFIG}" \
  --target-pdb "${TARGET_PDB}" --vllm-base-url "http://127.0.0.1:${TREX_VLLM_PORT}/v1" \
  --llm-model "${TREX_LLM_MODEL}" \
  --max-wall-h "${TREX_MAX_WALL_H}" --enabled-families "${TREX_ENABLED_FAMILIES}" \
  --enable-critic "${TREX_ENABLE_CRITIC:-1}" \
  --enable-evidence-skip "${TREX_ENABLE_EVIDENCE_SKIP:-0}" \
  --enable-exemplars "${TREX_ENABLE_EXEMPLARS:-1}" \
  --foldseek-su-tm-score "${TREX_FOLDSEEK_SU_TM_SCORE:-0.60}" \
  --foldseek-collapse-tm-score "${TREX_FOLDSEEK_COLLAPSE_TM_SCORE:-0.60}" \
  --selector-quota-realization "${TREX_SELECTOR_QUOTA_REALIZATION:-fractional_carry}" \
  --selector-mode-window-k "${TREX_SELECTOR_MODE_WINDOW_K:-1}" \
  --selector-adaptive-mode-window-k "${TREX_SELECTOR_ADAPTIVE_MODE_WINDOW_K:-0}" \
  --seed "${TREX_SEED}"
