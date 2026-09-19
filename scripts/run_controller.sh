#!/usr/bin/env bash
set -euo pipefail

REPO="${TREX_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${TARGET:?set TARGET to a config name such as cd45}"
: "${TREX_ARCHIVE_ROOT:?set TREX_ARCHIVE_ROOT to the run archive directory}"
: "${TREX_LLM_BASE_URL:?set TREX_LLM_BASE_URL to an OpenAI-compatible /v1 endpoint}"

PYTHON="${TREX_CONTROLLER_PYTHON:-python}"
SERVED_MODEL="${TREX_SERVED_MODEL_NAME:-Qwen/Qwen3.6-27B-FP8}"
LLM_MODEL="${TREX_LLM_MODEL:-vllm/${SERVED_MODEL}}"
ENABLED_FAMILIES="${TREX_ENABLED_FAMILIES:-bindcraft,boltzgen,complexa_beam,complexa_best_of_n,complexa_fk_steering,complexa_mcts,proteinmpnn_redesign,structure_refilter}"

TARGET_RESOLVE_ARGS=(--repo-root "${REPO}" resolve "${TARGET}" --format lines)
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
  PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" -m trex.targets \
    "${TARGET_RESOLVE_ARGS[@]}"
)" || { echo "could not resolve target: ${TARGET}" >&2; exit 2; }
mapfile -t TARGET_RESOLVED <<< "${TARGET_RESOLVED_TEXT}"
[[ "${#TARGET_RESOLVED[@]}" -eq 4 ]] || {
  echo "target resolver returned ${#TARGET_RESOLVED[@]} fields" >&2
  exit 2
}
TARGET_ID_VALUE="${TARGET_RESOLVED[0]}"
TARGET_CONFIG="${TARGET_RESOLVED[1]}"
TREX_TARGET_PDB="${TARGET_RESOLVED[2]}"
export TREX_TARGET_PDB
mkdir -p "${TREX_ARCHIVE_ROOT}"

export TREX_REPO_ROOT="${REPO}"
export TREX_COMPLEXA_REPO="${TREX_COMPLEXA_REPO:-${REPO}/external/Proteina-Complexa}"
export TREX_LEGACY_COMPLEXA_REPO="${TREX_LEGACY_COMPLEXA_REPO:-${TREX_COMPLEXA_REPO}}"
export TREX_COMPLEXA_PYTHON="${TREX_COMPLEXA_PYTHON:-${TREX_COMPLEXA_REPO}/.venv/bin/python}"
export TREX_EXTERNAL_ROOT="${TREX_EXTERNAL_ROOT:-${REPO}/external}"
export TREX_BINDCRAFT_REPO="${TREX_BINDCRAFT_REPO:-${TREX_EXTERNAL_ROOT}/BindCraft}"
export TREX_BINDCRAFT_ENV="${TREX_BINDCRAFT_ENV:-${TREX_BINDCRAFT_REPO}/.venv}"
export TREX_BOLTZGEN_REPO="${TREX_BOLTZGEN_REPO:-${TREX_EXTERNAL_ROOT}/BoltzGen}"
export TREX_BOLTZGEN_BIN="${TREX_BOLTZGEN_BIN:-${TREX_EXTERNAL_ROOT}/.venvs/boltzgen/bin/boltzgen}"
export TREX_BOLTZGEN_CACHE="${TREX_BOLTZGEN_CACHE:-${TREX_EXTERNAL_ROOT}/checkpoints/boltzgen}"
export TREX_WORKER_GPUS="${TREX_WORKER_GPUS:-0}"
if [[ ! "${TREX_WORKER_GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "TREX_WORKER_GPUS must be a comma-separated GPU index list" >&2
  exit 2
fi
IFS=, read -r -a TREX_WORKER_GPU_ARRAY <<< "${TREX_WORKER_GPUS}"
TREX_WORKER_GPU_SEEN=","
for gpu in "${TREX_WORKER_GPU_ARRAY[@]}"; do
  if [[ "${TREX_WORKER_GPU_SEEN}" == *,"${gpu}",* ]]; then
    echo "TREX_WORKER_GPUS must contain unique GPU indices" >&2
    exit 2
  fi
  TREX_WORKER_GPU_SEEN+="${gpu},"
done
export TREX_CHARGED_GPUS="${TREX_CHARGED_GPUS:-${#TREX_WORKER_GPU_ARRAY[@]}}"
if [[ -n "${TREX_FOLDSEEK_BIN:-}" ]]; then export PATH="$(dirname "${TREX_FOLDSEEK_BIN}"):${PATH}"; fi
if [[ -n "${TREX_MMSEQS_BIN:-}" ]]; then export PATH="$(dirname "${TREX_MMSEQS_BIN}"):${PATH}"; fi
PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" -m trex.validation \
  --target "${TARGET}" --target-config "${TARGET_CONFIG}" --target-pdb "${TREX_TARGET_PDB}" \
  --enabled-families "${ENABLED_FAMILIES}" --require-backends

TREX_MODEL_MANIFEST="${TREX_MODEL_MANIFEST:-${REPO}/config/trex/qwen3_6_27b_fp8_model_manifest.json}"
if [[ -n "${TREX_QWEN_MODEL_PATH:-}" ]]; then
  PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" -m trex.validation \
    --target "${TARGET}" --target-config "${TARGET_CONFIG}" --target-pdb "${TREX_TARGET_PDB}" \
    --enabled-families "${ENABLED_FAMILIES}" --require-model
  PROVENANCE_OUT="${TREX_ARCHIVE_ROOT}/run_provenance.json"
  if [[ -e "${PROVENANCE_OUT}" ]]; then
    PROVENANCE_OUT="${TREX_ARCHIVE_ROOT}/run_provenance_resume_$(date +%Y%m%d_%H%M%S).json"
  fi
  PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" -m trex.provenance capture \
    --out "${PROVENANCE_OUT}" --source-root "${REPO}" \
    --model-path "${TREX_QWEN_MODEL_PATH}" --model-manifest "${TREX_MODEL_MANIFEST}" \
    --served-model "${SERVED_MODEL}" --llm-model "${LLM_MODEL}" \
    --llm-base-url "${TREX_LLM_BASE_URL}" --target "${TARGET}" --target-id "${TARGET_ID_VALUE}" --target-pdb "${TREX_TARGET_PDB}" \
    --target-config "${TARGET_CONFIG}" --max-wall-h "${TREX_MAX_WALL_H:-48.0}" \
    --foldseek-su-tm-score "${TREX_FOLDSEEK_SU_TM_SCORE:-0.60}" \
    --foldseek-collapse-tm-score "${TREX_FOLDSEEK_COLLAPSE_TM_SCORE:-0.60}" \
    --enabled-families "${ENABLED_FAMILIES}" --worker-gpus "${TREX_WORKER_GPUS}" \
    --charged-gpus "${TREX_CHARGED_GPUS}" --seed "${TREX_SEED:-0}" \
    --critic-enabled "${TREX_ENABLE_CRITIC:-1}" \
    --evidence-skip-enabled "${TREX_ENABLE_EVIDENCE_SKIP:-0}"
elif [[ "${TREX_ALLOW_UNVERIFIED_LLM:-0}" != "1" ]]; then
  echo "missing TREX_QWEN_MODEL_PATH; set it for full provenance or explicitly set TREX_ALLOW_UNVERIFIED_LLM=1" >&2
  exit 2
else
  echo "[WARN] running against an unverified remote LLM; this run is not benchmark-reproducible" >&2
fi

exec env PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" \
  -m trex.controller \
  --archive-root "${TREX_ARCHIVE_ROOT}" \
  --target-constraint "${TARGET_CONFIG}" \
  --target-pdb "${TREX_TARGET_PDB}" \
  --vllm-base-url "${TREX_LLM_BASE_URL}" \
  --llm-model "${LLM_MODEL}" \
  --max-wall-h "${TREX_MAX_WALL_H:-48.0}" \
  --enabled-families "${ENABLED_FAMILIES}" \
  --enable-critic "${TREX_ENABLE_CRITIC:-1}" \
  --enable-evidence-skip "${TREX_ENABLE_EVIDENCE_SKIP:-0}" \
  --enable-exemplars "${TREX_ENABLE_EXEMPLARS:-1}" \
  --foldseek-su-tm-score "${TREX_FOLDSEEK_SU_TM_SCORE:-0.60}" \
  --foldseek-collapse-tm-score "${TREX_FOLDSEEK_COLLAPSE_TM_SCORE:-0.60}" \
  --selector-quota-realization "${TREX_SELECTOR_QUOTA_REALIZATION:-fractional_carry}" \
  --selector-mode-window-k "${TREX_SELECTOR_MODE_WINDOW_K:-1}" \
  --selector-adaptive-mode-window-k "${TREX_SELECTOR_ADAPTIVE_MODE_WINDOW_K:-0}" \
  --seed "${TREX_SEED:-0}"
