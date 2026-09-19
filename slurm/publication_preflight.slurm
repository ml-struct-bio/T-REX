#!/usr/bin/env bash
#SBATCH --job-name=trex_preflight
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Read-only publication preflight. Supply site-specific account/partition/QOS
# options to sbatch and export the paths described in .env.example.

set -euo pipefail

REPO="${TREX_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
PYTHON="${TREX_VALIDATION_PYTHON:-${TREX_CONTROLLER_PYTHON:-python}}"
TEST_PYTHON="${TREX_TEST_PYTHON:-${PYTHON}}"
: "${REPO:?set TREX_REPO_ROOT or submit from the T-ReX root}"
: "${TREX_TARGET_ASSET_ROOT:?set TREX_TARGET_ASSET_ROOT}"
: "${TREX_QWEN_MODEL_PATH:?set TREX_QWEN_MODEL_PATH}"

[[ -r "${REPO}/trex/controller.py" ]] || {
  echo "[FATAL] invalid T-ReX source checkout: ${REPO}" >&2
  exit 2
}
[[ -x "${PYTHON}" ]] || {
  echo "[FATAL] validation Python is not executable: ${PYTHON}" >&2
  exit 2
}
[[ -x "${TEST_PYTHON}" ]] || {
  echo "[FATAL] test Python is not executable: ${TEST_PYTHON}" >&2
  exit 2
}

export TREX_REPO_ROOT="${REPO}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${REPO}/slurm_logs"

# Some environment-module hooks read PS1 even in a noninteractive Slurm shell.
export PS1="${PS1:-}"
if [[ -n "${TREX_FOLDSEEK_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
  module load "${TREX_FOLDSEEK_MODULE}"
fi
if [[ -n "${TREX_MMSEQS_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
  module load "${TREX_MMSEQS_MODULE}"
fi
if [[ -n "${TREX_FOLDSEEK_BIN:-}" ]]; then
  export PATH="$(dirname "${TREX_FOLDSEEK_BIN}"):${PATH}"
fi
if [[ -n "${TREX_MMSEQS_BIN:-}" ]]; then
  export PATH="$(dirname "${TREX_MMSEQS_BIN}"):${PATH}"
fi

TARGETS="${TREX_PREFLIGHT_TARGETS:-cd45,betv1,cbago,her2aav,sc2rbd,il7ra,pdl1,vegfa,il17a,tnfalpha}"
IFS=, read -r -a TARGET_ARRAY <<< "${TARGETS}"
[[ "${#TARGET_ARRAY[@]}" -gt 0 ]] || {
  echo "[FATAL] TREX_PREFLIGHT_TARGETS is empty" >&2
  exit 2
}

"${PYTHON}" -m trex.validation \
  --target "${TARGET_ARRAY[0]}" \
  --asset-root "${TREX_TARGET_ASSET_ROOT}" \
  --enabled-families "${TREX_ENABLED_FAMILIES:-bindcraft,boltzgen,complexa_beam,complexa_best_of_n,complexa_fk_steering,complexa_mcts,proteinmpnn_redesign,structure_refilter}" \
  --require-backends \
  --require-model \
  --verify-backend-revisions \
  --verify-checkpoint-content

for target in "${TARGET_ARRAY[@]:1}"; do
  "${PYTHON}" -m trex.validation \
    --target "${target}" \
    --asset-root "${TREX_TARGET_ASSET_ROOT}"
done

"${TEST_PYTHON}" -m compileall -q "${REPO}/trex"
"${TEST_PYTHON}" -m pytest -q "${REPO}/trex/tests"
"${TEST_PYTHON}" -m trex.tests.policy_contracts \
  --repo-root "${REPO}" \
  --out "${TMPDIR:-/tmp}/trex_deterministic_smoke.json"
"${TEST_PYTHON}" "${REPO}/benchmarks/paper_7target_144h/verify.py"

bash -n "${REPO}/slurm/trex_per_target_node.slurm"
bash -n "${REPO}/slurm/publication_preflight.slurm"
bash -n "${REPO}/scripts/run_controller.sh"
git -C "${REPO}" diff --check

echo "T-ReX publication preflight passed."
