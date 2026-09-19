#!/usr/bin/env bash
# Re-validate the v7_3 prompt-form + hybrid-allocation + diagnostic changes against
# the LIVE vLLM-served Qwen model. The deterministic tests prove the plumbing; these
# qwen_* smokes prove the BEHAVIORAL lift (diagnostic citation rate off the old
# ~0.9%, mode-allocation sanity, evidence-form legibility, critic calibration).
#
# Safe to launch anytime: it detects whether the endpoint is up and exits cleanly
# with a message if it is not. Run it the moment the endpoint returns:
#   bash trex/revalidate_qwen.sh
# Optional env: BASE_URL, MODEL, SEEDS, OUT, WAIT_MIN (poll up to N min for endpoint).
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-/usr/licensed/anaconda3/2022.10/bin/python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8500/v1}"
MODEL="${MODEL:-vllm/Qwen/Qwen3.6-27B-FP8}"
SEEDS="${SEEDS:-3}"
WAIT_MIN="${WAIT_MIN:-0}"          # >0 = poll for the endpoint up to N minutes
STAMP="$(date +%Y%m%d_%H%M%S 2>/dev/null || echo manual)"
OUT="${OUT:-${REPO}/trex_outputs/v7_3_revalidation/${STAMP}}"
mkdir -p "${OUT}"

cd "${REPO}" || { echo "[FATAL] repo not found: ${REPO}"; exit 2; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

_endpoint_up() { curl -sf --max-time 5 "${BASE_URL}/models" >/dev/null 2>&1; }

# Wait for the endpoint if asked (poll every 30s up to WAIT_MIN minutes).
if ! _endpoint_up; then
  if [[ "${WAIT_MIN}" -gt 0 ]]; then
    echo "[revalidate] endpoint ${BASE_URL} down; polling up to ${WAIT_MIN} min..."
    deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
    until _endpoint_up; do
      [[ $(date +%s) -ge ${deadline} ]] && break
      sleep 30
    done
  fi
fi
if ! _endpoint_up; then
  echo "[revalidate] vLLM endpoint ${BASE_URL} is NOT reachable — nothing run."
  echo "[revalidate] Start the server, then re-run: bash trex/revalidate_qwen.sh"
  exit 3
fi
echo "[revalidate] endpoint UP at ${BASE_URL}; model=${MODEL}; out=${OUT}"

# Smokes that exercise THIS session's changes. Each: name -> module -> extra args.
SMOKES=(
  "diagnostic_citation|qwen_production_diagnostic_smoke|--seeds ${SEEDS}"   # lever map + per-design blocker + form
  "supervisor_mode_alloc|qwen_supervisor_mode_resource_smoke|--seeds ${SEEDS}"  # hybrid allocation
  "multi_metric_reasoning|qwen_multi_metric_reasoning_smoke|--seeds ${SEEDS}"   # diagnostic reasoning
  "evidence_form|qwen_evidence_richness_smoke|"                                 # TL;DR header / form
  "critic_calibration|qwen_critic_calibration_smoke|"                           # enriched critic payload
)

declare -A RC
for entry in "${SMOKES[@]}"; do
  IFS='|' read -r name mod extra <<< "${entry}"
  log="${OUT}/${name}.log"
  res="${OUT}/${name}.json"
  echo "[revalidate] running ${name} (${mod}) ..."
  # shellcheck disable=SC2086
  "${PY}" -m "trex.${mod}" \
      --model "${MODEL}" --base-url "${BASE_URL}" --out "${res}" ${extra} \
      >"${log}" 2>&1
  RC[$name]=$?
  echo "    exit=${RC[$name]}  log=${log}  result=${res}"
done

echo ""
echo "==================== RE-VALIDATION SUMMARY ===================="
fail=0
for entry in "${SMOKES[@]}"; do
  IFS='|' read -r name _ _ <<< "${entry}"
  if [[ "${RC[$name]:-1}" -eq 0 ]]; then
    status=PASS
  else
    fail=1
    status="FAIL(rc=${RC[$name]:-?})"
  fi
  printf "  %-24s %s\n" "${name}" "${status}"
done
echo "  results dir: ${OUT}"
echo "=============================================================="
echo "[revalidate] Inspect ${name%/*}/*.json for the per-smoke metrics (esp. diagnostic"
echo "             citation rate vs the historical ~0.9%, and mode-allocation sanity)."
exit ${fail}
