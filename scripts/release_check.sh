#!/usr/bin/env bash
set -euo pipefail

REPO="${TREX_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${TREX_CONTROLLER_PYTHON:-${1:-python}}"

cd "${REPO}"
"${PYTHON}" -m compileall -q trex
"${PYTHON}" scripts/check_architecture.py
"${PYTHON}" -m pytest -q
"${PYTHON}" -m trex.tests.policy_contracts \
  --repo-root "${REPO}" --out "${TMPDIR:-/tmp}/trex_deterministic_smoke.json"
bash -n slurm/T-REX.slurm
bash -n slurm/publication_preflight.slurm
bash -n slurm/multigpu_smoke.slurm
bash -n scripts/run_controller.sh
bash -n scripts/submit.sh
bash -n .env.example
git diff --check

WHEEL_SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/trex-wheel-smoke.XXXXXX")"
cleanup_wheel_smoke() {
  rm -rf -- "${WHEEL_SMOKE_ROOT}"
}
trap cleanup_wheel_smoke EXIT
if ! "${PYTHON}" -m build \
  --outdir "${WHEEL_SMOKE_ROOT}/dist" "${REPO}" \
  > "${WHEEL_SMOKE_ROOT}/build.log" 2>&1; then
  echo "Source-distribution/wheel build failed:" >&2
  tail -n 200 "${WHEEL_SMOKE_ROOT}/build.log" >&2
  exit 1
fi
echo "Built source distribution and wheel for isolated smoke."
"${PYTHON}" scripts/check_distribution.py --dist-dir "${WHEEL_SMOKE_ROOT}/dist"
"${PYTHON}" -m venv "${WHEEL_SMOKE_ROOT}/venv"
# Install T-REX before exposing the parent environment, so pip cannot satisfy
# the request with an editable/source-tree copy of the same distribution.
WHEEL_PATHS=("${WHEEL_SMOKE_ROOT}"/dist/*.whl)
if [[ "${#WHEEL_PATHS[@]}" -ne 1 || ! -f "${WHEEL_PATHS[0]}" ]]; then
  echo "Expected exactly one built wheel, found ${#WHEEL_PATHS[@]}" >&2
  exit 1
fi
(
  cd "${WHEEL_SMOKE_ROOT}"
  PYTHONPATH= "${WHEEL_SMOKE_ROOT}/venv/bin/python" -m pip install \
    --disable-pip-version-check --no-deps --quiet "${WHEEL_PATHS[0]}"
)
# Reuse dependencies already resolved by `.[dev]`; the child wheel remains
# earlier on sys.path and the smoke script asserts its exact import origin.
DEPENDENCY_SITE_PACKAGES="$(
  "${PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))'
)"
WHEEL_SITE_PACKAGES="$(
  "${WHEEL_SMOKE_ROOT}/venv/bin/python" -c \
    'import sysconfig; print(sysconfig.get_path("purelib"))'
)"
printf '%s\n' "${DEPENDENCY_SITE_PACKAGES}" \
  > "${WHEEL_SITE_PACKAGES}/trex-release-dependencies.pth"
(
  cd "${WHEEL_SMOKE_ROOT}"
  PYTHONPATH= "${WHEEL_SMOKE_ROOT}/venv/bin/python" \
    "${REPO}/scripts/installed_wheel_smoke.py" \
    --bin-dir "${WHEEL_SMOKE_ROOT}/venv/bin" \
    --expected-prefix "${WHEEL_SMOKE_ROOT}/venv"
)
cleanup_wheel_smoke
trap - EXIT


echo "T-REX release checks passed."
