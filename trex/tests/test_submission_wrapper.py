"""Submission must stop on invalid setup and preserve exported configuration."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def fixture(tmp_path, *, validation_exit=0, config=None):
    repo = tmp_path / "checkout with spaces"
    (repo / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/submit.sh", repo / "scripts/submit.sh")
    bins = tmp_path / "bin"
    bins.mkdir()
    for name in ("python", "sbatch"):
        script = bins / name
        script.write_text(
            "#!" + sys.executable + "\n"
            "import json, os, pathlib, sys\n"
            "pathlib.Path(os.environ['TEST_LOG'] + '." + name + "').write_text("
            "json.dumps({'argv':sys.argv[1:], 'families':os.environ.get('TREX_ENABLED_FAMILIES'),"
            "'repo':os.environ.get('TREX_REPO_ROOT'),"
            "'workers':os.environ.get('TREX_WORKER_GPUS'),"
            "'charged':os.environ.get('TREX_CHARGED_GPUS'),"
            "'hours':os.environ.get('TREX_MAX_WALL_H'),"
            "'output':os.environ.get('TREX_ARCHIVE_BASE'),"
            "'require_three':os.environ.get('TREX_REQUIRE_THREE_WORKERS')}))\n"
            + ("sys.exit(" + str(validation_exit) + ")\n" if name == "python" else "")
        )
        script.chmod(0o755)
    values = {
        "TARGET": "cd45",
        "TREX_CONTROLLER_PYTHON": str(bins / "python"),
        "TREX_TARGET_CONFIG": str(repo / "target.json"),
        "TREX_TARGET_PDB": str(repo / "target.pdb"),
        "TREX_ARCHIVE_BASE": str(repo / "runs"),
        "TREX_ENABLED_FAMILIES": "family_a,family_b",
        "TREX_NUM_GPUS": "4",
        "TREX_MAX_WALL_H": "48.0",
    }
    values.update(config or {})
    (repo / ".env").write_text(
        "".join(
            k + "=" + shlex.quote(v) + "\n" for k, v in values.items() if v is not None
        )
    )
    env = {
        **os.environ,
        "PATH": str(bins) + ":" + os.environ["PATH"],
        "TEST_LOG": str(tmp_path / "record"),
    }
    env.pop("TREX_ENV_FILE", None)
    for key in ("TREX_TARGET_CONFIG", "TREX_TARGET_PDB"):
        env.pop(key, None)
    return repo, env


def test_failed_validation_never_submits(tmp_path):
    repo, env = fixture(tmp_path, validation_exit=9)
    result = subprocess.run(
        ["bash", str(repo / "scripts/submit.sh")], env=env, capture_output=True
    )
    assert result.returncode == 9
    assert not (tmp_path / "record.sbatch").exists()


def test_missing_controller_python_explains_serving_environment(tmp_path):
    missing = tmp_path / ".venv-serving/bin/python"
    repo, env = fixture(tmp_path, config={"TREX_CONTROLLER_PYTHON": str(missing)})

    result = subprocess.run(
        ["bash", str(repo / "scripts/submit.sh")],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert str(missing) in result.stderr
    assert "CLI-only" in result.stderr
    assert "trex setup" in result.stderr
    assert not (tmp_path / "record.sbatch").exists()


def test_check_only_never_submits(tmp_path):
    repo, env = fixture(tmp_path)
    result = subprocess.run(
        ["bash", str(repo / "scripts/submit.sh"), "--check"],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "record.sbatch").exists()
    args = json.loads((tmp_path / "record.python").read_text())["argv"]
    assert "--verify-backend-revisions" in args and "--require-model" in args


def test_cli_can_supply_validated_environment_without_env_file(tmp_path):
    repo, env = fixture(tmp_path)
    values = {}
    for line in (repo / ".env").read_text().splitlines():
        name, value = line.split("=", 1)
        values[name] = shlex.split(value)[0]
    (repo / ".env").unlink()
    env.update(values)
    env["TREX_SKIP_ENV_FILE"] = "1"

    result = subprocess.run(
        ["bash", str(repo / "scripts/submit.sh"), "--check"],
        env=env,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "record.sbatch").exists()


def test_submit_preserves_commas_paths_and_scheduler_options(tmp_path):
    repo, env = fixture(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts/submit.sh"),
            "--account=example",
            "--partition=gpu",
        ],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "record.sbatch").read_text())
    assert record["families"] == "family_a,family_b"
    assert record["repo"] == str(repo)
    assert record["argv"] == [
        "--export=ALL",
        "--nodes=1",
        "--gres=gpu:h100:4",
        "--time=2940",
        "--account=example",
        "--partition=gpu",
        str(repo / "slurm/T-REX.slurm"),
    ]
    assert (repo / "slurm_logs").is_dir()
    assert record["output"] == str(repo / "runs")
    assert record["workers"] == "1,2,3"
    assert record["charged"] == "4"
    assert record["require_three"] == "0"


def test_unavailable_node_configuration_has_actionable_guidance(tmp_path):
    repo, env = fixture(tmp_path)
    sbatch = Path(env["PATH"].split(":", 1)[0]) / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'sbatch: error: Batch job submission failed: Requested node "
        "configuration is not available' >&2\n"
        "exit 1\n"
    )
    sbatch.chmod(0o755)

    result = subprocess.run(
        ["bash", str(repo / "scripts/submit.sh")],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Requested node configuration is not available" in result.stdout
    assert "--account YOUR_ACCOUNT --partition YOUR_GPU_PARTITION" in result.stderr


def test_template_loads_generated_assets_without_manual_path_copy(tmp_path):
    repo = tmp_path / "checkout with spaces"
    repo.mkdir()
    shutil.copyfile(ROOT / ".env.example", repo / ".env")
    values = {
        "TREX_REPO_ROOT": str(repo),
        "TREX_TARGET_ASSET_ROOT": str(tmp_path / "assets/targets"),
        "TREX_COMPLEXA_REPO": str(tmp_path / "external/Complexa"),
    }
    (repo / ".env.assets").write_text(
        "".join("export " + k + "=" + shlex.quote(v) + "\n" for k, v in values.items())
    )
    result = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            'source "$1"; printf "%s\n" "$TREX_TARGET_ASSET_ROOT" "$TREX_COMPLEXA_PYTHON"',
            "bash",
            str(repo / ".env"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(tmp_path / "assets/targets"),
        str(tmp_path / "external/Complexa/.venv/bin/python"),
    ]


@pytest.mark.parametrize(
    "gpus,hours,workers,minutes",
    [
        ("2", "0.5", "1", "90"),
        ("4", "12", "1,2,3", "780"),
        ("8", "48", "1,2,3,4,5,6,7", "2980"),
        ("2", "0.01", "1", "61"),
    ],
)
def test_resource_settings_drive_the_actual_allocation(
    tmp_path, gpus, hours, workers, minutes
):
    repo, env = fixture(tmp_path)
    with (repo / ".env").open("a") as f:
        f.write(f"TREX_NUM_GPUS={gpus}\nTREX_MAX_WALL_H={hours}\n")
    result = subprocess.run(
        ["bash", str(repo / "scripts/submit.sh")], env=env, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "record.sbatch").read_text())
    assert f"--gres=gpu:h100:{gpus}" in record["argv"]
    assert f"--time={minutes}" in record["argv"]
    assert record["workers"] == workers
    assert record["charged"] == gpus
    assert record["hours"] == hours


@pytest.mark.parametrize(
    "option",
    [
        "--gres=gpu:h100:8",
        "--gpus-per-node=8",
        "-G8",
        "--time=1",
        "-t1",
        "--nodes=2",
        "--export=NONE",
    ],
)
def test_conflicting_scheduler_resource_options_do_not_submit(tmp_path, option):
    repo, env = fixture(tmp_path)
    result = subprocess.run(
        ["bash", str(repo / "scripts/submit.sh"), option],
        env=env,
        capture_output=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "record.sbatch").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("TREX_NUM_GPUS", "1"),
        ("TREX_NUM_GPUS", "0"),
        ("TREX_NUM_GPUS", "2.5"),
        ("TREX_NUM_GPUS", "invalid"),
        ("TREX_MAX_WALL_H", "0"),
        ("TREX_MAX_WALL_H", "-1"),
        ("TREX_MAX_WALL_H", "nan"),
        ("TREX_MAX_WALL_H", "inf"),
        ("TREX_GPU_TYPE", "h100:4"),
    ],
)
def test_invalid_resource_settings_never_submit(tmp_path, key, value):
    repo, env = fixture(tmp_path, config={key: value})
    result = subprocess.run(
        ["bash", str(repo / "scripts/submit.sh")], env=env, capture_output=True
    )
    assert result.returncode == 2
    assert key.encode() in result.stderr
    assert not (tmp_path / "record.sbatch").exists()


def test_registered_target_does_not_require_explicit_cd45_paths(tmp_path):
    repo, env = fixture(
        tmp_path,
        config={"TARGET": "betv1", "TREX_TARGET_CONFIG": None, "TREX_TARGET_PDB": None},
    )
    result = subprocess.run(
        ["bash", str(repo / "scripts/submit.sh"), "--check"],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads((tmp_path / "record.python").read_text())["argv"]
    assert args[args.index("--target") + 1] == "betv1"
    assert "--target-config" not in args and "--target-pdb" not in args


def test_custom_target_paths_and_alternative_config_file(tmp_path):
    repo, env = fixture(tmp_path, config={"TARGET": "custom"})
    selected = tmp_path / "custom config.env"
    (repo / ".env").rename(selected)
    env["TREX_ENV_FILE"] = str(selected)
    result = subprocess.run(
        ["bash", str(repo / "scripts/submit.sh"), "--check"],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "record.python").read_text())
    assert record["repo"] == str(repo)
    args = record["argv"]
    assert args[args.index("--target-config") + 1] == str(repo / "target.json")
    assert args[args.index("--target-pdb") + 1] == str(repo / "target.pdb")


def test_template_loads_checkout_assets_for_external_config_file(tmp_path):
    repo, env = fixture(tmp_path)
    (repo / ".env.assets").write_text(
        "TREX_TARGET_ASSET_ROOT=" + shlex.quote(str(tmp_path / "assets")) + "\n"
        "TREX_COMPLEXA_REPO=" + shlex.quote(str(tmp_path / "Complexa")) + "\n"
    )
    selected = tmp_path / "campaign.env"
    selected.write_text(
        (ROOT / ".env.example").read_text()
        + "\nTREX_CONTROLLER_PYTHON="
        + shlex.quote(str(tmp_path / "bin/python"))
        + "\n"
    )
    env["TREX_ENV_FILE"] = str(selected)
    result = subprocess.run(
        ["bash", str(repo / "scripts/submit.sh"), "--check"],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "record.python").read_text())
    assert record["repo"] == str(repo)
    assert record["workers"] == "1,2,3"
