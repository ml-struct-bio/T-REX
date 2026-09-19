"""Helpers for local OpenAI-compatible LLM servers.

The LLM clients can already talk to ``vllm/<model>`` through the
OpenAI-compatible API. This module owns the host-side concern of checking
whether the vLLM server is already available and, when requested, launching
it pinned to a dedicated GPU.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence
from urllib.error import URLError
from urllib.request import urlopen


def is_vllm_model(model: Optional[str]) -> bool:
    return bool(model and model.lower().startswith("vllm/"))


def vllm_actual_model(model: str) -> str:
    return model.split("/", 1)[1]


def _models_endpoint(host: str, port: int) -> str:
    return "http://%s:%d/v1/models" % (host, int(port))


def _served_model_ids(payload: object) -> Sequence[str]:
    if not isinstance(payload, dict):
        return ()
    data = payload.get("data", ())
    if not isinstance(data, list):
        return ()
    ids = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return tuple(ids)


def vllm_server_models(host: str = "127.0.0.1", port: int = 8000) -> Sequence[str]:
    """Return model ids from a local OpenAI-compatible `/v1/models` endpoint."""
    try:
        with urlopen(_models_endpoint(host, port), timeout=2) as response:
            if not 200 <= int(response.status) < 300:
                return ()
            return _served_model_ids(json.loads(response.read().decode("utf-8")))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return ()


def _model_matches(requested_model: str, served_model: str) -> bool:
    requested = requested_model.rstrip("/")
    served = served_model.rstrip("/")
    return (
        served == requested
        or served.endswith("/" + requested)
        or requested.endswith("/" + served)
    )


def _has_requested_model(served_models: Sequence[str], requested_model: str) -> bool:
    return any(_model_matches(requested_model, served) for served in served_models)


def vllm_server_ready(
    host: str = "127.0.0.1",
    port: int = 8000,
    expected_model: Optional[str] = None,
) -> bool:
    if expected_model is None:
        try:
            with urlopen(_models_endpoint(host, port), timeout=2) as response:
                return 200 <= int(response.status) < 300
        except (OSError, URLError, ValueError):
            return False
    served_models = vllm_server_models(host=host, port=port)
    if not served_models:
        return False
    return _has_requested_model(served_models, expected_model)


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.25)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()


def ensure_vllm_server(
    model: Optional[str],
    *,
    llm_gpu: Optional[int] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_dir: Optional[Path] = None,
    wait_seconds: float = 120.0,
) -> bool:
    """Ensure a local vLLM OpenAI-compatible server exists.

    Returns True when this call started a new server, False when an
    existing server was already reachable or the model is not a vLLM model.
    If ``model`` is ``vllm/...`` and no server is reachable, ``llm_gpu`` is
    required so the launcher can pin vLLM away from program-evaluation GPUs.
    """
    if not is_vllm_model(model):
        return False
    actual_model = vllm_actual_model(str(model))
    served_models = vllm_server_models(host=host, port=port)
    if served_models:
        if _has_requested_model(served_models, actual_model):
            return False
        raise RuntimeError(
            "A local vLLM/OpenAI-compatible server is already reachable at %s, "
            "but it is serving %s instead of requested model %r."
            % (_models_endpoint(host, port), list(served_models), actual_model)
        )
    if vllm_server_ready(host=host, port=port):
        raise RuntimeError(
            "A local vLLM/OpenAI-compatible server is reachable at %s but its "
            "/v1/models response did not include model ids; refusing to attach "
            "to an unverified server for requested model %r."
            % (_models_endpoint(host, port), actual_model)
        )
    if llm_gpu is None:
        raise RuntimeError(
            "vLLM model %r requested but no server is reachable at %s and "
            "no llm_gpu was provided to launch one." % (model, _models_endpoint(host, port))
        )

    return launch_vllm_server(
        actual_model,
        llm_gpu=llm_gpu,
        host=host,
        port=port,
        log_dir=log_dir,
        wait_seconds=wait_seconds,
    )


def launch_vllm_server(
    actual_model: str,
    *,
    llm_gpu: int,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_dir: Optional[Path] = None,
    wait_seconds: float = 120.0,
) -> bool:
    """Launch vLLM pinned to one GPU and wait for the requested model."""
    if vllm_server_ready(host=host, port=port):
        served_models = vllm_server_models(host=host, port=port)
        if _has_requested_model(served_models, actual_model):
            return False
        raise RuntimeError(
            "Port %s already has an OpenAI-compatible server, but it is not "
            "serving requested model %r. Served models: %s"
            % (port, actual_model, list(served_models))
        )
    if llm_gpu is None:
        raise RuntimeError(
            "Cannot launch vLLM model %r without a dedicated llm_gpu."
            % actual_model
        )

    log_root = Path(log_dir or Path("trex_outputs") / "llm_servers")
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / ("vllm_%s.log" % str(port))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(llm_gpu)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        actual_model,
        "--host",
        host,
        "--port",
        str(port),
    ]
    with open(str(log_path), "a") as log_handle:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    metadata_path = log_root / ("vllm_%s.json" % str(port))
    metadata_path.write_text(
        json.dumps(
            {
                "model": actual_model,
                "host": host,
                "port": port,
                "llm_gpu": llm_gpu,
                "pid": process.pid,
                "command": command,
                "log_path": str(log_path),
            },
            indent=2,
            sort_keys=True,
        )
    )

    deadline = time.time() + float(wait_seconds)
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "vLLM process exited before becoming ready at %s. See %s"
                % (_models_endpoint(host, port), log_path)
            )
        if vllm_server_ready(host=host, port=port, expected_model=actual_model):
            return True
        time.sleep(2.0)

    _terminate_process_group(process)
    raise RuntimeError(
        "Timed out waiting for vLLM server at %s to serve model %r. "
        "Terminated launched process group pid=%s. See %s"
        % (_models_endpoint(host, port), actual_model, process.pid, log_path)
    )
