"""Job-owned Qwen service and local HTTP client."""
from __future__ import annotations
import json
import http.client
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from .common import write_json


class ServiceFatalError(RuntimeError):
    """The job-owned model engine can no longer serve requests."""


class QwenService:
    """Own one local vLLM process group and expose its OpenAI-compatible URL."""

    def __init__(self, runner: Path, model: Path, model_name: str, gpu_ids: list[str], directory: Path,
                 max_model_len: int, max_num_seqs: int, gpu_memory_utilization: float, timeout: float,
                 prefix_caching: bool = False):
        self.runner = runner
        self.model = model
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.directory = directory
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.timeout = timeout
        self.prefix_caching = prefix_caching
        self.process = None
        self.log = None
        self.url = None
        self.startup_seconds = None

    def __enter__(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/v1"
        command = [
            str(self.runner), "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.model), "--served-model-name", self.model_name,
            "--host", "127.0.0.1", "--port", str(port),
            "--tensor-parallel-size", str(len(self.gpu_ids)), "--dtype", "bfloat16",
            "--max-model-len", str(self.max_model_len), "--max-num-seqs", str(self.max_num_seqs),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--enable-prefix-caching" if self.prefix_caching else "--no-enable-prefix-caching",
            "--mm-processor-cache-gb", "0",
        ]
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": ",".join(self.gpu_ids),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MODELSCOPE_OFFLINE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "PYTHONDONTWRITEBYTECODE": "1",
        })
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = (self.directory / "qwen_server.log").open("w", encoding="utf-8")
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                                            start_new_session=True, env=env)
            write_json(self.directory / "qwen_service.json", {
                "pid": self.process.pid, "url": self.url, "model": str(self.model),
                "model_name": self.model_name, "gpu_ids": self.gpu_ids, "command": command,
            })
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Qwen service exited ({self.process.returncode}); see {self.directory / 'qwen_server.log'}")
                try:
                    with urllib.request.urlopen(self.url + "/models", timeout=2) as response:
                        value = json.load(response)
                    names = {item.get("id") for item in value.get("data", [])}
                    if self.model_name in names:
                        self.startup_seconds = time.perf_counter() - started
                        return self
                except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
                    time.sleep(1)
            raise TimeoutError("Qwen service startup timed out")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            try:
                if os.name == "posix":
                    os.killpg(self.process.pid, signal.SIGTERM)
                elif self.process.poll() is None:
                    self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "posix":
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:
                        self.process.kill()
                except ProcessLookupError:
                    pass
                self.process.wait()
        if self.log is not None:
            self.log.close()

    def __exit__(self, *error):
        self.close()


class LocalQwenClient:
    """Send the bundled review request schema to local vLLM."""

    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.chat = self
        self.completions = self

    @classmethod
    def _from_client(cls, base_url: str, timeout: float):
        value = object.__new__(cls)
        value.base_url = base_url
        value.timeout = timeout
        value.chat = value
        value.completions = value
        return value

    def with_options(self, **options):
        return self._from_client(self.base_url, float(options.get("timeout", self.timeout)))

    def create(self, **request):
        extra = dict(request.pop("extra_body", {}) or {})
        if "enable_thinking" in extra:
            enabled = extra.pop("enable_thinking")
            template = dict(extra.pop("chat_template_kwargs", {}) or {})
            template["enable_thinking"] = enabled
            extra["chat_template_kwargs"] = template
        request.update(extra)
        body = json.dumps(request, ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Qwen HTTP {error.code}: {detail}") from error
        message = payload.get("choices", [{}])[0].get("message", {})
        usage = payload.get("usage")
        usage_value = SimpleNamespace(**usage) if isinstance(usage, dict) else None
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=message.get("content") or ""))],
            usage=usage_value,
        )

    def close(self):
        return None

    def ensure_healthy(self):
        try:
            with urllib.request.urlopen(self.base_url.removesuffix('/v1') + '/health', timeout=5):
                pass
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ServiceFatalError(f'Qwen engine health check failed: {error}') from error

    def warmup(self, model_name: str, output: Path) -> float:
        started = time.perf_counter()
        self.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0,
            max_completion_tokens=8,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        elapsed = time.perf_counter() - started
        write_json(output, {"model": model_name, "seconds": elapsed, "excluded_from_measurement": True})
        return elapsed
