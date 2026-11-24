"""Atlas Cloud image client with single-submit generation semantics."""
from __future__ import annotations

import json
import mimetypes
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_API_BASE = "https://api.atlascloud.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-image-2/edit"
TERMINAL_FAILURE_STATES = {"canceled", "cancelled", "failed", "error", "timeout"}


class AtlasCallError(RuntimeError):
    """Atlas Cloud returned an error or an unexpected response."""


def _unwrap(payload: dict) -> dict:
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise AtlasCallError("Atlas Cloud returned an unexpected response shape")
    return data


class AtlasImageClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ATLASCLOUD_API_KEY", "")
        if not self.api_key:
            raise AtlasCallError("ATLASCLOUD_API_KEY is required for the Atlas provider")
        self.api_base = (api_base or os.getenv("ATLASCLOUD_BASE_URL") or DEFAULT_API_BASE).rstrip("/")
        self.model = model or os.getenv("ATLASCLOUD_IMAGE_MODEL") or DEFAULT_MODEL
        self._upload_cache: dict[Path, str] = {}
        self._upload_lock = threading.Lock()

    def _headers(self, *, json_body: bool = True) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "codex-sangpye-skill/0.5.0",
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _request_json(
        self, method: str, path: str, body: dict | None = None, timeout: int = 120
    ) -> dict:
        request = urllib.request.Request(
            f"{self.api_base}{path}",
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers=self._headers(),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise AtlasCallError(f"Atlas Cloud HTTP {error.code}: {detail}") from error
        if not isinstance(payload, dict):
            raise AtlasCallError("Atlas Cloud returned non-object JSON")
        return _unwrap(payload)

    def upload_reference(self, image_path: Path) -> str:
        image_path = image_path.resolve()
        with self._upload_lock:
            cached = self._upload_cache.get(image_path)
            if cached:
                return cached

            mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            boundary = f"----sangpye-atlas-{time.time_ns()}"
            safe_name = image_path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
                f"Content-Type: {mime}\r\n\r\n"
            ).encode() + image_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
            request = urllib.request.Request(
                f"{self.api_base}/model/uploadMedia",
                data=body,
                headers={
                    **self._headers(json_body=False),
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                raise AtlasCallError(f"Atlas upload HTTP {error.code}: {detail}") from error
            data = _unwrap(payload)
            url = data.get("download_url") or data.get("url")
            if not isinstance(url, str) or not url.startswith("https://"):
                raise AtlasCallError("Atlas upload returned no HTTPS download URL")
            self._upload_cache[image_path] = url
            return url

    def generate_with_reference(
        self,
        *,
        reference_image: Path,
        prompt: str,
        size: tuple[int, int],
        quality: str = "high",
        max_polls: int = 60,
        poll_interval: float = 2.0,
    ) -> bytes:
        if max_polls < 1:
            raise ValueError("max_polls must be at least 1")
        reference_url = self.upload_reference(reference_image)

        # Submit exactly once. Retrying this POST can create duplicate billable jobs.
        prediction = self._request_json(
            "POST",
            "/model/generateImage",
            {
                "model": self.model,
                "prompt": prompt,
                "images": [reference_url],
                "size": f"{size[0]}x{size[1]}",
                "quality": "medium" if quality == "standard" else quality,
                "output_format": "png",
            },
        )
        prediction_id = str(prediction.get("id") or "").strip()

        last_error: Exception | None = None
        for attempt in range(max_polls):
            status = str(prediction.get("status") or "").lower()
            if status in {"completed", "succeeded"}:
                outputs = prediction.get("outputs") or []
                if not outputs or not isinstance(outputs[0], str):
                    raise AtlasCallError("Atlas Cloud completed without an output URL")
                if not outputs[0].startswith("https://"):
                    raise AtlasCallError("Atlas Cloud returned a non-HTTPS output URL")
                with urllib.request.urlopen(outputs[0], timeout=120) as response:
                    return response.read()
            if status in TERMINAL_FAILURE_STATES:
                detail = prediction.get("error") or prediction.get("message") or status
                raise AtlasCallError(f"Atlas generation failed: {detail}")
            if not prediction_id:
                raise AtlasCallError("Atlas submission returned no prediction ID")
            if attempt + 1 >= max_polls:
                break

            time.sleep(min(poll_interval * (2 ** min(attempt, 3)), 15.0))
            try:
                prediction = self._request_json(
                    "GET",
                    f"/model/prediction/{urllib.parse.quote(prediction_id, safe='')}",
                    timeout=30,
                )
                last_error = None
            except (AtlasCallError, urllib.error.URLError) as error:
                # Prediction reads are safe to retry within the bounded poll budget.
                last_error = error

        detail = f": {last_error}" if last_error else ""
        raise TimeoutError(f"Atlas prediction did not complete after {max_polls} polls{detail}")
