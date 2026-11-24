"""Unit tests for the optional Atlas Cloud image provider."""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from sangpye_skill.atlas_client import AtlasCallError, AtlasImageClient


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def json_response(payload: dict) -> FakeResponse:
    return FakeResponse(json.dumps(payload).encode())


def test_uploads_once_submits_once_then_polls_and_downloads(tmp_path, monkeypatch):
    reference = tmp_path / "product.png"
    reference.write_bytes(b"png-reference")
    calls: list[tuple[str, str]] = []
    generation_payloads: list[dict] = []
    responses = iter([
        json_response({"data": {"download_url": "https://cdn.example/ref.png"}}),
        json_response({"data": {"id": "pred-1", "status": "created"}}),
        json_response({"data": {"id": "pred-1", "status": "processing"}}),
        json_response({"data": {"id": "pred-1", "status": "completed", "outputs": ["https://cdn.example/out.png"]}}),
        FakeResponse(b"generated-png"),
    ])

    def fake_urlopen(request, timeout=None):
        if isinstance(request, urllib.request.Request):
            calls.append((request.get_method(), request.full_url))
            if request.full_url.endswith("/model/generateImage"):
                generation_payloads.append(json.loads(request.data.decode()))
        else:
            calls.append(("GET", request))
        return next(responses)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sangpye_skill.atlas_client.time.sleep", lambda _delay: None)
    client = AtlasImageClient(api_key="test-key")

    result = client.generate_with_reference(
        reference_image=reference,
        prompt="Korean product detail page",
        size=(1088, 1600),
        quality="standard",
        max_polls=3,
        poll_interval=0,
    )

    assert result == b"generated-png"
    assert [method for method, _url in calls] == ["POST", "POST", "GET", "GET", "GET"]
    assert sum(url.endswith("/model/generateImage") for _method, url in calls) == 1
    assert generation_payloads[0]["quality"] == "medium"
    assert generation_payloads[0]["size"] == "1088x1600"


def test_upload_accepts_url_field_when_download_url_is_absent(tmp_path, monkeypatch):
    reference = tmp_path / "product.png"
    reference.write_bytes(b"png-reference")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout=None: json_response({"data": {"url": "https://cdn.example/ref.png"}}),
    )

    assert AtlasImageClient(api_key="test-key").upload_reference(reference) == "https://cdn.example/ref.png"


def test_upload_without_https_url_fails(tmp_path, monkeypatch):
    reference = tmp_path / "product.png"
    reference.write_bytes(b"png-reference")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout=None: json_response({"data": {"url": "http://cdn.example/ref.png"}}),
    )

    with pytest.raises(AtlasCallError, match="HTTPS"):
        AtlasImageClient(api_key="test-key").upload_reference(reference)


def test_generation_post_failure_is_not_retried(tmp_path, monkeypatch):
    reference = tmp_path / "product.png"
    reference.write_bytes(b"png-reference")
    calls = 0

    def fake_urlopen(request, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return json_response({"data": {"download_url": "https://cdn.example/ref.png"}})
        raise urllib.error.HTTPError(
            request.full_url, 503, "unavailable", {}, io.BytesIO(b'{"message":"busy"}')
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = AtlasImageClient(api_key="test-key")

    with pytest.raises(AtlasCallError, match="HTTP 503"):
        client.generate_with_reference(
            reference_image=reference,
            prompt="Korean product detail page",
            size=(1088, 1600),
        )

    assert calls == 2


def test_upload_cache_reuses_reference_url(tmp_path, monkeypatch):
    reference = tmp_path / "product.jpg"
    reference.write_bytes(b"jpeg-reference")
    calls = 0

    def fake_urlopen(_request, timeout=None):
        nonlocal calls
        calls += 1
        return json_response({"data": {"download_url": "https://cdn.example/ref.jpg"}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = AtlasImageClient(api_key="test-key")

    assert client.upload_reference(reference) == client.upload_reference(reference)
    assert calls == 1
