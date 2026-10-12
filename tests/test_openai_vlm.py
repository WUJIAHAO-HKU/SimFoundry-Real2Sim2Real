# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for name in ("CACHE_MODE", "TEST_MODE", "SIMFOUNDRY_MODEL_CACHE_DIR", "SIMFOUNDRY_VLM_BACKEND"):
        monkeypatch.delenv(name, raising=False)


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="line 1\n{\"answer\": true}", status="completed")


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def write_png(path, color):
    Image.new("RGB", (2, 2), color).save(path)


def test_openai_vision_sends_ordered_data_urls_and_preserves_text(tmp_path):
    from simfoundry.models.vlm import OpenAIVision

    first, second = tmp_path / "a.png", tmp_path / "b.png"
    write_png(first, "red")
    write_png(second, "blue")
    client = FakeClient()
    model = OpenAIVision(model="gpt-5.6-luna", client=client)
    result = model("describe", image_paths=[first, second])

    assert model.get_result_text(result) == "line 1\n{\"answer\": true}"
    content = client.responses.calls[0]["input"][0]["content"]
    assert [part["type"] for part in content] == ["input_text", "input_image", "input_image"]
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert content[2]["image_url"] != content[1]["image_url"]
    assert client.responses.calls[0]["reasoning"] == {"effort": "high"}
    assert content[1]["detail"] == "high"


def test_single_path_and_cache_replay_without_key(monkeypatch, tmp_path):
    from simfoundry.models.vlm import OpenAIVision

    image = tmp_path / "one.png"
    write_png(image, "red")
    client = FakeClient()
    monkeypatch.setenv("CACHE_MODE", "1")
    monkeypatch.setenv("SIMFOUNDRY_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    model = OpenAIVision(client=client)
    first = model("describe", image_paths=str(image))
    assert len(client.responses.calls) == 1
    assert len(list((tmp_path / "cache" / "openai").glob("*.json"))) == 1
    monkeypatch.delenv("CACHE_MODE")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    replay = OpenAIVision()("describe", image_paths=str(image))
    assert replay.output_text == first.output_text


def test_provider_resolver_keeps_gemini_fallback():
    from simfoundry.models.vlm import create_vlm, OpenAIVision, Gemini, GPT

    cfg = {"vlm_backend": "openai", "openai": {"vision_model": "gpt-5.6-luna"}}
    assert isinstance(create_vlm(cfg, model="gemini-2.5-flash"), OpenAIVision)
    assert isinstance(create_vlm(cfg, purpose="image"), GPT)
    cfg["vlm_backend"] = "gemini"
    assert isinstance(create_vlm(cfg, model="gemini-2.5-flash"), Gemini)


def test_openai_missing_key_fails_before_request(monkeypatch):
    from simfoundry.models.vlm import OpenAIVision

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIVision(model="gpt-5.6-luna")("prompt")


def test_openai_cache_key_includes_model_prompt_and_image(tmp_path):
    from simfoundry.models.remote_cache import RemoteModelCache, image_digests

    image = tmp_path / "a.png"
    write_png(image, "red")
    cache = RemoteModelCache(tmp_path / "cache", "off")
    request = {"prompt": "a", "image_inputs": image_digests(image)}
    assert cache.key_for("openai", "gpt-5.6-luna", request) != cache.key_for(
        "openai", "gpt-5.6-luna", {**request, "prompt": "b"}
    )
    assert cache.key_for("openai", "gpt-5.6-luna", request) != cache.key_for(
        "openai", "gpt-5.6-luna", {**request, "image_inputs": []}
    )
    assert cache.key_for("openai", "gpt-5.6-luna", request) != cache.key_for(
        "openai", "gpt-5.6-other", request
    )


@pytest.mark.parametrize("status", [401, 403, 429])
def test_openai_http_errors_are_bounded(monkeypatch, status):
    from simfoundry.models.vlm import OpenAIVision, RemoteCallFailed

    class Error(Exception):
        status_code = status
        body = {"error": {"code": "insufficient_quota" if status == 429 else "access_denied"}}

    class Responses:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise Error("provider error containing SECRET_FOR_TEST")

    client = SimpleNamespace(responses=Responses())
    monkeypatch.setattr("simfoundry.models.vlm.time.sleep", lambda _: None)
    with pytest.raises(RemoteCallFailed) as caught:
        OpenAIVision(client=client)("prompt", n_retries=3)
    assert client.responses.calls == 1
    assert f"HTTP {status}" in str(caught.value)
    assert "SECRET_FOR_TEST" not in str(caught.value)
    assert caught.value.status_code == status


def test_incomplete_response_is_rejected_without_cache(monkeypatch, tmp_path):
    from simfoundry.models.vlm import OpenAIVision, RemoteCallFailed

    client = FakeClient()
    client.responses.create = lambda **_: SimpleNamespace(status="incomplete", output_text="partial")
    monkeypatch.setenv("CACHE_MODE", "1")
    monkeypatch.setenv("SIMFOUNDRY_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(RemoteCallFailed, match="incomplete"):
        OpenAIVision(client=client)("describe")
    assert not list((tmp_path / "cache").rglob("*.json"))


@pytest.mark.parametrize("model_name", ["gpt-image-1", "gpt-image-2"])
def test_gpt_image_compatibility_kwargs_keep_shape_and_cache(monkeypatch, tmp_path, model_name):
    from simfoundry.models.vlm import GPT
    import base64

    image = tmp_path / "source.png"
    write_png(image, "red")
    calls = []

    def edit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(image.read_bytes()).decode())])

    model = GPT(model=model_name)
    model.client = SimpleNamespace(images=SimpleNamespace(edit=edit))
    result = model("edit", image_paths=image, temperature=0, top_p=0, seed=0, image_shape="landscape")
    assert calls[0]["size"] == "1536x1024"
    assert ("input_fidelity" in calls[0]) == (model_name == "gpt-image-1")
    assert model.get_result_images(result)[0].size == (2, 2)


def test_gpt_image_url_is_downloaded_without_auth_and_cached_for_offline_replay(monkeypatch, tmp_path):
    from openai import omit
    from simfoundry.models.vlm import GPT

    image = tmp_path / "source.png"
    write_png(image, "red")
    downloads = []

    def download(url, **kwargs):
        downloads.append((url, kwargs))
        return image.read_bytes()

    model = GPT(model="gpt-image-2")
    model.client = SimpleNamespace(
        images=SimpleNamespace(edit=lambda **_: SimpleNamespace(
            data=[SimpleNamespace(b64_json=None, url="https://cdn.example/image.png?signature=test")]
        )),
        get=download,
    )
    monkeypatch.setenv("CACHE_MODE", "1")
    monkeypatch.setenv("SIMFOUNDRY_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    result = model("edit", image_path=image)
    assert model.get_result_images(result)[0].getpixel((0, 0)) == (255, 0, 0)
    assert downloads[0][1]["cast_to"] is bytes
    assert downloads[0][1]["options"]["headers"]["Authorization"] is omit
    cache_file = next((tmp_path / "cache").rglob("*.json"))
    assert "cdn.example" not in cache_file.read_text()

    monkeypatch.delenv("CACHE_MODE")
    monkeypatch.setenv("TEST_MODE", "1")
    replay = GPT(model="gpt-image-2")("edit", image_path=image)
    assert model.get_result_images(replay)[0].size == (2, 2)
    assert len(downloads) == 1


@pytest.mark.parametrize("datum", [
    SimpleNamespace(b64_json=None, url=None),
    SimpleNamespace(b64_json=None, url="http://cdn.example/image.png"),
    SimpleNamespace(b64_json="not-an-image"),
])
def test_invalid_image_responses_are_not_cached(monkeypatch, tmp_path, datum):
    from simfoundry.models.vlm import GPT, RemoteCallFailed

    image = tmp_path / "source.png"
    write_png(image, "red")
    model = GPT(model="gpt-image-2")
    model.client = SimpleNamespace(images=SimpleNamespace(edit=lambda **_: SimpleNamespace(data=[datum])))
    monkeypatch.setenv("CACHE_MODE", "1")
    monkeypatch.setenv("SIMFOUNDRY_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(RemoteCallFailed):
        model("edit", image_path=image)
    assert not list((tmp_path / "cache").rglob("*.json"))
