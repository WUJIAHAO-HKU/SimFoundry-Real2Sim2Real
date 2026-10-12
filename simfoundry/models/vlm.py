# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import mimetypes
import os
import random
import re
import time
from io import BytesIO
from types import SimpleNamespace
from urllib.parse import urlsplit
from openai import OpenAI, omit
from PIL import Image as PILImage
import torch
try:
    from diffusers import FluxKontextPipeline
    from diffusers.pipelines.flux.pipeline_flux_kontext import PREFERRED_KONTEXT_RESOLUTIONS
    from diffusers.utils import load_image
except (ImportError, RuntimeError) as e:
    print(f"Error importing diffusers, due to error: \n\n{e}\n")
    FluxKontextPipeline = None
    PREFERRED_KONTEXT_RESOLUTIONS = dict()

from simfoundry.utils.python_utils import assert_valid_key
from simfoundry.models.remote_cache import RemoteModelCache, image_digests


genai = SimpleNamespace(Client=None)
_GOOGLE_GENAI_IMPORTED = False


def _has_patched_genai_client():
    return not _GOOGLE_GENAI_IMPORTED and getattr(genai, "Client", None) is not None


def _get_genai_module():
    global genai, _GOOGLE_GENAI_IMPORTED
    if _has_patched_genai_client():
        return genai
    if not _GOOGLE_GENAI_IMPORTED:
        try:
            from google import genai as google_genai
        except Exception as exc:
            raise RuntimeError(
                "Failed to import google.genai. Install a compatible Google GenAI SDK "
                "or run with TEST_MODE using a cached response."
            ) from exc
        genai = google_genai
        _GOOGLE_GENAI_IMPORTED = True
    return genai


class _FallbackGeminiPart:
    @classmethod
    def from_text(cls, text):
        return SimpleNamespace(text=text, inline_data=None)

    @classmethod
    def from_bytes(cls, data, mime_type):
        return SimpleNamespace(
            text=None,
            inline_data=SimpleNamespace(data=_coerce_bytes(data), mime_type=mime_type),
        )


class _FallbackGeminiContent(SimpleNamespace):
    pass


class _FallbackGeminiConfig(SimpleNamespace):
    pass


class _FallbackGeminiSafetySetting(SimpleNamespace):
    pass


class _FallbackGeminiHttpOptions(SimpleNamespace):
    pass


def _get_genai_types(allow_fallback=False):
    try:
        from google.genai.types import Content, GenerateContentConfig, HttpOptions, Part, SafetySetting
    except Exception as exc:
        if allow_fallback:
            return SimpleNamespace(
                Content=_FallbackGeminiContent,
                GenerateContentConfig=_FallbackGeminiConfig,
                HttpOptions=_FallbackGeminiHttpOptions,
                Part=_FallbackGeminiPart,
                SafetySetting=_FallbackGeminiSafetySetting,
            )
        raise RuntimeError(
            "Failed to import google.genai.types. Install a compatible Google GenAI SDK "
            "or run with TEST_MODE using a cached response."
        ) from exc
    return SimpleNamespace(
        Content=Content,
        GenerateContentConfig=GenerateContentConfig,
        HttpOptions=HttpOptions,
        Part=Part,
        SafetySetting=SafetySetting,
    )


def load_api_keys(path=None):
    """Load KEY=VALUE pairs from api_keys.txt into os.environ (no-op if already set)."""
    from pathlib import Path
    if path is None:
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "api_keys.txt"
            if candidate.exists():
                path = candidate
                break
    if path is None:
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


load_api_keys()


GEMINI_API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def resolve_gemini_auth(project=None, location="global", api_key=None, backend=None):
    """Pick the Gemini auth route. Returns (client_kwargs, route).

    route "api_key" -> Gemini Developer API (generativelanguage.googleapis.com);
    route "vertex"  -> Vertex AI (aiplatform.googleapis.com) via gcloud ADC.

    An API key and ADC are NOT interchangeable: a key identifies a project and
    carries no IAM principal, so Vertex cannot accept one. They are separate
    endpoints with separate quota and billing.

    backend forces a route ("api_key" or "vertex"); "auto" (the default) takes a
    key if one is configured and otherwise falls back to ADC. Override without
    touching code via SIMFOUNDRY_GEMINI_BACKEND.
    """
    backend = (backend or os.environ.get("SIMFOUNDRY_GEMINI_BACKEND") or "auto").lower()
    if backend not in ("auto", "api_key", "vertex"):
        raise ValueError(
            f"Unknown Gemini backend {backend!r}: expected 'auto', 'api_key' or 'vertex'."
        )
    key = api_key or next(
        (os.environ[k] for k in GEMINI_API_KEY_ENVS if os.environ.get(k)), None
    )

    if backend == "vertex" or (backend == "auto" and not key):
        if not project:
            raise ValueError(
                "Gemini needs a Vertex project: pass project=, set GCLOUD_PROJECT, or "
                "supply an API key via api_key=/GEMINI_API_KEY. Vertex also needs ADC "
                "(`gcloud auth application-default login`, or a service-account JSON in "
                "GOOGLE_APPLICATION_CREDENTIALS)."
            )
        return {"vertexai": True, "project": project, "location": location}, "vertex"

    if not key:
        raise ValueError(
            "Gemini backend 'api_key' requires a key: pass api_key= or set "
            + " / ".join(GEMINI_API_KEY_ENVS) + "."
        )
    return {"api_key": key}, "api_key"


GEMINI_TEXT_HARM_CATEGORIES = (
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_HARASSMENT",
)
GEMINI_IMAGE_HARM_CATEGORIES = (
    "HARM_CATEGORY_IMAGE_HATE",
    "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
    "HARM_CATEGORY_IMAGE_HARASSMENT",
    "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
)


def gemini_safety_settings(SafetySetting, route="vertex", threshold="OFF"):
    """Safety settings for the given auth route.

    The Developer API rejects the HARM_CATEGORY_IMAGE_* categories with a 400
    INVALID_ARGUMENT, so they are sent on the vertex route only.
    """
    categories = GEMINI_TEXT_HARM_CATEGORIES
    if route != "api_key":
        categories += GEMINI_IMAGE_HARM_CATEGORIES
    return [SafetySetting(category=c, threshold=threshold) for c in categories]


def _remote_timeout_ms(*env_names, default_ms):
    for env_name in env_names:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            timeout_ms = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer number of milliseconds, got {raw_value!r}") from exc
        return timeout_ms if timeout_ms > 0 else None
    return default_ms


def _get_vertex_imagen_api():
    try:
        import vertexai
        from vertexai.preview.vision_models import Image as VertexImage
        from vertexai.preview.vision_models import ImageGenerationModel, MaskReferenceImage, RawReferenceImage
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Vertex AI image generation dependencies. Install google-cloud-aiplatform "
            "or run with TEST_MODE using a cached response."
        ) from exc
    return SimpleNamespace(
        ImageGenerationModel=ImageGenerationModel,
        MaskReferenceImage=MaskReferenceImage,
        RawReferenceImage=RawReferenceImage,
        VertexImage=VertexImage,
        vertexai=vertexai,
    )


# ==============================================================================
# Remote-call retry policy
# ==============================================================================
# Rate limits (HTTP 429 / RESOURCE_EXHAUSTED) are the common failure mode on
# pay-as-you-go quota. Retrying them immediately makes throttling worse, so they get
# exponential backoff with jitter and a generous attempt budget. Client errors that
# will never succeed on retry (malformed request, auth, safety blocks) fail fast
# instead of burning the budget.

RETRY_BASE_DELAY_S = float(os.environ.get("SIMFOUNDRY_REMOTE_RETRY_BASE_S", 2.0))
RETRY_MAX_DELAY_S = float(os.environ.get("SIMFOUNDRY_REMOTE_RETRY_MAX_S", 60.0))
# Rate limits are worth waiting out, so they get their own (larger) attempt budget.
RATE_LIMIT_MAX_ATTEMPTS = int(os.environ.get("SIMFOUNDRY_REMOTE_RATE_LIMIT_ATTEMPTS", 8))

_RATE_LIMIT_MARKERS = (
    "429",
    "resource_exhausted",
    "resourceexhausted",
    "rate limit",
    "ratelimit",
    "quota exceeded",
    "too many requests",
)
_NON_RETRYABLE_MARKERS = (
    "400",
    "401",
    "403",
    "404",
    "invalid_argument",
    "permission_denied",
    "unauthenticated",
    "not_found",
    "api key not valid",
)


class RemoteCallFailed(RuntimeError):
    """A remote model call exhausted its retries."""


def is_rate_limit_error(exc: BaseException) -> bool:
    """True when an exception looks like provider-side throttling."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def is_non_retryable_error(exc: BaseException) -> bool:
    """True for client errors that will not succeed on retry."""
    if is_rate_limit_error(exc):
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _NON_RETRYABLE_MARKERS)


def retry_delay_from_error(exc: BaseException) -> float | None:
    """Honor a server-provided retry delay when the SDK surfaces one."""
    for attr in ("retry_delay", "retry_after"):
        value = getattr(exc, attr, None)
        seconds = getattr(value, "seconds", value)
        try:
            if seconds is not None and float(seconds) > 0:
                return float(seconds)
        except (TypeError, ValueError):
            pass
    match = re.search(r"retry[_ -]?(?:delay|after)\D{0,12}?(\d+(?:\.\d+)?)", str(exc), re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def backoff_sleep_s(attempt: int, exc: BaseException | None = None) -> float:
    """Exponential backoff with full jitter; `attempt` is 0-based."""
    if exc is not None:
        server_delay = retry_delay_from_error(exc)
        if server_delay is not None:
            return min(server_delay, RETRY_MAX_DELAY_S)
    ceiling = min(RETRY_BASE_DELAY_S * (2 ** attempt), RETRY_MAX_DELAY_S)
    return random.uniform(0.0, ceiling)


def handle_remote_exception(exc, *, attempt, n_retries, provider, model, sleep_fn=time.sleep):
    """Decide whether to retry `exc`, sleeping first when appropriate.

    Returns the effective attempt budget so a rate-limited call can keep going past
    ``n_retries``. Re-raises immediately for errors that retrying cannot fix.
    """
    if is_non_retryable_error(exc):
        raise RemoteCallFailed(
            f"{provider} [{model}] failed with a non-retryable error: {exc}"
        ) from exc

    rate_limited = is_rate_limit_error(exc)
    budget = max(n_retries, RATE_LIMIT_MAX_ATTEMPTS) if rate_limited else n_retries
    if attempt + 1 >= budget:
        return budget

    delay = backoff_sleep_s(attempt, exc)
    kind = "rate limited (429)" if rate_limited else "error"
    print(
        f"{provider} [{model}] {kind} on attempt {attempt + 1}/{budget}; "
        f"retrying in {delay:.1f}s: {exc}",
        flush=True,
    )
    sleep_fn(delay)
    return budget


# ==============================================================================
# Gemini response inspection
# ==============================================================================
# A streamed chunk that carries only an image or thought part has `.text is None`,
# and a blocked or truncated response may consist entirely of such chunks. Joining
# chunk text blindly either crashes (None in join) or silently returns incomplete
# text, so responses are inspected before their text is used.

class GeminiResponseIncomplete(RuntimeError):
    """A Gemini response was blocked or cut off; its joined text cannot be trusted."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


# Finish reasons that mean the model completed normally. Anything else (SAFETY,
# RECITATION, MAX_TOKENS, ...) means the joined text is silently missing content.
_GEMINI_NORMAL_FINISH_REASONS = {"STOP", "FINISH_REASON_UNSPECIFIED"}
_GEMINI_UNSPECIFIED_BLOCK_REASONS = {"BLOCKED_REASON_UNSPECIFIED", "BLOCK_REASON_UNSPECIFIED"}


def gemini_response_problem(result):
    """Why a Gemini response is unusable ("prompt blocked (SAFETY)", "stopped early
    (MAX_TOKENS)"), or None when it finished normally.

    Cache-replayed chunks carry no finish_reason/prompt_feedback and always pass.
    """
    for res in result or []:
        block_reason = getattr(getattr(res, "prompt_feedback", None), "block_reason", None)
        if block_reason is not None:
            name = str(getattr(block_reason, "name", block_reason))
            if name not in _GEMINI_UNSPECIFIED_BLOCK_REASONS:
                return f"prompt blocked ({name})"
        for candidate in getattr(res, "candidates", None) or []:
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason is None:
                continue
            name = str(getattr(finish_reason, "name", finish_reason))
            if name not in _GEMINI_NORMAL_FINISH_REASONS:
                return f"stopped early ({name})"
    return None


def join_gemini_result_text(result, model="gemini"):
    """Join streamed chunk text, skipping chunks that carry no text.

    Raises GeminiResponseIncomplete when the response was blocked or truncated,
    instead of returning silently incomplete text.
    """
    problem = gemini_response_problem(result)
    if problem is not None:
        raise GeminiResponseIncomplete(f"Gemini [{model}] response is incomplete: {problem}")
    return "".join(res.text for res in result if getattr(res, "text", None) is not None)


class VLM_API:
    """
    Class for interfacing with remote VLM APIs, e.g.: ChatGPT, Gemini, etc.
    """
    VERSIONS = None

    @staticmethod
    def encode_image(image_path):
        """
        Encodes image located at @image_path so that it can be included as part of GPT prompts

        Args:
            image_path (str): Absolute path to image to encode

        Returns:
            bytes: Encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


class _CachedInlineData:
    def __init__(self, data, mime_type="image/png"):
        self.data = data
        self.mime_type = mime_type

    def __bool__(self):
        return bool(self.data)


class _CachedGeminiPart:
    def __init__(self, payload):
        self.text = payload.get("text")
        inline_data = payload.get("inline_data")
        self.inline_data = None if inline_data is None else _CachedInlineData(
            data=base64.b64decode(inline_data["data_base64"]),
            mime_type=inline_data.get("mime_type", "image/png"),
        )


class _CachedGeminiContent:
    def __init__(self, parts):
        self.parts = [_CachedGeminiPart(part) for part in parts]


class _CachedGeminiCandidate:
    def __init__(self, parts):
        self.content = _CachedGeminiContent(parts)


class _CachedGeminiChunk:
    def __init__(self, payload):
        self.text = payload.get("text", "")
        self.candidates = [_CachedGeminiCandidate(payload.get("parts", []))]


class _CachedImageDatum:
    def __init__(self, b64_json):
        self.b64_json = b64_json


class _CachedImageResponse:
    def __init__(self, images_base64):
        self.data = [_CachedImageDatum(b64_json=image_base64) for image_base64 in images_base64]


class _CachedImagenResult:
    def __init__(self, image_base64):
        image = PILImage.open(BytesIO(base64.b64decode(image_base64)))
        image.load()
        self._pil_image = image


def _coerce_bytes(data):
    if data is None:
        return b""
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True)
        except Exception:
            return data.encode("utf-8")
    return bytes(data)


def _pil_image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _serialize_gemini_result(result):
    chunks = []
    for res in result or []:
        chunk = {
            "text": getattr(res, "text", "") or "",
            "parts": [],
        }
        candidates = getattr(res, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            for part in getattr(content, "parts", []) or []:
                part_payload = {}
                part_text = getattr(part, "text", None)
                if part_text is not None:
                    part_payload["text"] = part_text
                inline_data = getattr(part, "inline_data", None)
                if inline_data:
                    part_payload["inline_data"] = {
                        "data_base64": base64.b64encode(_coerce_bytes(getattr(inline_data, "data", None))).decode("ascii"),
                        "mime_type": getattr(inline_data, "mime_type", "image/png"),
                    }
                if part_payload:
                    chunk["parts"].append(part_payload)
        chunks.append(chunk)
    return {"chunks": chunks}


def _deserialize_gemini_result(payload):
    return [_CachedGeminiChunk(chunk) for chunk in payload["chunks"]]


def _serialize_gpt_image_result(result):
    return {
        "images_base64": [dat.b64_json for dat in result.data],
    }


def _deserialize_gpt_image_result(payload):
    images = payload["images_base64"]
    if not images or not all(isinstance(value, str) and value for value in images):
        raise RemoteCallFailed("Cached OpenAI image response contains no image data.")
    return _CachedImageResponse(images_base64=images)


def _serialize_imagen_result(result):
    return {
        "images_base64": [_pil_image_to_base64(res._pil_image) for res in result],
    }


def _deserialize_imagen_result(payload):
    return [_CachedImagenResult(image_base64=image_base64) for image_base64 in payload["images_base64"]]


class Gemini(VLM_API):
    """
    Class for interfacing with supported Gemini models
    """
    RESOLUTIONS = {
        "1:1": (1024, 1024),
        "3:4": (864, 1184),
        "4:3": (1184, 864),
        "9:16": (736, 1408),
        "16:9": (1408, 736),
    }
    IMAGE_SHAPES = {res for res in RESOLUTIONS.values()}

    # TODO: we need a central location for all models
    VERSIONS = {
        "gemini-2.0-flash-preview-image-generation": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 8192,
        },
        "gemini-2.5-pro": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-2.5-flash": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-2.5-flash-image-preview": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 32768,
        },
        "gemini-2.5-flash-image": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 32768,
        },
        "gemini-3-pro": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-3-pro-preview": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-3.1-pro-preview": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-3-flash-preview": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-3-pro-image": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 32768,
        },
        "gemini-3.0-flash": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
    }
    def __init__(
        self,
        project=None,
        location="global",
        model="gemini-3-pro-image",
        verbose=False,
        timeout_ms=None,
        api_key=None,
        backend=None,
    ):
        """
        Args:
            project (None or str): Vertex project. Required for the "vertex" route only.
            location (str): Location to use when calling the Gemini client
            model (str): Gemini model to use. Must be one of self.VERSIONS
            timeout_ms (None or int): Request timeout in milliseconds. If None, uses SIMFOUNDRY_GEMINI_TIMEOUT_MS,
                SIMFOUNDRY_REMOTE_TIMEOUT_MS, or the default.
            api_key (None or str): Gemini Developer API key. Falls back to
                GEMINI_API_KEY / GOOGLE_API_KEY.
            backend (None or str): "api_key", "vertex", or "auto" (default). See
                resolve_gemini_auth. Auth is resolved lazily at first call, so
                TEST_MODE and cache replay still need no credentials.
        """
        self.project = project
        self.verbose = verbose
        if self.verbose:
            print("="*100)
            print(f"USING PROJECT: {self.project}")
            print("="*100)
        self.location = location
        self.api_key = api_key
        self.backend = backend
        self.auth_route = None
        self._client_kwargs = None
        assert_valid_key(key=model, valid_keys=self.VERSIONS, name="Gemini model")
        self.model = model
        self.client = None
        self.timeout_ms = (
            timeout_ms
            if timeout_ms is not None
            else _remote_timeout_ms("SIMFOUNDRY_GEMINI_TIMEOUT_MS", "SIMFOUNDRY_REMOTE_TIMEOUT_MS", default_ms=300_000)
        )

    def _resolve_auth(self):
        """Resolve and cache the auth route. Called on the first request, not in
        __init__, so TEST_MODE and cache replay still need no credentials."""
        if self.auth_route is None:
            self._client_kwargs, self.auth_route = resolve_gemini_auth(
                project=self.project,
                location=self.location,
                api_key=self.api_key,
                backend=self.backend,
            )
            if self.verbose:
                print(f"Gemini auth route: {self.auth_route}", flush=True)
        return self.auth_route

    def __call__(
        self,
        prompt,
        image_paths=None,
        temperature=0,
        top_p=0,
        seed=0,
        n_retries=3,
        print_results=False,
    ):
        """
        Calls the Gemini model using the client API.

        Args:
            prompt (str): Text prompt to use
            image_paths (None or str or list of str): If specified, absolute path(s) corresponding to reference image(s)
                to use as part of the overall prompt
            temperature (float): Temperature of the model when querying. Lower values correspond to more deterministic
                outputs
            top_p (float): Determines the cumulative probability of top-p tokens to select from probabilistically.
                E.g.: If top_p=0.7 and tokens a, b, c have probabilities of 0.4, 0.3, 0.2 respectively, only tokens
                a and b will be sampled from
            seed (int): Random seed to use
            n_retries (int): Number of retries to attempt
            print_results (bool): Whether to print results as they're being streamed

        Returns:
            None or list of google.genai.types.GenerateContentResponse: Stream of responses generated from Gemini
        """
        image_paths = [] if image_paths is None else [image_paths] if isinstance(image_paths, (str, os.PathLike)) else list(image_paths)
        cache = RemoteModelCache.from_env()
        cache_request = {
            "prompt": prompt,
            "image_inputs": image_digests(image_paths),
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_output_tokens": self.VERSIONS[self.model]["max_tokens"],
            "response_modalities": self.VERSIONS[self.model]["modalities"],
        }
        cache_key = cache.key_for(provider="gemini", model=self.model, request=cache_request)
        if cache.test_enabled:
            return _deserialize_gemini_result(cache.load_response(provider="gemini", key=cache_key))
        if cache.cache_enabled:
            cached_response = cache.load_response_if_exists(provider="gemini", key=cache_key)
            if cached_response is not None:
                return _deserialize_gemini_result(cached_response)

        genai_types = _get_genai_types(allow_fallback=_has_patched_genai_client())
        parts = [genai_types.Part.from_text(text=prompt)]
        if image_paths:
            msg1_images = []
            for image_path in image_paths:
                msg1_images.append(genai_types.Part.from_bytes(
                    data=self.encode_image(image_path),
                    mime_type="image/png",
                ))
            parts = msg1_images + parts
        contents = [genai_types.Content(
            role="user",
            parts=parts,
        )]

        generate_content_config = genai_types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_output_tokens=self.VERSIONS[self.model]["max_tokens"],
            response_modalities=self.VERSIONS[self.model]["modalities"],
            safety_settings=gemini_safety_settings(
                genai_types.SafetySetting, route=self._resolve_auth()
            ),
        )

        result = None
        if self.client is None:
            client_kwargs = dict(self._client_kwargs)
            if self.timeout_ms is not None:
                client_kwargs["http_options"] = genai_types.HttpOptions(timeout=self.timeout_ms)
            self.client = _get_genai_module().Client(**client_kwargs)
        budget = n_retries
        last_exc = None
        i = 0
        while i < budget:
            if result is not None:
                break
            if self.verbose:
                print(f"Querying Gemini [{self.model}]: attempt {i + 1} of {budget}...", flush=True)
            _result = []
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if print_results:
                        # Image-only / thought chunks have text=None; print nothing for them.
                        print(chunk.text or "", end="")
                    _result.append(chunk)
                result = _result
            except Exception as e:
                last_exc = e
                budget = handle_remote_exception(
                    e, attempt=i, n_retries=budget, provider="Gemini", model=self.model
                )
            i += 1

        if result is None:
            raise RemoteCallFailed(
                f"Gemini [{self.model}] failed after {budget} attempts: {last_exc}"
            ) from last_exc

        print()
        # A blocked/truncated response must never be cached: serialization keeps only
        # text/parts (no finish_reason/prompt_feedback), so a replayed copy would pass
        # gemini_response_problem and silently return the incomplete text that
        # get_result_text exists to reject.
        if cache.cache_enabled and result is not None and gemini_response_problem(result) is None:
            cache.store_response(
                provider="gemini",
                model=self.model,
                key=cache_key,
                request=cache_request,
                response=_serialize_gemini_result(result),
            )
        return result

    def get_result_text(self, result):
        return join_gemini_result_text(result, model=self.model)

    def get_result_images(self, result):
        images = []
        for res in result:
            for part in res.candidates[0].content.parts:
                if part.inline_data:
                    # The image data is in base64 encoded format within part.inline_data.data
                    image_data = part.inline_data.data

                    # You can then process this data, for example, save it as an image file
                    # Decode the base64 data and open it with PIL (Pillow)
                    image = PILImage.open(BytesIO(image_data))
                    images.append(image)
        return images


def openai_error_diagnostic(exc):
    """Keep provider status/code without copying request data into diagnostics."""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    safe_code = str(code) if code and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", str(code)) else "unknown"
    return f"{type(exc).__name__} (HTTP {status or 'unknown'}, code={safe_code})"


def _openai_call(operation, *, model, n_retries):
    # SDK retries are disabled; auth, quota and regional errors fail immediately.
    attempts = max(1, min(int(n_retries), 3))
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if isinstance(exc, RemoteCallFailed):
                raise exc from None
            status = getattr(exc, "status_code", None)
            diagnostic = openai_error_diagnostic(exc)
            if (status is not None and status < 500 and status != 408) or attempt + 1 == attempts:
                failure = RemoteCallFailed(
                    f"OpenAI [{model}] failed: {diagnostic}"
                )
                failure.status_code = status
                raise failure from None
            delay = backoff_sleep_s(attempt)
            print(f"OpenAI [{model}] {diagnostic}; retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)


class OpenAIVision(VLM_API):
    """Text/vision Responses API adapter with provider-separated cache replay."""

    def __init__(self, model="gpt-5.6-luna", api_key=None, base_url=None,
                 reasoning_effort="high", image_detail="high", timeout_ms=180_000,
                 max_output_tokens=16_384, client=None, verbose=False):
        if image_detail not in {"auto", "low", "high"}:
            raise ValueError("OpenAI image_detail must be auto, low, or high")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.image_detail = image_detail
        self.timeout_ms = timeout_ms
        self.max_output_tokens = max_output_tokens
        self.client = client
        self.verbose = verbose

    def __call__(self, prompt, image_paths=None, temperature=0, top_p=0,
                 seed=0, n_retries=2, print_results=False):
        paths = [] if image_paths is None else [image_paths] if isinstance(image_paths, (str, os.PathLike)) else list(image_paths)
        # Reasoning models do not accept the legacy Gemini sampling/seed fields.
        options = {"max_output_tokens": self.max_output_tokens}
        if self.reasoning_effort:
            options["reasoning"] = {"effort": self.reasoning_effort}
        else:
            options["temperature"] = temperature
            if top_p > 0:
                options["top_p"] = top_p
        cache = RemoteModelCache.from_env()
        request = {
            "prompt": prompt, "image_inputs": image_digests(paths),
            "image_detail": self.image_detail, "options": options,
            "base_url": self.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        }
        key = cache.key_for(provider="openai", model=self.model, request=request)
        if cache.test_enabled:
            return SimpleNamespace(**cache.load_response(provider="openai", key=key))
        if cache.cache_enabled:
            cached = cache.load_response_if_exists(provider="openai", key=key)
            if cached is not None:
                return SimpleNamespace(**cached)
        if self.client is None:
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI backend requires OPENAI_API_KEY (or an explicit api_key).")
            self.client = OpenAI(
                api_key=api_key, base_url=self.base_url or os.environ.get("OPENAI_BASE_URL"),
                timeout=self.timeout_ms / 1000 if self.timeout_ms else None, max_retries=0,
            )
        content = [{"type": "input_text", "text": prompt}]
        for path in paths:
            mime = mimetypes.guess_type(os.fspath(path))[0]
            if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                raise ValueError(f"Unsupported OpenAI image format: {mime}")
            content.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{self.encode_image(path)}",
                "detail": self.image_detail,
            })

        def generate():
            response = self.client.responses.create(
                model=self.model, input=[{"role": "user", "content": content}], **options,
            )
            text = self.get_result_text(response)
            return SimpleNamespace(output_text=text, status="completed", model=self.model)

        result = _openai_call(generate, model=self.model, n_retries=n_retries)
        if print_results:
            print(result.output_text, flush=True)
        if cache.cache_enabled:
            cache.store_response(provider="openai", model=self.model, key=key,
                                 request=request, response=vars(result))
        return result

    def get_result_text(self, result):
        status = getattr(result, "status", "completed")
        if status != "completed":
            raise RemoteCallFailed(f"OpenAI [{self.model}] returned an incomplete response")
        text = getattr(result, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            raise RemoteCallFailed(f"OpenAI [{self.model}] returned no text")
        return text


def create_vlm(cfg=None, *, purpose="vision", model=None, project=None,
               location="global", backend=None, **kwargs):
    """Resolve reconstruction provider without passing OpenAI IDs to Gemini."""
    cfg = cfg or {}
    backend = (backend or os.environ.get("SIMFOUNDRY_VLM_BACKEND") or cfg.get("vlm_backend", "gemini")).lower()
    if backend == "gemini":
        return Gemini(project=project, location=location,
                      model=model or ("gemini-3-pro-image" if purpose == "image" else "gemini-3.1-pro-preview"), **kwargs)
    if backend != "openai":
        raise ValueError(f"Unknown VLM backend {backend!r}: expected openai or gemini")
    options = cfg.get("openai", {})
    if purpose == "image":
        return GPT(model=os.environ.get("OPENAI_IMAGE_MODEL") or options.get("image_model", "gpt-image-1"),
                   timeout_ms=options.get("timeout_ms", 180_000), **kwargs)
    if purpose != "vision":
        raise ValueError(f"Unknown VLM purpose: {purpose}")
    return OpenAIVision(
        model=os.environ.get("OPENAI_VISION_MODEL") or options.get("vision_model", "gpt-5.6-luna"),
        reasoning_effort=options.get("reasoning_effort", "high"),
        image_detail=options.get("image_detail", "high"),
        timeout_ms=options.get("timeout_ms", 180_000),
        max_output_tokens=options.get("max_output_tokens", 16_384), **kwargs,
    )


class Imagen3(VLM_API):
    """
    Class for interfacing with supported Imagen3 models
    """
    VERSIONS = {
        "imagen-3.0-capability-001",  # Imagen3
        "imagen-3.0-generate-002",  # Imagen3 -- currently not allowed for our account )):
        "imagen-3.0-generate-001",  # Imagen3
        "imagegeneration@006",      # Imagen2
        "imagegeneration@002",      # Imagen
    }

    RESOLUTIONS = {
        "1:1": (1024, 1024),
        "3:4": (896, 1280),
        "4:3": (1280, 896),
        "9:16": (768, 1408),
        "16:9": (1408, 768),
    }

    def __init__(
        self,
        project,
        location="us-central1",
        model="imagen-3.0-capability-001",
        verbose=False,
    ):
        """
        Args:
            project (str): Name of the project to use when calling the Gemini client
            location (str): Location to use when calling the Gemini client
            model (str): Gemini model to use. Must be one of self.VERSIONS
        """
        self.project = project
        self.verbose = verbose
        if self.verbose:
            print("="*100)
            print(f"vlm.py line 251: USING PROJECT: {self.project}")
            print("="*100)
        self.location = location
        assert_valid_key(key=model, valid_keys=self.VERSIONS, name="Imagen3 model")
        self.model = model
        self.client = None

    def __call__(
        self,
        prompt,
        image_path,
        negative_prompt="",
        mask_image_path=None,
        edit_mode="default",
        aspect_ratio="1:1",
        n_images=4,
        seed=0,
        n_retries=3,
        print_results=False,
    ):
        """
        Calls the Imagen3 model using the client API.

        Args:
            prompt (str): Text prompt to use
            image_path (None or str): If specified, absolute path corresponding to reference image to use as part of
                the overall prompt
            negative_prompt (str): Text prompt to use for negative prompting
            mask_image_path (None or str): If specified, absolute path corresponding to reference image to use as a
                mask conditioning agent, e.g. for directly supervised image editing
            edit_mode (str): Mode for Imagen3 to operate in, e.g.: "default", "inpainting-remove", ...
            aspect_ratio (str): Aspect ratio to use for generated photos
            n_images (int): Number of images to generate. Can be between 1-4
            seed (int): Random seed to use
            n_retries (int): Number of retries to attempt
            print_results (bool): Whether to print results as they're being streamed

        Returns:
            None or list of Results: Imagen3 raw generated results
        """
        assert_valid_key(key=aspect_ratio, valid_keys=self.RESOLUTIONS, name="Aspect ratio")
        cache = RemoteModelCache.from_env()
        cache_request = {
            "prompt": prompt,
            "image_input": image_digests(image_path),
            "negative_prompt": negative_prompt,
            "mask_image_input": image_digests(mask_image_path),
            "edit_mode": edit_mode,
            "aspect_ratio": aspect_ratio,
            "n_images": n_images,
            "seed": seed,
            "safety_filter_level": "block_few",
            "person_generation": "allow_adult",
        }
        cache_key = cache.key_for(provider="imagen3", model=self.model, request=cache_request)
        if cache.test_enabled:
            return _deserialize_imagen_result(cache.load_response(provider="imagen3", key=cache_key))
        if cache.cache_enabled:
            cached_response = cache.load_response_if_exists(provider="imagen3", key=cache_key)
            if cached_response is not None:
                return _deserialize_imagen_result(cached_response)

        imagen_api = _get_vertex_imagen_api()
        raw_ref_image = imagen_api.RawReferenceImage(
            image=imagen_api.VertexImage.load_from_file(location=image_path),
            reference_id=1,
        )
        ref_images = [raw_ref_image]
        if mask_image_path is not None:
            mask_ref_image = imagen_api.MaskReferenceImage(
                reference_id=1,
                image=imagen_api.VertexImage.load_from_file(location=mask_image_path),
                mask_mode="foreground",
            )
            ref_images.append(mask_ref_image)
        result = None
        if self.client is None:
            imagen_api.vertexai.init(project=self.project, location=self.location)
            self.client = imagen_api.ImageGenerationModel.from_pretrained(self.model)
        budget = n_retries
        last_exc = None
        i = 0
        while i < budget:
            if result is not None:
                break
            print(f"Querying Imagen3 [{self.model}]: {i + 1} of {budget}...")
            try:
                result = self.client._generate_images(
                    prompt=prompt,
                    edit_mode=edit_mode,
                    reference_images=ref_images,
                    seed=seed,
                    number_of_images=n_images,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    safety_filter_level="block_few",
                    person_generation="allow_adult",
                )

            except Exception as e:
                last_exc = e
                budget = handle_remote_exception(
                    e, attempt=i, n_retries=budget, provider="Imagen3", model=self.model
                )
            i += 1

        if result is None:
            raise RemoteCallFailed(
                f"Imagen3 [{self.model}] failed after {budget} attempts: {last_exc}"
            ) from last_exc

        if print_results and result is not None:
            for res in result:
                res._pil_image.show()

        if cache.cache_enabled and result is not None:
            cache.store_response(
                provider="imagen3",
                model=self.model,
                key=cache_key,
                request=cache_request,
                response=_serialize_imagen_result(result),
            )
        return result

    def get_result_images(self, result):
        return [res._pil_image for res in result]



class GPT(VLM_API):
    """
    Class for interfacing with supported GPT models
    """
    VERSIONS = {
        "gpt-image-1",
    }

    IMAGE_SHAPES = {
        "portrait": "1024x1536",
        "square": "1024x1024",
        "landscape": "1536x1024",
    }

    def __init__(
        self,
        model="gpt-image-1",
        api_key=None,
        timeout_ms=180_000,
    ):
        """
        Args:
            model (str): GPT model to use. Must be one of self.VERSIONS
            api_key (None or str): OpenAI API key to use. If not set, the OpenAI client falls back
                to the OPENAI_API_KEY environment variable
        """
        if not model.startswith("gpt-image-"):
            raise ValueError("GPT image editing requires a gpt-image-* model")
        self.model = model
        self.api_key = api_key
        self.timeout_ms = timeout_ms

        self.client = None

    def __call__(
        self,
        prompt,
        image_path=None,
        n_images=1,
        n_retries=3,
        image_shape="square",
        print_results=False,
        image_paths=None,
        temperature=0,
        top_p=0,
        seed=0,
    ):
        """
        Calls the Gemini model using the client API.

        Args:
            prompt (str): Text prompt to use
            image_path (None or str): If specified, absolute path corresponding to reference image to use as part of
                the overall prompt
            n_images (int): Number of images to generate
            n_retries (int): Number of retries to attempt
            image_shape (str): Shape of the images to generate. Valid options: {portrait, square, landscape}
            print_results (bool): Whether to print results as they're being streamed

        Returns:
            dict: Output of the model. Valid keys are {"image", "text"} based on the desired model used
        """
        if image_paths is not None:
            paths = [image_paths] if isinstance(image_paths, (str, os.PathLike)) else list(image_paths)
            if len(paths) != 1 or image_path is not None:
                raise ValueError("GPT image editing requires exactly one image")
            image_path = paths[0]
        if self.model.startswith("gpt-image-"):
            assert_valid_key(key=image_shape, valid_keys=self.IMAGE_SHAPES, name="image shape")
            cache = RemoteModelCache.from_env()
            cache_request = {
                "prompt": prompt,
                "image_input": image_digests(image_path),
                "n_images": n_images,
                "image_shape": image_shape,
                "size": self.IMAGE_SHAPES[image_shape],
                "quality": "high",
                "output_format": "png",
                "background": "auto",
            }
            fidelity = "low" if self.model == "gpt-image-1-mini" else "high"
            if self.model.startswith("gpt-image-1"):
                cache_request["input_fidelity"] = fidelity
            cache_key = cache.key_for(provider="gpt-image", model=self.model, request=cache_request)
            if cache.test_enabled:
                return _deserialize_gpt_image_result(cache.load_response(provider="gpt-image", key=cache_key))
            if cache.cache_enabled:
                cached_response = cache.load_response_if_exists(provider="gpt-image", key=cache_key)
                if cached_response is not None:
                    return _deserialize_gpt_image_result(cached_response)

            result = None
            if self.client is None:
                api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI image backend requires OPENAI_API_KEY.")
                self.client = OpenAI(api_key=api_key, max_retries=0,
                                     timeout=self.timeout_ms / 1000 if self.timeout_ms else None)

            def generate():
                with open(image_path, "rb") as image_file:
                    return self.client.images.edit(
                        model=self.model,
                        image=image_file,
                        prompt=prompt,
                        n=n_images,
                        size=self.IMAGE_SHAPES[image_shape],
                        quality="high",
                        output_format="png",
                        background="auto",
                        **({"input_fidelity": fidelity} if self.model.startswith("gpt-image-1") else {}),
                    )

            result = _openai_call(generate, model=self.model, n_retries=n_retries)
            if result is not None:
                result = self._resolve_image_result(result, n_retries=n_retries)

            if print_results and result is not None:
                for image in self.get_result_images(result):
                    image.show()

            if cache.cache_enabled and result is not None:
                cache.store_response(
                    provider="gpt-image",
                    model=self.model,
                    key=cache_key,
                    request=cache_request,
                    response=_serialize_gpt_image_result(result),
                )
            return result

        else:
            raise ValueError(f"Got invalid GPT model for inference: {self.model}")

    def get_result_text(self, result):
        # return "".join(res.text for res in result)
        raise NotImplementedError

    def _resolve_image_result(self, result, n_retries):
        images = []
        for dat in result.data or []:
            encoded = getattr(dat, "b64_json", None)
            if not encoded:
                url = getattr(dat, "url", None)
                parsed = urlsplit(url or "")
                if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                    raise RemoteCallFailed("OpenAI image response has neither base64 data nor a valid HTTPS image URL.")
                # Some compatible providers return URLs; downloads must not receive API credentials.
                payload = _openai_call(
                    lambda: self.client.get(url, cast_to=bytes, options={
                        "headers": {"Authorization": omit, "OpenAI-Organization": omit, "OpenAI-Project": omit},
                        "timeout": self.timeout_ms / 1000 if self.timeout_ms else None,
                    }),
                    model=self.model, n_retries=n_retries,
                )
                encoded = base64.b64encode(payload).decode("ascii")
            images.append(encoded)
        resolved = _CachedImageResponse(images_base64=images)
        self.get_result_images(resolved)
        return resolved

    def get_result_images(self, result):
        try:
            images = []
            for dat in result.data or []:
                image = PILImage.open(BytesIO(base64.b64decode(dat.b64_json, validate=True)))
                image.load()
                images.append(image)
            if not images:
                raise ValueError("empty image response")
            return images
        except (TypeError, ValueError, OSError):
            raise RemoteCallFailed("OpenAI image response contains no valid image data.") from None


class FLUX1(VLM_API):
    """
    Class for interfacing with Flux Kontext model
    """
    VERSIONS = {
        "FLUX.1-Kontext-dev": FluxKontextPipeline,
    }
    MODEL_IDS = {
        "FLUX.1-Kontext-dev": "black-forest-labs/FLUX.1-Kontext-dev",
    }

    IMAGE_SHAPES = {res for res in PREFERRED_KONTEXT_RESOLUTIONS}

    def __init__(
        self,
        model="FLUX.1-Kontext-dev",
        dtype=torch.bfloat16,
        device="cuda",
        enable_cpu_offload=True,
    ):
        """
        Args:
            model (str): FLUX.1 model to use. Must be one of self.VERSIONS
            dtype (torch.dtype): Torch datatype to use
            device (str): Device to use
            enable_cpu_offload (bool): Whether to enable CPU offloading to save on VRAM usage
        """
        assert_valid_key(key=model, valid_keys=self.VERSIONS, name="FLUX.1 model")
        pipeline_cls = self.VERSIONS[model]
        if pipeline_cls is None:
            raise ImportError(
                "The Flux backend requires a diffusers version that provides "
                "FluxKontextPipeline. Install or upgrade the project's diffusers dependency."
            )
        self.pipeline = pipeline_cls.from_pretrained(self.MODEL_IDS[model], torch_dtype=dtype)
        if enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        self.device = device
        # self.pipeline.to(device=self.device)

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        image_path=None,
        guidance_scale=2.5,
        num_inference_steps=28,
        max_sequence_length=512,
        seed=None,
        print_results=False,
    ):
        """
        Calls the FLUX.1 model

        Args:
            prompt (str): Text prompt to use
            negative_prompt (None or str): Optional negative prompt to use
            image_path (None or str): If specified, absolute path corresponding to reference image to use as part of
                the overall prompt
            guidance_scale (float): Guidance scale to use during diffusion
            print_results (bool): Whether to print results as they're being streamed

        Returns:
            PIL.Image: Outputted image from the model
        """
        input_img = load_image(image_path)
        assert_valid_key((input_img.width, input_img.height), valid_keys=self.IMAGE_SHAPES, name="input image shape")
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        image = self.pipeline(
            image=input_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=input_img.width,
            height=input_img.height,
            generator=generator,
            max_sequence_length=max_sequence_length,
        )[0][0]

        if print_results:
            image.show()

        return image
