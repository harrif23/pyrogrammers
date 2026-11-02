"""One Fireworks request function: response metadata, timing, and an explicit retry policy."""

import base64
import io
import time
from pathlib import Path

import openai
from PIL import Image

from .config import BASE_URL, Credentials, InferenceConfig
from .errors import ApiFailure, ImageInputError

# Fireworks documents "total base64-encoded images must be less than 10MB".
MAX_BASE64_BYTES = 10_000_000
_MIME_BY_FORMAT = {"PNG": "image/png", "JPEG": "image/jpeg"}


def encode_image(image):
    """Encode the original file bytes as a data URI.

    rules: never re-encodes or resizes; the benchmark's input is the full-resolution file.
    raises: ImageInputError if the format is unsupported or exceeds the request limit.
    """
    data = Path(image).read_bytes()
    with Image.open(io.BytesIO(data)) as opened:
        mime = _MIME_BY_FORMAT.get(opened.format)
    if mime is None:
        raise ImageInputError(f"unsupported image format in {image}")
    encoded = base64.b64encode(data).decode("ascii")
    if len(encoded) >= MAX_BASE64_BYTES:
        raise ImageInputError(f"{image} exceeds the base64 request limit; not resizing silently")
    return f"data:{mime};base64,{encoded}"


def build_request(model, config: InferenceConfig, image_uri, prompt, system=None):
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": prompt},
        ],
    })
    extra_body = {
        # why: the default silently lowers max_tokens to fit the context window, which
        # would change the frozen budget without showing up in any log.
        "context_length_exceeded_behavior": "error",
    }
    if config.reasoning_effort is not None:
        # In extra_body because Fireworks also allows an integer cap, outside the SDK's types.
        extra_body["reasoning_effort"] = config.reasoning_effort
    request = {
        "model": model,
        "messages": messages,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "extra_body": extra_body,
    }
    if config.response_format != "none":
        request["response_format"] = {"type": config.response_format}
    return request


def failure_from_api_error(error, latency, api_key):
    """The one place SDK errors become ApiFailure, with the key removed from the message."""
    message = str(error).replace(api_key, "[REDACTED]")[:500] if api_key else str(error)[:500]
    if isinstance(error, openai.APITimeoutError):
        return ApiFailure("timeout", message, latency_seconds=latency)
    if isinstance(error, openai.APIConnectionError):
        return ApiFailure("connection", message, latency_seconds=latency)
    status = getattr(error, "status_code", None)
    # These cannot succeed on the next example, so continuing would only waste the run.
    kinds = {401: "authentication", 403: "permission", 404: "deployment_not_found"}
    if status in kinds:
        return ApiFailure(kinds[status], message, status=status, systematic=True,
                          latency_seconds=latency)
    return ApiFailure("api_status", message, status=status, latency_seconds=latency)


class FireworksClient:
    """Sends requests to one existing deployment through the OpenAI-compatible API."""

    def __init__(self, credentials: Credentials, config: InferenceConfig, http_client=None,
                 max_retries=0):
        self._credentials = credentials
        self._config = config
        # why: retries by default would hide failures, double-bill tokens, and distort latency.
        self._client = openai.OpenAI(
            api_key=credentials.api_key,
            base_url=BASE_URL,
            timeout=config.timeout_s,
            max_retries=max_retries,
            http_client=http_client,
        )

    def complete(self, image_uri, prompt, system=None):
        """Make one model call and return a JSON-ready record of it.

        side effects: one network request, more only if max_retries was raised.
        raises: ApiFailure when no usable response came back.
        """
        request = build_request(self._credentials.model, self._config, image_uri, prompt, system)
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
        except openai.APIError as error:
            raise failure_from_api_error(error, time.perf_counter() - started,
                                         self._credentials.api_key) from None
        latency = time.perf_counter() - started

        if not response.choices:
            raise ApiFailure("malformed_response", "response has no choices",
                             latency_seconds=latency)
        choice = response.choices[0]
        return {
            "content": choice.message.content,
            "reasoning_content": getattr(choice.message, "reasoning_content", None),
            "finish_reason": choice.finish_reason,
            "response_id": response.id,
            "returned_model": response.model,
            "latency_seconds": latency,
            "usage": response.usage.model_dump() if response.usage is not None else None,
        }
