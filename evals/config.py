"""Local configuration: credentials from the environment, explicit inference settings."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigError

BASE_URL = "https://api.fireworks.ai/inference/v1"
EVALS_DIR = Path(__file__).resolve().parent
REASONING_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")
RESPONSE_FORMATS = ("none", "json_object")


@dataclass(frozen=True)
class Credentials:
    api_key: str = field(repr=False)
    model: str


@dataclass(frozen=True)
class InferenceConfig:
    # None omits the parameter, leaving the model's default reasoning behavior.
    reasoning_effort: str | int | None = "xhigh"
    max_tokens: int = 4096
    temperature: float = 0.0
    response_format: str = "none"
    timeout_s: float = 180.0

    def scored_fields(self):
        # why: timeout cannot change an answer, and raising it is the normal fix for
        # timeouts mid-run, so it must not block resume.
        fields = asdict(self)
        del fields["timeout_s"]
        return fields


def parse_reasoning_effort(value):
    text = str(value).strip().lower()
    if text == "omit":
        return None
    if text in REASONING_LEVELS:
        return text
    if text.isdigit() and int(text) > 0:
        return int(text)
    raise ConfigError(
        f"reasoning effort must be one of {', '.join(REASONING_LEVELS)}, omit, "
        "or a positive integer token cap"
    )


def build_inference_config(reasoning_effort="xhigh", max_tokens=4096, temperature=0.0,
                           response_format="none", timeout_s=180.0):
    if max_tokens <= 0:
        raise ConfigError("max_tokens must be positive")
    if timeout_s <= 0:
        raise ConfigError("timeout must be positive")
    if response_format not in RESPONSE_FORMATS:
        raise ConfigError(f"response_format must be one of {RESPONSE_FORMATS}")
    return InferenceConfig(
        reasoning_effort=parse_reasoning_effort(reasoning_effort),
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
        timeout_s=timeout_s,
    )


def load_env_file():
    # The repo's convention: .env in the working directory, and real variables win over it.
    load_dotenv(Path.cwd() / ".env", override=False)


def model_id_problem(model):
    """Why a model value is not deployment-qualified, or None if it looks right."""
    # why: a bare model id would route to serverless, billing and measuring the wrong
    # thing while appearing to work.
    # Documented forms: accounts/<a>/deployments/<d> and accounts/<a>/models/<m>#<d>.
    if model.startswith("accounts/") and ("/deployments/" in model or "#" in model):
        return None
    return "must be the deployment-qualified value from your deployment's API example"


def load_credentials(environ=None):
    """Read credentials, naming missing fields without ever echoing values.

    raises: ConfigError listing every field to fill.
    """
    environ = os.environ if environ is None else environ
    api_key = environ.get("FIREWORKS_API_KEY", "").strip()
    model = environ.get("FLGROUND_MODEL", "").strip()
    problems = []
    if not api_key:
        problems.append("FIREWORKS_API_KEY is empty")
    if not model:
        problems.append("FLGROUND_MODEL is empty")
    elif model_id_problem(model):
        problems.append(f"FLGROUND_MODEL {model_id_problem(model)}")
    if problems:
        raise ConfigError("; ".join(problems) + " (fill these in .env)")
    return Credentials(api_key=api_key, model=model)


def run_fingerprint(mode, model, inference, prompt_sha256, dataset_revision, annotation_sha256,
                    split, example_ids):
    payload = {
        "mode": mode,
        "base_url": BASE_URL,
        "model": model,
        "inference": inference.scored_fields(),
        "prompt_sha256": prompt_sha256,
        "dataset_revision": dataset_revision,
        "annotation_sha256": annotation_sha256,
        "split": split,
        "example_ids": list(example_ids),
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
