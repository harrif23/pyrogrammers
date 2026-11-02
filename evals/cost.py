"""Cost arithmetic. Dedicated deployments bill GPU time; token rates are a separate what-if."""

from .errors import ConfigError


def parse_segments(text):
    """Parse "hours:replicas,hours:replicas" for deployments whose replica count changed."""
    segments = []
    for part in text.split(","):
        try:
            hours, replicas = part.split(":")
            segments.append((float(hours), int(replicas)))
        except ValueError:
            raise ConfigError(f"bad replica segment {part!r}; expected hours:replicas") from None
    return segments


def gpu_estimate(hourly_price, gpus_per_replica, segments):
    """Sum of hourly price x GPUs x replicas x hours over the billable segments."""
    if hourly_price < 0 or gpus_per_replica <= 0:
        raise ConfigError("GPU price must be non-negative and GPUs per replica positive")
    if not segments or any(hours < 0 or replicas < 0 for hours, replicas in segments):
        raise ConfigError("billable segments must be present and non-negative")
    gpu_hours = sum(hours * replicas * gpus_per_replica for hours, replicas in segments)
    return {
        "billable_hours": sum(hours for hours, _ in segments),
        "gpu_hours": gpu_hours,
        "estimated_usd": gpu_hours * hourly_price,
    }


def benchmark_share(billable_hours, active_inference_s):
    """Separate time spent answering requests from the rest of the deployment's lifetime."""
    active_hours = active_inference_s / 3600
    return {
        "active_inference_hours": active_hours,
        "idle_or_other_hours": billable_hours - active_hours,
    }


def token_priced_estimate(tokens, input_rate, output_rate, cached_rate):
    """Hypothetical token-priced cost from explicit USD-per-million rates.

    rules: reasoning tokens are inside completion and are not added again; cached tokens
    are inside prompt and are charged only at the cached rate.
    why: never presented as the dedicated bill; flagged partial when usage is unknown,
    since unknown counts would otherwise read as free tokens.
    """
    if min(input_rate, output_rate, cached_rate) < 0:
        raise ConfigError("token rates must be non-negative")
    prompt = tokens["prompt"]["sum"] or 0
    completion = tokens["completion"]["sum"] or 0
    cached = tokens["cached"]["sum"] or 0
    usd = ((prompt - cached) * input_rate + cached * cached_rate
           + completion * output_rate) / 1_000_000
    return {
        "hypothetical_usd": usd,
        "partial_usage": any(tokens[field]["unknown_calls"]
                             for field in ("prompt", "completion", "cached")),
    }
