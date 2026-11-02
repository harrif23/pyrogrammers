import pytest

from evals.cost import benchmark_share, gpu_estimate, parse_segments, token_priced_estimate
from evals.errors import ConfigError


def tokens(prompt=None, completion=None, cached=None, unknown=0):
    def entry(value):
        return {"sum": value, "known_calls": 1, "unknown_calls": unknown}
    return {"prompt": entry(prompt), "completion": entry(completion), "cached": entry(cached)}


def test_gpu_estimate_is_price_times_gpus_times_hours():
    result = gpu_estimate(3.0, 2, [(4.0, 1)])
    assert result["gpu_hours"] == 8.0
    assert result["estimated_usd"] == 24.0


def test_gpu_estimate_accounts_for_replica_changes():
    result = gpu_estimate(2.0, 1, parse_segments("1:1,0.5:3,2:0"))
    assert result["billable_hours"] == 3.5
    assert result["gpu_hours"] == 2.5
    assert result["estimated_usd"] == 5.0


def test_benchmark_runtime_is_distinct_from_lifetime():
    share = benchmark_share(billable_hours=2.0, active_inference_s=1800)
    assert share == {"active_inference_hours": 0.5, "idle_or_other_hours": 1.5}


@pytest.mark.parametrize("args", [(-1, 1, [(1, 1)]), (1, 0, [(1, 1)]), (1, 1, []),
                                  (1, 1, [(-1, 1)])])
def test_gpu_estimate_rejects_bad_input(args):
    with pytest.raises(ConfigError):
        gpu_estimate(*args)


def test_bad_segment_is_rejected():
    with pytest.raises(ConfigError):
        parse_segments("1h")


def test_token_cost_charges_cached_prompt_tokens_only_at_the_cached_rate():
    result = token_priced_estimate(tokens(1_000_000, 500_000, 400_000), 1.0, 4.0, 0.5)
    assert result["hypothetical_usd"] == pytest.approx(0.6 + 0.2 + 2.0)
    assert result["partial_usage"] is False


def test_token_cost_flags_unknown_usage():
    assert token_priced_estimate(tokens(10, 10, 0, unknown=1), 1, 1, 1)["partial_usage"] is True
