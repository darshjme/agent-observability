"""Tests for CostTracker and ModelPricing."""

import pytest
from agent_observability.cost import CostTracker, ModelPricing


class TestModelPricing:
    def test_compute_cost_gpt4o(self):
        pricing = ModelPricing("gpt-4o", cost_per_1k_in=0.005, cost_per_1k_out=0.015)
        # 1000 in + 1000 out = $0.005 + $0.015 = $0.020
        cost = pricing.compute_cost(1000, 1000)
        assert abs(cost - 0.020) < 1e-9

    def test_compute_cost_zero_tokens(self):
        pricing = ModelPricing("test", cost_per_1k_in=0.01, cost_per_1k_out=0.02)
        assert pricing.compute_cost(0, 0) == 0.0

    def test_compute_cost_partial(self):
        pricing = ModelPricing("gpt-4o", cost_per_1k_in=0.005, cost_per_1k_out=0.015)
        # 500 in = $0.0025, 0 out = $0
        cost = pricing.compute_cost(500, 0)
        assert abs(cost - 0.0025) < 1e-9


class TestCostTracker:
    def test_builtin_gpt4o_pricing(self):
        tracker = CostTracker()
        cost = tracker.compute_cost("gpt-4o", tokens_in=1000, tokens_out=1000)
        assert abs(cost - 0.020) < 1e-9

    def test_builtin_claude_sonnet_pricing(self):
        tracker = CostTracker()
        cost = tracker.compute_cost("claude-sonnet", tokens_in=1000, tokens_out=1000)
        # $0.003 + $0.015 = $0.018
        assert abs(cost - 0.018) < 1e-9

    def test_builtin_gemini_flash_pricing(self):
        tracker = CostTracker()
        cost = tracker.compute_cost("gemini-flash", tokens_in=1000, tokens_out=1000)
        # $0.000075 + $0.0003 = $0.000375
        assert abs(cost - 0.000375) < 1e-9

    def test_unknown_model_raises(self):
        tracker = CostTracker()
        with pytest.raises(ValueError, match="No pricing registered"):
            tracker.compute_cost("unknown-model-xyz", 100, 100)

    def test_custom_pricing_override(self):
        custom = {"my-model": ModelPricing("my-model", 0.001, 0.002)}
        tracker = CostTracker(custom_pricing=custom)
        cost = tracker.compute_cost("my-model", 1000, 1000)
        assert abs(cost - 0.003) < 1e-9

    def test_ledger_accumulates(self):
        tracker = CostTracker()
        tracker.compute_cost("gpt-4o", 1000, 1000, record=True)
        tracker.compute_cost("gpt-4o", 1000, 1000, record=True)
        summary = tracker.summary()
        assert "gpt-4o" in summary
        assert summary["gpt-4o"]["tokens_in"] == 2000
        assert summary["gpt-4o"]["tokens_out"] == 2000
        assert abs(summary["gpt-4o"]["cost_usd"] - 0.040) < 1e-9

    def test_no_record_does_not_accumulate(self):
        tracker = CostTracker()
        tracker.compute_cost("gpt-4o", 1000, 1000, record=False)
        assert tracker.total_cost() == 0.0

    def test_total_cost_multi_model(self):
        tracker = CostTracker()
        tracker.compute_cost("gpt-4o", 1000, 0)       # $0.005
        tracker.compute_cost("gemini-flash", 1000, 0)  # $0.000075
        total = tracker.total_cost()
        assert abs(total - (0.005 + 0.000075)) < 1e-9

    def test_reset_clears_ledger(self):
        tracker = CostTracker()
        tracker.compute_cost("gpt-4o", 1000, 1000)
        tracker.reset()
        assert tracker.total_cost() == 0.0
        assert tracker.summary() == {}

    def test_register_model(self):
        tracker = CostTracker()
        tracker.register_model(ModelPricing("new-model", 0.01, 0.02))
        cost = tracker.compute_cost("new-model", 500, 500)
        assert abs(cost - 0.015) < 1e-9

    def test_get_pricing_returns_none_for_unknown(self):
        tracker = CostTracker()
        assert tracker.get_pricing("nonexistent") is None
