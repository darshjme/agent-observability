"""Cost tracking module for LLM agent runs.

Tracks token-based costs per model with configurable pricing.
Supports built-in pricing for common models and custom pricing overrides.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional


@dataclasses.dataclass
class ModelPricing:
    """Per-model token pricing configuration.

    Attributes:
        model_id: Unique model identifier string.
        cost_per_1k_in: Cost in USD per 1,000 input tokens.
        cost_per_1k_out: Cost in USD per 1,000 output tokens.
    """

    model_id: str
    cost_per_1k_in: float
    cost_per_1k_out: float

    def compute_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Compute total cost for a given token usage.

        Args:
            tokens_in: Number of input (prompt) tokens consumed.
            tokens_out: Number of output (completion) tokens generated.

        Returns:
            Total cost in USD as a float.
        """
        return (tokens_in / 1000.0) * self.cost_per_1k_in + (
            tokens_out / 1000.0
        ) * self.cost_per_1k_out


# Built-in pricing table (USD per 1k tokens)
_BUILTIN_PRICING: Dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(
        model_id="gpt-4o",
        cost_per_1k_in=0.005,
        cost_per_1k_out=0.015,
    ),
    "claude-sonnet": ModelPricing(
        model_id="claude-sonnet",
        cost_per_1k_in=0.003,
        cost_per_1k_out=0.015,
    ),
    # Aliases for common Claude Sonnet variants
    "claude-3-5-sonnet": ModelPricing(
        model_id="claude-3-5-sonnet",
        cost_per_1k_in=0.003,
        cost_per_1k_out=0.015,
    ),
    "gemini-flash": ModelPricing(
        model_id="gemini-flash",
        cost_per_1k_in=0.000075,
        cost_per_1k_out=0.0003,
    ),
    # Alias
    "gemini-1.5-flash": ModelPricing(
        model_id="gemini-1.5-flash",
        cost_per_1k_in=0.000075,
        cost_per_1k_out=0.0003,
    ),
}


class CostTracker:
    """Tracks and accumulates token-based LLM costs.

    Maintains a per-run cost ledger keyed by model ID.
    Supports custom pricing overrides alongside built-in defaults.

    Attributes:
        _pricing: Mapping from model_id to ModelPricing.
        _ledger: Accumulated cost entries per model.

    Example:
        >>> tracker = CostTracker()
        >>> cost = tracker.compute_cost("gpt-4o", tokens_in=1000, tokens_out=500)
        >>> print(f"${cost:.6f}")
        $0.012500
    """

    def __init__(
        self,
        custom_pricing: Optional[Dict[str, ModelPricing]] = None,
    ) -> None:
        """Initialize CostTracker with optional custom pricing.

        Args:
            custom_pricing: Optional dict mapping model_id strings to
                ModelPricing instances. Overrides built-in defaults.
        """
        self._pricing: Dict[str, ModelPricing] = dict(_BUILTIN_PRICING)
        if custom_pricing:
            self._pricing.update(custom_pricing)
        # ledger: model_id -> {"tokens_in": int, "tokens_out": int, "cost_usd": float}
        self._ledger: Dict[str, Dict[str, float]] = {}

    def register_model(self, pricing: ModelPricing) -> None:
        """Register or override pricing for a model.

        Args:
            pricing: ModelPricing instance to register.
        """
        self._pricing[pricing.model_id] = pricing

    def get_pricing(self, model_id: str) -> Optional[ModelPricing]:
        """Retrieve pricing configuration for a model.

        Args:
            model_id: The model identifier to look up.

        Returns:
            ModelPricing if found, else None.
        """
        return self._pricing.get(model_id)

    def compute_cost(
        self,
        model_id: str,
        tokens_in: int,
        tokens_out: int,
        record: bool = True,
    ) -> float:
        """Compute cost for a token usage event, optionally recording it.

        Args:
            model_id: Model identifier string.
            tokens_in: Number of input tokens.
            tokens_out: Number of output tokens.
            record: If True, accumulate this cost in the internal ledger.

        Returns:
            Cost in USD as a float.

        Raises:
            ValueError: If model_id has no registered pricing.
        """
        pricing = self._pricing.get(model_id)
        if pricing is None:
            raise ValueError(
                f"No pricing registered for model '{model_id}'. "
                f"Available: {list(self._pricing.keys())}"
            )
        cost = pricing.compute_cost(tokens_in, tokens_out)

        if record:
            if model_id not in self._ledger:
                self._ledger[model_id] = {
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                }
            entry = self._ledger[model_id]
            entry["tokens_in"] += tokens_in
            entry["tokens_out"] += tokens_out
            entry["cost_usd"] += cost

        return cost

    def total_cost(self) -> float:
        """Return total accumulated cost across all models.

        Returns:
            Sum of all recorded costs in USD.
        """
        return sum(entry["cost_usd"] for entry in self._ledger.values())

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return a copy of the cost ledger.

        Returns:
            Dict mapping model_id to usage summary dicts with keys:
            tokens_in, tokens_out, cost_usd.
        """
        return {model: dict(entry) for model, entry in self._ledger.items()}

    def reset(self) -> None:
        """Clear all accumulated ledger data."""
        self._ledger.clear()
