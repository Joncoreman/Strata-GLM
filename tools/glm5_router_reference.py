"""CPU scalar reference for GLM5-Next routing; no model payloads or device dependencies.

Source: llama.cpp b11461 src/llama-graph.cpp:2091-2118, 2160-2203.
This uses Python double arithmetic as a numerical oracle, not a bitwise FP32 kernel model.
Equal selection scores use ascending expert ID here. The reference's CPU std::sort only
compares values and does not specify tie order; GPU tie parity remains to be decided.
"""
from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

# Exact constant in llama-graph.cpp:2190 (2**-14), not the rounded 6.1e-5 shorthand.
WEIGHT_SUM_FLOOR = 6.103515625e-5


@dataclass(frozen=True)
class RouterResult:
    experts: tuple[int, ...]
    weights: tuple[float, ...]


def sigmoid(x: float) -> float:
    """Stable for large positive/negative finite logits, and for +/- infinity."""
    if math.isnan(x):
        raise ValueError("router logits must not contain NaN")
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def route(logits: Sequence[float], selection_bias: Sequence[float], *, top_k: int, scale: float) -> RouterResult:
    """sigmoid -> biased top-k selection -> unbiased weights -> floor-normalize -> scale.

    top_k and scale must come from the model metadata; no architecture dimensions are assumed.
    Group selection is intentionally outside this reference (the measured model has one group).
    """
    if len(logits) != len(selection_bias) or not isinstance(top_k, int) or not 0 < top_k <= len(logits):
        raise ValueError("router needs equal logits/bias lengths and 0 < top_k <= expert count")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("router scale must be finite and positive")
    if not all(math.isfinite(float(b)) for b in selection_bias):
        raise ValueError("router selection bias must be finite")
    probabilities = [sigmoid(float(x)) for x in logits]
    scores = [p + float(b) for p, b in zip(probabilities, selection_bias)]
    selected = tuple(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k])
    unbiased = [probabilities[i] for i in selected]
    denominator = max(sum(unbiased), WEIGHT_SUM_FLOOR)
    return RouterResult(selected, tuple((p / denominator) * scale for p in unbiased))


def route_from_metadata(logits: Sequence[float], selection_bias: Sequence[float], metadata: Mapping) -> RouterResult:
    """Bind the numerical reference to a header read by tools/gguf_reader.py."""
    if metadata.get("general.architecture") != "glm5-next":
        raise ValueError("glm5-next metadata required")
    prefix = "glm5-next."
    if metadata[prefix + "expert_gating_func"] != 2 or not metadata[prefix + "expert_weights_norm"]:
        raise ValueError("reference requires sigmoid routing with weight normalization")
    if metadata[prefix + "expert_group_count"] != 1 or metadata[prefix + "expert_group_used_count"] != 1:
        raise ValueError("group selection is not implemented by this reference")
    if len(logits) != metadata[prefix + "expert_count"]:
        raise ValueError("logit count does not match expert_count")
    return route(logits, selection_bias, top_k=metadata[prefix + "expert_used_count"],
                 scale=metadata[prefix + "expert_weights_scale"])
