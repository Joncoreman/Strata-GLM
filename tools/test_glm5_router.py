"""Independent analytic cases for the GLM router's selection and weight paths."""
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm5_router_reference import route, route_from_metadata, sigmoid, WEIGHT_SUM_FLOOR


def logits(probabilities):
    return [math.log(p / (1 - p)) for p in probabilities]


class RouterTests(unittest.TestCase):
    def test_sigmoid_not_softmax(self):
        result = route(logits([0.2, 0.4, 0.8]), [0, 0, 0], top_k=2, scale=2.5)
        self.assertEqual(result.experts, (2, 1))
        self.assertAlmostEqual(result.weights[0], 2.5 * 2 / 3)
        self.assertAlmostEqual(result.weights[1], 2.5 / 3)

    def test_bias_selects_low_probability_expert_without_inflating_weight(self):
        result = route(logits([0.1, 0.2, 0.8, 0.9]), [2, 1, 0, 0], top_k=2, scale=2.5)
        self.assertEqual(result.experts, (0, 1))
        self.assertAlmostEqual(result.weights[0], 2.5 / 3)
        self.assertAlmostEqual(result.weights[1], 2.5 * 2 / 3)

    def test_bias_changes_only_order_when_selected_set_is_unchanged(self):
        x = logits([0.1, 0.2, 0.7, 0.9])
        a = route(x, [0, 0, 0, 0], top_k=2, scale=2.5)
        b = route(x, [0, 0, 0.4, 0], top_k=2, scale=2.5)
        self.assertNotEqual(a.experts, b.experts)
        for expert, w in zip(a.experts, a.weights):
            self.assertAlmostEqual(dict(zip(b.experts, b.weights))[expert], w)

    def test_real_width_and_top_eight(self):
        # Real header dimensions; the synthetic probabilities have no selection ties.
        p = [(i + 1) / 300 for i in range(288)]
        result = route(logits(p), [0] * 288, top_k=8, scale=2.5)
        self.assertEqual(result.experts, tuple(range(287, 279, -1)))
        self.assertAlmostEqual(sum(result.weights), 2.5)
        denominator = sum(p[280:])
        for i, w in zip(result.experts, result.weights):
            self.assertAlmostEqual(w, 2.5 * p[i] / denominator)

    def test_exact_floor_from_reference(self):
        self.assertEqual(WEIGHT_SUM_FLOOR, 2 ** -14)
        result = route(logits([1e-9] * 8), [0] * 8, top_k=8, scale=2.5)
        for w in result.weights:
            self.assertAlmostEqual(w, 2.5e-9 * 16384, places=15)
        self.assertLess(sum(result.weights), 2.5)

    def test_floor_boundary(self):
        for multiplier in (0.5, 1, 2):
            p = WEIGHT_SUM_FLOOR / 8 * multiplier
            result = route(logits([p] * 8), [0] * 8, top_k=8, scale=2.5)
            self.assertAlmostEqual(sum(result.weights), 2.5 * min(multiplier, 1))

    def test_extreme_logits_are_finite(self):
        result = route([1000, -1000, math.inf, -math.inf], [0, 2, 0, 3], top_k=4, scale=2.5)
        self.assertEqual(result.experts, (3, 1, 0, 2))
        self.assertEqual(result.weights, (0, 0, 1.25, 1.25))
        self.assertEqual(sigmoid(-1000), 0)
        self.assertEqual(sigmoid(1000), 1)

    def test_all_zero_probabilities(self):
        result = route([-1000] * 288, [0] * 288, top_k=8, scale=2.5)
        self.assertEqual(result.weights, (0,) * 8)

    def test_ties_use_documented_reference_convention(self):
        result = route([0] * 16, [0] * 16, top_k=8, scale=2.5)
        self.assertEqual(result.experts, tuple(range(8)))
        self.assertEqual(result.weights, (2.5 / 8,) * 8)

    def test_random_selection_by_independent_rank_count(self):
        rng = random.Random(53)
        for _ in range(20):
            x = [rng.uniform(-5, 5) for _ in range(288)]
            b = [rng.uniform(-1, 1) for _ in x]
            result = route(x, b, top_k=8, scale=2.5)
            probabilities = [1 / (1 + math.exp(-v)) for v in x]
            score = [p + bias for p, bias in zip(probabilities, b)]
            selected = {i for i in range(288) if sum(s > score[i] for s in score) < 8}
            self.assertEqual(set(result.experts), selected)
            self.assertAlmostEqual(sum(result.weights), 2.5)
            for i, w in zip(result.experts, result.weights):
                self.assertAlmostEqual(w, 2.5 * probabilities[i] / sum(probabilities[j] for j in selected))

    def test_invalid_input_rejected(self):
        for x, b, k, scale in (([0], [], 1, 2.5), ([0], [0], 0, 2.5), ([0], [0], 2, 2.5),
                               ([math.nan], [0], 1, 2.5), ([0], [math.inf], 1, 2.5), ([0], [0], 1, -1)):
            with self.assertRaises(ValueError):
                route(x, b, top_k=k, scale=scale)

    def test_metadata_binding(self):
        md = {"general.architecture": "glm5-next", "glm5-next.expert_count": 288,
              "glm5-next.expert_used_count": 8, "glm5-next.expert_weights_scale": 2.5,
              "glm5-next.expert_gating_func": 2, "glm5-next.expert_weights_norm": True,
              "glm5-next.expert_group_count": 1, "glm5-next.expert_group_used_count": 1}
        self.assertEqual(sum(route_from_metadata([0] * 288, [0] * 288, md).weights), 2.5)
        with self.assertRaisesRegex(ValueError, "expert_count"):
            route_from_metadata([0] * 287, [0] * 287, md)
        md["glm5-next.expert_group_count"] = 2
        with self.assertRaisesRegex(ValueError, "group selection"):
            route_from_metadata([0] * 288, [0] * 288, md)


if __name__ == "__main__":
    unittest.main()
