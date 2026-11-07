import unittest

from notebooks.impact_estimation import estimate_plan_impact


class ImpactEstimationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = {
            "results": [
                {
                    "stats": {"cout_total_eur": 1000},
                    "adapted_actions": [
                        {"id": "SOL-A", "title": "Action A"},
                        {"id": "SOL-B", "title": "Action B"},
                    ],
                }
            ]
        }
        self.library = {
            "_meta": {
                "effectiveness_scenarios": {
                    "note": "Des hypothèses, sans valeur contractuelle.",
                    "faible": {"low": 0.05, "central": 0.1, "high": 0.2},
                    "forte": {"low": 0.25, "central": 0.5, "high": 0.7},
                }
            },
            "actions": [
                {
                    "id": "SOL-A",
                    "expected_effectiveness": "faible",
                    "effort": "S",
                },
                {
                    "id": "SOL-B",
                    "expected_effectiveness": "forte",
                    "effort": "L",
                },
            ],
        }

    def test_calculates_low_central_high_gain_and_effort_per_action(self) -> None:
        estimated = estimate_plan_impact(self.plan, self.library)
        actions = estimated["results"][0]["adapted_actions"]

        self.assertEqual(
            actions[0]["impact_estimation"]["expected_gain_eur"],
            {"low": 50.0, "central": 100.0, "high": 200.0},
        )
        self.assertEqual(
            actions[1]["impact_estimation"]["expected_gain_eur"],
            {"low": 250.0, "central": 500.0, "high": 700.0},
        )
        self.assertEqual([action["effort"] for action in actions], ["S", "L"])
        self.assertTrue(
            all(action["impact_estimation"]["is_hypothesis"] for action in actions)
        )
        self.assertNotIn("impact_estimation", self.plan["results"][0]["adapted_actions"][0])

    def test_rejects_missing_action_and_invalid_cost(self) -> None:
        self.plan["results"][0]["adapted_actions"][0]["id"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "Action absente"):
            estimate_plan_impact(self.plan, self.library)

        self.plan["results"][0]["adapted_actions"] = []
        self.plan["results"][0]["stats"]["cout_total_eur"] = None
        with self.assertRaisesRegex(ValueError, "Coût de cellule invalide"):
            estimate_plan_impact(self.plan, self.library)


if __name__ == "__main__":
    unittest.main()
