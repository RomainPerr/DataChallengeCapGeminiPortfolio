import unittest

from notebooks.site_profile_adaptation import RULES, adapt_plan


class SiteProfileAdaptationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.action_ids = {
            action_id for rule in RULES for action_id in rule.action_ids
        } | {"SOL-CAB-02"}
        self.action_library = {
            "actions": [
                {
                    "id": action_id,
                    "title": action_id,
                    "nature": "organisationnelle"
                    if action_id.startswith(("SOL-MAI", "SOL-INV"))
                    else "corrective",
                    "horizon": "court_terme",
                }
                for action_id in self.action_ids
            ]
        }
        self.plan = {
            "results": [
                {
                    "cell": {"site": "Site test"},
                    "candidates": [
                        {
                            "id": "SOL-CAB-02",
                            "title": "Remplacement connecteurs",
                            "nature": "corrective",
                        }
                    ],
                }
            ]
        }
        self.indicators = {
            indicator_id: {
                "unit": "percent",
                "values_by_site": {"Site test": 0.0},
                "relative_rank_by_site": {"Site test": "moyen"},
            }
            for rule in RULES
            for condition in rule.conditions
            for indicator_id in (condition.indicator,)
        }
        self.site_mapping = {
            "indicators": self.indicators,
            "profile_mapping": {"status": "ambiguous"},
        }

    def test_applies_direct_indicators_when_profile_is_ambiguous(self) -> None:
        self.plan["results"][0]["candidates"].extend(
            [
                {"id": "SOL-MAI-01", "title": "Plan préventif"},
                {"id": "SOL-INV-02", "title": "Suivi de récidive"},
            ]
        )
        self.indicators["maintenance_defaut"]["values_by_site"]["Site test"] = 10
        self.indicators["maintenance_defaut"]["relative_rank_by_site"]["Site test"] = (
            "haut"
        )
        result = adapt_plan(
            self.plan, self.site_mapping, self.action_library
        )["results"][0]

        self.assertEqual(
            [action["id"] for action in result["adapted_actions"][:2]],
            ["SOL-MAI-01", "SOL-INV-02"],
        )
        self.assertEqual(
            result["candidates"], self.plan["results"][0]["candidates"]
        )
        self.assertIn(
            "maintenance-defaillante",
            [rule["rule_id"] for rule in result["applied_rules"]],
        )

    def test_excludes_corrosion_action_only_when_measurement_is_zero(self) -> None:
        self.plan["results"][0]["candidates"].append(
            {"id": "SOL-COR-01", "title": "Anticorrosion"}
        )
        corrosion = self.indicators["environnement_corrosion_salin"]
        corrosion["relative_rank_by_site"]["Site test"] = "bas"
        result = adapt_plan(
            self.plan, self.site_mapping, self.action_library
        )["results"][0]
        self.assertIn(
            "SOL-COR-01",
            [item["action_id"] for item in result["excluded_actions"]],
        )

        corrosion["values_by_site"]["Site test"] = 0.1
        result = adapt_plan(
            self.plan, self.site_mapping, self.action_library
        )["results"][0]
        self.assertIn(
            "SOL-COR-01",
            [action["id"] for action in result["adapted_actions"]],
        )

    def test_adds_actions_from_age_environment_and_climate_indicators(self) -> None:
        self.plan["results"][0]["candidates"].append(
            {
                "id": "SOL-MET-02",
                "title": "Revue de structure",
                "nature": "conception",
            }
        )
        for indicator_id in (
            "age_vieillissement",
            "environnement_poussieres",
            "climat_vent_tempete",
        ):
            indicator = self.indicators[indicator_id]
            indicator["values_by_site"]["Site test"] = 4
            indicator["relative_rank_by_site"]["Site test"] = "haut"
        result = adapt_plan(
            self.plan, self.site_mapping, self.action_library
        )["results"][0]

        self.assertTrue(
            {
                "SOL-AGE-01",
                "SOL-ENC-01",
                "SOL-MET-02",
            }.issubset({item["id"] for item in result["adapted_actions"]})
        )
        self.assertEqual(result["adapted_actions"][0]["id"], "SOL-MET-02")

    def test_adds_and_phases_stock_for_measured_access_difficulty(self) -> None:
        access = self.indicators["maintenance_delay_or_access_issue"]
        access["values_by_site"]["Site test"] = 2
        access["relative_rank_by_site"]["Site test"] = "haut"
        result = adapt_plan(
            self.plan, self.site_mapping, self.action_library
        )["results"][0]
        stock = next(
            item for item in result["adapted_actions"] if item["id"] == "SOL-MAI-02"
        )
        self.assertEqual(stock["horizon"], "court_terme")
        self.assertEqual(stock["phase"], "court_terme")
        self.assertEqual(
            [item["id"] for item in result["adapted_actions"][:2]],
            ["SOL-MAI-02", "SOL-CAB-02"],
        )


if __name__ == "__main__":
    unittest.main()
