import unittest

from notebooks.report_generation import (
    render_abbreviated_markdown,
    render_markdown,
)


class ReportGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = {
            "generated_at": "2026-10-22",
            "source_plan": "final_action_plan_solar.json",
            "actions": [
                {
                    "identifiant": "ACT-001",
                    "action_id": "SOL-A",
                    "site": "Site A",
                    "cause_ciblee": {
                        "cluster_name": "Défaut électrique",
                        "trigger_category": "cause_indeterminee",
                        "failure_modes": ["incendie"],
                        "components": ["onduleur"],
                    },
                    "preuve": {
                        "nombre_incidents": 2,
                        "part_cout_pct": 1.25,
                        "cout_historique_cellule_eur": 1200,
                        "exemples": [
                            {
                                "incident_id": "I-1",
                                "date": "2024-01-02",
                                "equipment_id": "INV-1",
                                "evidence_level": "explicite",
                                "trigger_quote": "défaut | détecté",
                            }
                        ],
                    },
                    "action": {
                        "titre": "Contrôle sécurité",
                        "description": "Réaliser un contrôle ciblé.",
                        "responsable_role": "Responsable HSE",
                        "effort": "S",
                    },
                    "sequencement": {
                        "horizon": "immediat",
                        "phase": "Immédiat · 0 à 30 jours",
                        "echeance": "2026-11-21",
                    },
                    "kpi": {
                        "name": "recidive_90j",
                        "definition": "Récidive équipement sous 90 jours",
                        "direction": "baisse",
                        "baseline": {
                            "value": 0.5,
                            "numerator": 1,
                            "denominator": 2,
                            "unit": "part",
                        },
                    },
                    "cible_kpi": {
                        "proposed": 0.45,
                        "unit": "part",
                        "note": "Cible indicative.",
                    },
                    "gain_attendu_eur": {
                        "low": 50,
                        "central": 100,
                        "high": 200,
                        "hypothese": True,
                        "scenario": "faible",
                    },
                    "risque_inaction": "Risque de répétition.",
                    "confiance": {
                        "accord_extraction": {
                            "value": 1,
                            "numerator": 2,
                            "denominator": 2,
                        },
                        "verdict_test": {
                            "statut": "non_testé",
                            "raison": "Aucune mesure post-action.",
                        },
                    },
                    "suivi": {
                        "statut_mise_en_oeuvre": "a_planifier",
                        "date_mise_en_oeuvre": None,
                    },
                },
                {
                    "identifiant": "ACT-002",
                    "action_id": "SOL-B",
                    "site": "Site A",
                    "cause_ciblee": {"cluster_name": "Maintenance"},
                    "preuve": {"exemples": []},
                    "action": {"titre": "Plan de maintenance", "effort": "M"},
                    "sequencement": {
                        "horizon": "structurel",
                        "phase": "Structurel",
                        "echeance": "2028-10-22",
                    },
                    "kpi": {"name": "incidents", "baseline": None},
                    "cible_kpi": {"proposed": None},
                    "gain_attendu_eur": {"hypothese": False},
                    "confiance": {},
                    "suivi": {},
                },
            ],
        }
        self.site_mapping = {
            "profile_mapping": {
                "status": "ambiguous",
                "best_candidates": [
                    {
                        "profile_to_site": {"A": "Site A", "B": "Site B"},
                        "directional_agreement": 0.7,
                    },
                    {
                        "profile_to_site": {"C": "Site A", "B": "Site B"},
                        "directional_agreement": 0.7,
                    },
                ],
            },
            "indicators": {
                "maintenance_defaut": {
                    "relative_rank_by_site": {
                        "Site A": "haut",
                        "Site B": "moyen",
                    }
                },
                "climat_vent_tempete": {
                    "relative_rank_by_site": {"Site A": "faible"}
                },
            },
        }

    def test_renders_complete_action_cards_in_phase_order(self) -> None:
        report = render_markdown(self.plan)

        self.assertLess(
            report.index("## Immédiat"),
            report.index("## Structurel"),
        )
        for expected in (
            "ACT-001",
            "Site A",
            "Défaut électrique",
            "2 incidents",
            "Responsable HSE",
            "2026-11-21",
            "recidive_90j",
            "50,0 %",
            "hypothèse",
            "non_testé",
            "Risque de répétition",
            "pas un accord inter-évaluateurs",
        ):
            self.assertIn(expected, report)
        self.assertIn(r"défaut \| détecté", report)
        self.assertIn("ne doivent pas être additionnées", report)
        self.assertIn("Aucun exemple", report)

    def test_rejects_unknown_horizon(self) -> None:
        self.plan["actions"][0]["sequencement"]["horizon"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Horizon invalide"):
            render_markdown(self.plan)

    def test_renders_abbreviated_report_with_mapping_and_collapsible_sites(self) -> None:
        report = render_abbreviated_markdown(self.plan, self.site_mapping)

        self.assertTrue(report.startswith("# Plan de remédiation solaire — synthèse"))
        self.assertIn("## Introduction", report)
        self.assertIn("## Mapping des sites", report)
        self.assertIn("**Statut de l’affectation aux profils : ambiguous.**", report)
        self.assertIn("`A` / `C`", report)
        self.assertIn("maintenance : haut", report)
        self.assertIn("<summary><strong>Site A</strong> · 2 actions</summary>", report)
        self.assertIn("| Échéance | Cause ciblée | Effort | Gain attendu (hypothèse) | Recommandation |", report)
        self.assertLess(report.index("2026-11-21"), report.index("2028-10-22"))
        self.assertIn("basse 50,00 € · centrale 100,00 € · haute 200,00 €", report)
        self.assertIn("Contrôle sécurité — Réaliser un contrôle ciblé.", report)
        self.assertNotIn("responsable_role", report)
        self.assertNotIn("Baseline", report)


if __name__ == "__main__":
    unittest.main()
