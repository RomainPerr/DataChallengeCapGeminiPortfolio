import unittest
from datetime import date

from notebooks.final_action_plan import build_final_action_plan


class FinalActionPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.impact_plan = {
            "asset_type": "solar",
            "results": [
                {
                    "cell": {
                        "site": "Site A",
                        "cluster_code": "C_1",
                        "cluster_name": "Défaillance connecteur",
                        "trigger_category": "cause_indeterminee",
                        "failure_modes": ["degradation_physique"],
                    },
                    "stats": {
                        "n_incidents": 2,
                        "cout_total_eur": 4000,
                        "part_cout_pct": 2.5,
                        "score_safety_risk": 0.1,
                    },
                    "example_incidents": [{"incident_id": "I1"}],
                    "adapted_actions": [
                        {
                            "id": "SOL-A",
                            "title": "Contrôler",
                            "description": "Contrôle ciblé",
                            "effort": "M",
                            "horizon": "court_terme",
                            "impact_estimation": {
                                "is_hypothesis": True,
                                "effectiveness_scenario": "moyenne",
                                "expected_gain_eur": {
                                    "low": 400,
                                    "central": 1000,
                                    "high": 1600,
                                },
                            },
                        }
                    ],
                }
            ],
        }
        self.incidents = [
            {
                "incident_id": "I1",
                "site": "Site A",
                "cluster_code": "C_1",
                "trigger_category": "cause_indeterminee",
                "date": "2024-01-01",
                "equipment_id": "INV-1",
                "evidence_level": "explicite",
                "failure_mode": "degradation_physique",
                "component": "connecteur",
                "trigger_quote": "connecteur dégradé",
                "detection_context": "controle_periodique",
            },
            {
                "incident_id": "I2",
                "site": "Site A",
                "cluster_code": "C_1",
                "trigger_category": "cause_indeterminee",
                "date": "2024-01-20",
                "equipment_id": "INV-1",
                "evidence_level": "suggeree",
                "failure_mode": "degradation_physique",
                "component": "connecteur",
                "trigger_quote": "dégradation probable",
                "detection_context": "inspection_visuelle_ou_thermique",
            },
            {
                "incident_id": "I3",
                "site": "Site A",
                "cluster_code": "C_2",
                "trigger_category": "autre_cause",
                "date": "2024-05-01",
                "equipment_id": "INV-2",
                "evidence_level": "non_indiquee",
                "failure_mode": "",
                "component": "",
                "trigger_quote": "",
                "detection_context": "",
            },
        ]
        self.library = {
            "actions": [
                {
                    "id": "SOL-A",
                    "title": "Contrôle",
                    "description": "Contrôle de connecteurs",
                    "effort": "M",
                    "horizon": "court_terme",
                    "owner_role": "Responsable O&M",
                    "evidence_basis": "jugement_expert",
                    "kpi": {
                        "name": "recidive_90j",
                        "definition": "Récidive équipement à 90 jours",
                        "direction": "baisse",
                    },
                }
            ]
        }

    def test_builds_scheduled_action_sheet_and_kpi_baseline(self) -> None:
        plan = build_final_action_plan(
            self.impact_plan,
            self.incidents,
            self.library,
            plan_date=date(2026, 10, 22),
        )
        sheet = plan["actions"][0]

        self.assertEqual(sheet["sequencement"]["phase"], "Court terme (1 à 6 mois)")
        self.assertEqual(sheet["sequencement"]["echeance"], "2027-04-22")
        self.assertEqual(sheet["action"]["responsable_role"], "Responsable O&M")
        self.assertEqual(sheet["action"]["effort"], "M")
        self.assertEqual(sheet["preuve"]["nombre_incidents"], 2)
        self.assertEqual(sheet["preuve"]["exemples"][0]["equipment_id"], "INV-1")
        self.assertEqual(sheet["kpi"]["baseline"]["numerator"], 1)
        self.assertEqual(sheet["kpi"]["baseline"]["denominator"], 2)
        self.assertEqual(sheet["confiance"]["accord_extraction"]["value"], 1.0)
        self.assertEqual(sheet["confiance"]["verdict_test"]["statut"], "non_testé")
        self.assertEqual(sheet["gain_attendu_eur"]["central"], 1000)

    def test_completed_action_replay_compares_post_implementation_kpi(self) -> None:
        first_plan = build_final_action_plan(
            self.impact_plan,
            self.incidents,
            self.library,
            plan_date=date(2026, 10, 22),
        )
        sheet = first_plan["actions"][0]
        sheet["suivi"]["statut_mise_en_oeuvre"] = "terminee"
        sheet["suivi"]["date_mise_en_oeuvre"] = "2024-05-01"
        self.incidents.append(
            {
                "incident_id": "I4",
                "site": "Site A",
                "cluster_code": "C_1",
                "trigger_category": "cause_indeterminee",
                "date": "2024-06-15",
                "equipment_id": "INV-1",
                "evidence_level": "explicite",
                "failure_mode": "degradation_physique",
                "component": "connecteur",
                "trigger_quote": "nouvelle dégradation",
                "detection_context": "controle_periodique",
            }
        )
        self.incidents.extend(
            [
                {
                    "incident_id": "I5",
                    "site": "Site A",
                    "cluster_code": "C_1",
                    "trigger_category": "cause_indeterminee",
                    "date": "2024-07-01",
                    "equipment_id": "INV-1",
                    "evidence_level": "explicite",
                    "failure_mode": "degradation_physique",
                    "component": "connecteur",
                    "trigger_quote": "nouvelle dégradation",
                    "detection_context": "controle_periodique",
                },
                {
                    "incident_id": "I6",
                    "site": "Site A",
                    "cluster_code": "C_2",
                    "trigger_category": "autre_cause",
                    "date": "2024-10-15",
                    "equipment_id": "INV-2",
                    "evidence_level": "non_indiquee",
                    "failure_mode": "",
                    "component": "",
                    "trigger_quote": "",
                    "detection_context": "",
                },
            ]
        )

        replay = build_final_action_plan(
            self.impact_plan,
            self.incidents,
            self.library,
            plan_date=date(2026, 10, 22),
            previous_plan=first_plan,
        )
        replayed_sheet = replay["actions"][0]
        self.assertEqual(
            replayed_sheet["suivi"]["derniere_mesure"]["observation_start"],
            "2024-05-01",
        )
        self.assertIn(
            replayed_sheet["confiance"]["verdict_test"]["statut"],
            {"cible_atteinte", "cible_non_atteinte"},
        )

    def test_safety_actions_are_scheduled_immediately(self) -> None:
        self.impact_plan["results"][0]["cell"]["trigger_category"] = "incendie"
        self.impact_plan["results"][0]["adapted_actions"][0]["nature"] = "securite"
        self.library["actions"][0]["nature"] = "securite"
        self.library["actions"][0]["horizon"] = "court_terme"
        for incident in self.incidents[:2]:
            incident["trigger_category"] = "incendie"
        plan = build_final_action_plan(
            self.impact_plan,
            self.incidents,
            self.library,
            plan_date=date(2026, 10, 22),
        )
        self.assertEqual(
            plan["actions"][0]["sequencement"]["horizon"], "immediat"
        )
        self.assertEqual(plan["actions"][0]["sequencement"]["echeance"], "2026-11-21")


if __name__ == "__main__":
    unittest.main()
