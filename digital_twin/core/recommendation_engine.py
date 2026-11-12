from typing import Optional
from ..pipeline.patient_profile import PatientProfile
from .twin_state import TwinState


class RecommendationEngine:
    def __init__(self):
        self.rules: list[dict] = [
            {"condition": "high_risk", "priority": 1,
             "recommendations": [
                 "Refer to oncologist for urgent evaluation",
                 "MRI with MRCP for detailed pancreatic imaging",
                 "EUS-FNA for tissue diagnosis if not already performed",
                 "CT chest/abdomen/pelvis for staging",
                 "Multi-disciplinary team (MDT) discussion",
             ]},
            {"condition": "elevated_ca199", "priority": 2,
             "recommendations": [
                 "Trend CA 19-9 levels monthly",
                 "Consider biliary stenting if jaundice present",
                 "Evaluate for resectability with dedicated pancreatic protocol CT",
             ]},
            {"condition": "late_stage", "priority": 3,
             "recommendations": [
                 "Consider neoadjuvant chemotherapy (FOLFIRINOX or Gem+Nab-P)",
                 "Assess performance status for treatment tolerance",
                 "Pain management and palliative care consultation",
                 "Nutritional support and pancreatic enzyme replacement",
             ]},
            {"condition": "family_history", "priority": 4,
             "recommendations": [
                 "Refer for genetic counselling (BRCA1/2, PALB2 testing)",
                 "Consider screening for at-risk family members",
                 "Discuss surveillance options with genetics team",
             ]},
            {"condition": "diabetes_new_onset", "priority": 5,
             "recommendations": [
                 "Monitor glycemic control closely",
                 "Consider new-onset diabetes as a paraneoplastic manifestation",
                 "Endocrine consultation for diabetes management",
             ]},
            {"condition": "default", "priority": 6,
             "recommendations": [
                 "Continue routine surveillance",
                 "Maintain healthy lifestyle with balanced diet and exercise",
                 "Annual physical examination and routine labs",
             ]},
        ]

    def generate(self, patient: PatientProfile, twin_state: TwinState) -> list[str]:
        matched = set()
        if twin_state.prob:
            prob = twin_state.prob
        else:
            prob = 0.0

        if prob > 0.6:
            matched.update(self._get_recommendations("high_risk"))
        if patient.ca199 > 37:
            matched.update(self._get_recommendations("elevated_ca199"))
        if "III" in twin_state.stage_label or "IV" in twin_state.stage_label:
            matched.update(self._get_recommendations("late_stage"))
        if patient.family_history:
            matched.update(self._get_recommendations("family_history"))
        if "New-onset" in (patient.diabetes or ""):
            matched.update(self._get_recommendations("diabetes_new_onset"))
        if not matched:
            matched.update(self._get_recommendations("default"))

        twin_state.recommendations = list(matched)
        return list(matched)

    def _get_recommendations(self, condition: str) -> list:
        for rule in self.rules:
            if rule["condition"] == condition:
                return rule["recommendations"]
        return []
