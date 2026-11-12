from typing import Optional
from .patient_profile import PatientProfile


class FeatureEngineering:
    def __init__(self):
        self._feature_names = ['age', 'gender_male']

    def extract_features(self, patient: PatientProfile) -> dict:
        return {
            "age": patient.age,
            "gender_male": patient.gender_male,
            "ca199": patient.ca199,
            "cea": patient.cea,
            "bilirubin": patient.bilirubin,
            "amylase": patient.amylase,
            "weight_loss_pct": patient.weight_loss_pct,
            "smoking": patient.smoking,
            "alcohol": patient.alcohol,
            "diabetes": patient.diabetes,
            "family_history": patient.family_history,
            "pancreatitis": patient.pancreatitis,
            "genetic": patient.genetic,
        }

    def clinical_inputs(self, patient: PatientProfile) -> dict:
        return patient.to_clinical_inputs()

    @property
    def feature_names(self) -> list:
        return self._feature_names
