from dataclasses import dataclass
from typing import Optional
import copy

from ..pipeline.patient_profile import PatientProfile


@dataclass
class WhatIfResult:
    parameter: str
    before_value: any
    after_value: any
    before_prob: float
    after_prob: float
    delta_prob: float
    before_stage: str
    after_stage: str
    before_survival_months: float
    after_survival_months: float
    before_organ_risks: dict[str, tuple]
    after_organ_risks: dict[str, tuple]

    @property
    def improvement(self) -> bool:
        return self.delta_prob < 0


class InterventionSimulator:
    def __init__(self, prediction_fn=None):
        self._predict = prediction_fn

    def set_predictor(self, prediction_fn):
        self._predict = prediction_fn

    def simulate(self, patient: PatientProfile, parameter: str, new_value: any) -> Optional[WhatIfResult]:
        if self._predict is None:
            return None
        baseline = self._predict(patient.to_dict())
        modified = copy.deepcopy(patient)
        if hasattr(modified, parameter):
            setattr(modified, parameter, new_value)
        else:
            inputs = modified.to_dict()
            if parameter in inputs:
                inputs[parameter] = new_value
                modified = PatientProfile.from_dict(inputs)
            else:
                return None
        scenario = self._predict(modified.to_dict())
        return WhatIfResult(
            parameter=parameter,
            before_value=getattr(patient, parameter, patient.to_dict().get(parameter)),
            after_value=new_value,
            before_prob=baseline.get('prob', 0.0),
            after_prob=scenario.get('prob', 0.0),
            delta_prob=scenario.get('prob', 0.0) - baseline.get('prob', 0.0),
            before_stage=baseline.get('stage_label', 'Unknown'),
            after_stage=scenario.get('stage_label', 'Unknown'),
            before_survival_months=baseline.get('survival', {}).get('median_months', 0),
            after_survival_months=scenario.get('survival', {}).get('median_months', 0),
            before_organ_risks=baseline.get('organ_risks', {}),
            after_organ_risks=scenario.get('organ_risks', {}),
        )

    def simulate_multi(self, patient: PatientProfile, changes: dict[str, any]) -> list[WhatIfResult]:
        return [
            self.simulate(patient, param, val)
            for param, val in changes.items()
        ]
