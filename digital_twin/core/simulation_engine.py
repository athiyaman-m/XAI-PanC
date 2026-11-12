from typing import Optional, Callable
from copy import deepcopy

from ..pipeline.patient_profile import PatientProfile
from ..simulation.intervention_simulator import InterventionSimulator, WhatIfResult
from .twin_state import TwinState


class SimulationEngine:
    def __init__(self, predictor_fn: Optional[Callable] = None):
        self._simulator = InterventionSimulator(prediction_fn=predictor_fn)
        self._predictor = predictor_fn

    def set_predictor(self, fn: Callable):
        self._predictor = fn
        self._simulator.set_predictor(fn)

    def what_if(self, patient: PatientProfile, parameter: str,
                new_value: any) -> Optional[WhatIfResult]:
        return self._simulator.simulate(patient, parameter, new_value)

    def what_if_multi(self, patient: PatientProfile,
                      changes: dict[str, any]) -> list[WhatIfResult]:
        return self._simulator.simulate_multi(patient, changes)

    def compare_scenarios(self, patient: PatientProfile,
                          scenarios: list[dict]) -> list[WhatIfResult]:
        all_results = []
        for changes in scenarios:
            modified = deepcopy(patient)
            for param, val in changes.items():
                if hasattr(modified, param):
                    setattr(modified, param, val)
            if self._predictor:
                result = self._predictor(modified.to_dict())
                all_results.append(result)
        return all_results

    def what_if_batch(self, patient: PatientProfile,
                      scenarios: list[dict]) -> list[tuple[WhatIfResult, TwinState]]:
        results = []
        for changes in scenarios:
            modified = deepcopy(patient)
            for param, val in changes.items():
                if hasattr(modified, param):
                    setattr(modified, param, val)
            if self._predictor:
                result = self._predictor(modified.to_dict())
                twin_state = TwinState(patient=modified)
                twin_state.update_from_prediction(result)
                whatif = WhatIfResult(
                    parameter=list(changes.keys())[0],
                    before_value=None, after_value=None,
                    before_prob=0, after_prob=0,
                    delta_prob=result.get('prob', 0),
                    before_stage='', after_stage=result.get('stage_label', ''),
                    before_survival_months=0, after_survival_months=0,
                    before_organ_risks={}, after_organ_risks=result.get('organ_risks', {}),
                )
                results.append((whatif, twin_state))
        return results
