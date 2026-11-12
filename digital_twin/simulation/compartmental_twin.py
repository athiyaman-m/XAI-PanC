from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
import json


DAYS_PER_MONTH = 30.0
HOURS_PER_DAY = 24.0


@dataclass
class PancreaticTwinState:
    T: float = 0.0
    CA: float = 0.0
    CE: float = 0.0
    B: float = 0.0
    D: float = 0.0
    I: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.T, self.CA, self.CE, self.B, self.D, self.I])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> PancreaticTwinState:
        return cls(T=arr[0], CA=arr[1], CE=arr[2], B=arr[3], D=arr[4], I=arr[5])


@dataclass
class PancreaticTwinParams:
    r: float = 0.02
    K: float = 50.0
    alpha_ca: float = 1.5
    beta_ca: float = 0.03
    alpha_ce: float = 0.08
    beta_ce: float = 0.02
    delta_d: float = 0.15
    gamma_d: float = 0.25
    delta_i: float = 0.05
    sigma: float = 0.01
    k_sigma: float = 10.0
    alpha_b: float = 0.04
    beta_b: float = 0.10
    alpha_i: float = 0.03
    beta_i: float = 0.02
    k_i: float = 5.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.r, self.K,
            self.alpha_ca, self.beta_ca, self.alpha_ce, self.beta_ce,
            self.delta_d, self.gamma_d, self.delta_i,
            self.sigma, self.k_sigma, self.alpha_b, self.beta_b,
            self.alpha_i, self.beta_i, self.k_i,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> PancreaticTwinParams:
        return cls(
            r=arr[0], K=arr[1],
            alpha_ca=arr[2], beta_ca=arr[3], alpha_ce=arr[4], beta_ce=arr[5],
            delta_d=arr[6], gamma_d=arr[7], delta_i=arr[8],
            sigma=arr[9], k_sigma=arr[10], alpha_b=arr[11], beta_b=arr[12],
            alpha_i=arr[13], beta_i=arr[14], k_i=arr[15],
        )


@dataclass
class Intervention:
    day: int
    compartment: str
    amount: float
    label: str = ""

    def apply(self, state: PancreaticTwinState) -> PancreaticTwinState:
        delta = {k: 0.0 for k in ('T', 'CA', 'CE', 'B', 'D', 'I')}
        delta[self.compartment] = self.amount
        return PancreaticTwinState(
            T=state.T + delta['T'],
            CA=max(0, state.CA + delta['CA']),
            CE=max(0, state.CE + delta['CE']),
            B=max(0, state.B + delta['B']),
            D=np.clip(state.D + delta['D'], 0, 1),
            I=np.clip(state.I + delta['I'], 0, 1),
        )


class PancreaticTwinModel:
    def __init__(
        self,
        initial_state: PancreaticTwinState,
        params: PancreaticTwinParams,
        tumor_threshold: float = 0.1,
        obstruction_threshold: float = 3.0,
    ):
        self.state = initial_state
        self.params = params
        self.tumor_threshold = tumor_threshold
        self.obstruction_threshold = obstruction_threshold
        self.history: list[PancreaticTwinState] = [initial_state]
        self.interventions: list[Intervention] = []
        self.day = 0

    def dynamics(self, s: PancreaticTwinState, p: PancreaticTwinParams) -> PancreaticTwinState:
        if s.T > 0:
            dT = p.r * s.T * (1.0 - s.T / p.K) - p.delta_d * s.D * s.T - p.delta_i * s.I * s.T
        else:
            dT = 0.0
        dCA = p.alpha_ca * max(0, s.T) - p.beta_ca * s.CA
        dCE = p.alpha_ce * max(0, s.T) - p.beta_ce * s.CE
        obstruction = max(0, s.T - self.obstruction_threshold)
        dB = p.alpha_b * obstruction - p.beta_b * s.B
        dD = -p.gamma_d * s.D
        dI = p.alpha_i * s.T / (s.T + p.k_i) - p.beta_i * s.I
        return PancreaticTwinState(T=dT, CA=dCA, CE=dCE, B=dB, D=dD, I=dI)

    def update(self, days: int = 1):
        for _ in range(days):
            self.day += 1
            dt = 1.0
            deriv = self.dynamics(self.state, self.params)
            self.state = PancreaticTwinState(
                T=max(0, self.state.T + dt * deriv.T),
                CA=max(0, self.state.CA + dt * deriv.CA),
                CE=max(0, self.state.CE + dt * deriv.CE),
                B=max(0, self.state.B + dt * deriv.B),
                D=np.clip(self.state.D + dt * deriv.D, 0, 1),
                I=np.clip(self.state.I + dt * deriv.I, 0, 1),
            )
            for iv in self.interventions:
                if iv.day == self.day:
                    self.state = iv.apply(self.state)
            self.history.append(self.state)

    def run_forecast(self, days: int) -> list[PancreaticTwinState]:
        saved_state = self.state
        saved_day = self.day
        saved_history = list(self.history)
        self.update(days)
        forecast = list(self.history[saved_day:])
        self.state = saved_state
        self.day = saved_day
        self.history = saved_history
        return forecast

    def add_intervention(self, iv: Intervention):
        self.interventions.append(iv)

    def clear_interventions(self):
        self.interventions.clear()

    def scenario(self, days: int, new_state: PancreaticTwinState | None = None,
                 new_params: PancreaticTwinParams | None = None) -> list[PancreaticTwinState]:
        state = new_state if new_state is not None else self.state
        params = new_params if new_params is not None else self.params
        model = PancreaticTwinModel(state, params)
        model.interventions = list(self.interventions)
        model.update(days)
        return model.history


def fit_twin_parameters(
    cancer_prob: float,
    stage_label: str,
    ca199_value: float,
    cea_value: float,
    bilirubin_value: float,
    treatment_responses: dict[str, float] | None = None,
    kras_expr: float = 0.0,
    tp53_expr: float = 0.0,
    smad4_expr: float = 0.0,
    age: int = 60,
) -> PancreaticTwinParams:
    p = PancreaticTwinParams()
    base_r = 0.008 + cancer_prob * 0.025
    kras_boost = max(0, (kras_expr - 2.0) / 10.0) * 0.015
    tp53_boost = max(0, (5.0 - tp53_expr) / 5.0) * 0.008
    p.r = min(0.12, base_r + kras_boost + tp53_boost)
    stage_mult = {"Normal": 0.0, "Stage I": 8.0, "Stage II": 20.0,
                  "Stage III": 40.0, "Stage IV": 80.0}
    p.K = stage_mult.get(stage_label, 10.0) * (0.8 + 0.4 * cancer_prob)
    if ca199_value > 0 and p.K > 0:
        p.alpha_ca = ca199_value / (p.K * 0.6)
        p.beta_ca = max(0.01, ca199_value / (p.K * p.alpha_ca * 5.0))
    p.alpha_ca = np.clip(p.alpha_ca, 0.1, 8.0)
    p.beta_ca = np.clip(p.beta_ca, 0.005, 0.15)
    if cea_value > 0 and p.K > 0:
        p.alpha_ce = cea_value / (p.K * 0.4)
        p.beta_ce = max(0.01, cea_value / (p.K * p.alpha_ce * 5.0))
    p.alpha_ce = np.clip(p.alpha_ce, 0.01, 0.5)
    p.beta_ce = np.clip(p.beta_ce, 0.005, 0.10)
    if treatment_responses:
        best_response = max(treatment_responses.values())
        p.delta_d = 0.05 + best_response * 0.30
        smad4_loss = max(0, (3.0 - smad4_expr) / 3.0)
        p.delta_d *= (1.0 - smad4_loss * 0.5)
        p.gamma_d = 0.15 + 0.20 * (1 - best_response)
        p.delta_i = 0.02 + cancer_prob * 0.06
    else:
        p.delta_d = 0.05 + cancer_prob * 0.15
        p.gamma_d = 0.25
        p.delta_i = 0.02 + cancer_prob * 0.05
    if bilirubin_value > 1.2:
        p.alpha_b = bilirubin_value / max(0.1, p.K * 0.02)
    p.beta_b = 0.08 + 0.04 * (1 - cancer_prob)
    stage_risk = {"Normal": 0.0, "Stage I": 0.2, "Stage II": 0.4,
                  "Stage III": 0.6, "Stage IV": 0.9}
    p.sigma = stage_risk.get(stage_label, 0.1) * (0.005 + 0.015 * cancer_prob)
    p.sigma *= (1.0 + smad4_expr * 0.1) if smad4_expr > 2 else 1.0
    return p


def compute_twin_metrics(
    state: PancreaticTwinState,
    params: PancreaticTwinParams,
) -> tuple[float, float, str, float]:
    tumour_size = max(0, state.T)
    tumour_radius_cm = (3 * tumour_size / (4 * np.pi)) ** (1/3) if tumour_size > 0 else 0
    size_risk = min(50, tumour_size / params.K * 50)
    ca_risk = min(20, state.CA / 2000 * 20) if state.CA > 37 else 0
    ce_risk = min(10, state.CE / 50 * 10) if state.CE > 5 else 0
    bili_risk = min(15, state.B / 10 * 15) if state.B > 1.2 else 0
    immune_risk = min(5, (1 - state.I) * 5)
    risk = min(100, size_risk + ca_risk + ce_risk + bili_risk + immune_risk)
    health = max(0, 100 - risk * 1.1)
    if risk >= 70:
        stage = "Stage IV"
    elif risk >= 50:
        stage = "Stage III"
    elif risk >= 35:
        stage = "Stage II"
    elif risk >= 20:
        stage = "Stage I"
    else:
        stage = "Normal"
    return health, risk, stage, tumour_radius_cm


def forecast_compartment_trajectory(
    initial_state: PancreaticTwinState,
    params: PancreaticTwinParams,
    days: int,
    interventions: list[Intervention] | None = None,
) -> dict:
    model = PancreaticTwinModel(initial_state, params)
    if interventions:
        for iv in interventions:
            model.add_intervention(iv)
    model.update(days)
    days_arr = np.arange(len(model.history))
    T = np.array([s.T for s in model.history])
    CA = np.array([s.CA for s in model.history])
    CE = np.array([s.CE for s in model.history])
    B = np.array([s.B for s in model.history])
    D = np.array([s.D for s in model.history])
    I = np.array([s.I for s in model.history])
    health = np.full_like(T, np.nan)
    risk = np.full_like(T, np.nan)
    stage = []
    for i, s in enumerate(model.history):
        h, r, st, _ = compute_twin_metrics(s, params)
        health[i] = h
        risk[i] = r
        stage.append(st)
    return {
        "days": days_arr,
        "T": T, "CA": CA, "CE": CE, "B": B, "D": D, "I": I,
        "health": health, "risk": risk, "stage": stage,
        "params": params,
    }
