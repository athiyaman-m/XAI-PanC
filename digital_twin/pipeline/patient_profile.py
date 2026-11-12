from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PatientProfile:
    patient_id: str = "PT-2024-001"
    age: int = 65
    sex: str = "Male"
    ca199: float = 45.0
    cea: float = 3.5
    bilirubin: float = 0.8
    amylase: float = 95.0
    weight_loss_pct: float = 5.0
    smoking: str = "Never"
    alcohol: str = "None / Minimal"
    diabetes: str = "None"
    family_history: bool = False
    pancreatitis: bool = False
    genetic: bool = False
    kras_expr: float = 5.0
    tp53_expr: float = 5.0
    smad4_expr: float = 5.0
    erbb2_expr: float = 5.0

    @property
    def gender_male(self) -> int:
        return 1 if self.sex == "Male" else 0

    @property
    def age_risk_baseline(self) -> float:
        score = max(0.0, (self.age - 40) / 80.0) * 0.20
        if self.age >= 70:
            score += 0.05
        elif self.age >= 60:
            score += 0.02
        return score

    def to_clinical_inputs(self) -> dict:
        return {
            "ca199": self.ca199,
            "cea": self.cea,
            "bilirubin": self.bilirubin,
            "amylase": self.amylase,
            "weight_loss_pct": self.weight_loss_pct,
            "smoking": self.smoking,
            "alcohol": self.alcohol,
            "diabetes": self.diabetes,
            "family_history": self.family_history,
            "pancreatitis": self.pancreatitis,
            "genetic": self.genetic,
            "kras_expr": self.kras_expr,
            "tp53_expr": self.tp53_expr,
            "smad4_expr": self.smad4_expr,
            "erbb2_expr": self.erbb2_expr,
            "age": self.age,
        }

    def to_dict(self) -> dict:
        return {
            "age": self.age,
            "sex": self.sex,
            "ca199": self.ca199,
            "cea": self.cea,
            "bilirubin": self.bilirubin,
            "amylase": self.amylase,
            "weight_loss_pct": self.weight_loss_pct,
            "smoking": self.smoking,
            "alcohol": self.alcohol,
            "diabetes": self.diabetes,
            "family_history": self.family_history,
            "pancreatitis": self.pancreatitis,
            "genetic": self.genetic,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PatientProfile":
        return cls(
            patient_id=data.get("patient_name", data.get("patient_id", "PT-2024-001")),
            age=int(data.get("age", 65)),
            sex=str(data.get("sex", "Male")),
            ca199=float(data.get("ca199", 45.0)),
            cea=float(data.get("cea", 3.5)),
            bilirubin=float(data.get("bilirubin", 0.8)),
            amylase=float(data.get("amylase", 95.0)),
            weight_loss_pct=float(data.get("weight_loss_pct", 5.0)),
            smoking=str(data.get("smoking", "Never")),
            alcohol=str(data.get("alcohol", "None / Minimal")),
            diabetes=str(data.get("diabetes", "None")),
            family_history=bool(data.get("family_history", False)),
            pancreatitis=bool(data.get("pancreatitis", False)),
            genetic=bool(data.get("genetic", False)),
        )
