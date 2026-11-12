import pandas as pd
from typing import Optional
from .patient_profile import PatientProfile


class DataPipeline:
    @staticmethod
    def parse_csv(uploaded_file) -> Optional[dict]:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                return None

            def get(keys, default, cast=float):
                for k in keys:
                    for col in df.columns:
                        if col.strip().lower() == k.lower():
                            try:
                                return cast(row[col])
                            except Exception:
                                pass
                return default

            row = df.iloc[0]
            return {
                "patient_name": str(row.get("patient_id", row.get("Patient ID", "PT-CSV-001"))),
                "age": int(get(["age"], 65, float)),
                "sex": str(row.get("sex", row.get("Sex", "Male"))),
                "ca199": get(["ca199", "ca 19-9", "ca19-9"], 45.0),
                "cea": get(["cea"], 3.5),
                "bilirubin": get(["bilirubin", "total bilirubin"], 0.8),
                "amylase": get(["amylase"], 95.0),
                "weight_loss_pct": get(["weight_loss_pct", "weight loss (%)"], 5.0),
                "smoking": str(row.get("smoking", "Never")),
                "alcohol": str(row.get("alcohol", "None / Minimal")),
                "diabetes": str(row.get("diabetes", "None")),
                "family_history": bool(int(get(["family_history"], 0, float))),
                "pancreatitis": bool(int(get(["pancreatitis"], 0, float))),
                "genetic": bool(int(get(["genetic"], 0, float))),
            }
        except Exception as e:
            raise ValueError(f"CSV parse error: {e}")

    @staticmethod
    def generate_template_csv() -> str:
        return pd.DataFrame([{
            "patient_id": "PT-2024-001", "age": 65, "sex": "Male",
            "ca199": 45.0, "cea": 3.5, "bilirubin": 0.8,
            "amylase": 95, "weight_loss_pct": 5, "smoking": "Never",
            "alcohol": "None / Minimal", "diabetes": "None",
            "family_history": 0, "pancreatitis": 0, "genetic": 0,
        }]).to_csv(index=False)

    @staticmethod
    def dict_to_patient(data: dict) -> PatientProfile:
        return PatientProfile.from_dict(data)
