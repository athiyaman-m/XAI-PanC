from enum import StrEnum


class UserRole(StrEnum):
    ADMINISTRATOR = "Administrator"
    RADIOLOGIST = "Radiologist"
    ONCOLOGIST = "Oncologist"
    RESEARCHER = "Researcher"
    CLINICIAN = "Clinician"
    DEMO_USER = "Demo User"


class WorkflowStep(StrEnum):
    PATIENT_REGISTRY = "patient_registry"
    CREATE_PATIENT = "create_patient"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    LABORATORY_RESULTS = "laboratory_results"
    MEDICAL_IMAGING = "medical_imaging"
    GENOMIC_ANALYSIS = "genomic_analysis"
    DATA_VALIDATION = "data_validation"
    RUN_AI_ANALYSIS = "run_ai_analysis"
    DIGITAL_TWIN = "digital_twin"
    DISEASE_SIMULATION = "disease_simulation"
    TREATMENT_PLANNING = "treatment_planning"
    TREATMENT_COMPARISON = "treatment_comparison"
    CLINICAL_REPORT = "clinical_report"
    EXPORT_SAVE = "export_save"


WORKFLOW_ORDER: list[WorkflowStep] = list(WorkflowStep)


class ImagingModality(StrEnum):
    CT = "CT"
    MRI = "MRI"
    PET = "PET"
    EUS = "EUS"
    DICOM = "DICOM"
    NIFTI = "NIfTI"


class SegmentationStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    QA_APPROVED = "qa_approved"


class PredictionStatus(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class VisitStatus(StrEnum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"
