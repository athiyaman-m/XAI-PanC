"""
Core database models for ExAI-PanC v2.
These are the SQLAlchemy models for the clinical decision support platform.
"""
from uuid import uuid4
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class UserRole(str, Enum):
    ADMINISTRATOR = "Administrator"
    RADIOLOGIST = "Radiologist"
    ONCOLOGIST = "Oncologist"
    RESEARCHER = "Researcher"
    CLINICIAN = "Clinician"
    DEMO_USER = "Demo User"


class WorkflowStep(str, Enum):
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


class VisitStatus(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ImagingModality(str, Enum):
    CT = "CT"
    MRI = "MRI"
    PET = "PET"
    EUS = "EUS"
    DICOM = "DICOM"
    NIFTI = "NIfTI"


class SegmentationStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    QA_APPROVED = "qa_approved"


class PredictionStatus(str, Enum):
    QUEUED = "queued"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    user_roles = relationship("UserRole", back_populates="user")
    patients = relationship("Patient", back_populates="user")
    visits = relationship("Visit", back_populates="patient")
    created_reviews = relationship("AuditLog", back_populates="reviewer")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user_roles = relationship("UserRole", back_populates="role")


class UserRole(Base):
    __tablename__ = "user_roles"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.id"), primary_key=True)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    assigned_by = Column(String, ForeignKey("users.id"), nullable=True)

    # Relationships
    user = relationship("User", back_populates="user_roles")
    role = relationship("Role", back_populates="user_roles")
    assigned_by_user = relationship("User", foreign_keys=[assigned_by], back_populates="created_reviews")


class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(String(50), primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    mrn = Column(String(50), unique=True, nullable=False, index=True)
    date_of_birth = Column(DateTime)
    sex = Column(String(20))
    blood_type = Column(String(10))
    emergency_contact = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="patients")
    visits = relationship("Visit", back_populates="patient")
    lab_results = relationship("LabResult", back_populates="patient")
    gene_expressions = relationship("GeneExpression", back_populates="patient")
    biomarkers = relationship("Biomarker", back_populates="patient")
    predictions = relationship("Prediction", back_populates="patient")
    twin_states = relationship("TwinState", back_populates="patient")
    treatments = relationship("Treatment", back_populates="patient")
    clinical_reports = relationship("ClinicalReport", back_populates="patient")


class Visit(Base):
    __tablename__ = "visits"

    visit_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    visit_date = Column(DateTime, nullable=False)
    visit_type = Column(String(50))
    admitting_doctor = Column(String(100))
    primary_diagnosis = Column(String(200))
    status = Column(String(50), default=VisitStatus.DRAFT)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="visits")
    clinical_assessments = relationship("ClinicalAssessment", back_populates="visit")
    lab_results = relationship("LabResult", back_populates="visit")
    imaging_studies = relationship("ImagingStudy", back_populates="visit")
    gene_expressions = relationship("GeneExpression", back_populates="visit")
    twin_states = relationship("TwinState", back_populates="visit")
    workflow_states = relationship("WorkflowState", back_populates="visit")


class ClinicalAssessment(Base):
    __tablename__ = "clinical_assessments"

    assessment_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    visit_id = Column(String, ForeignKey("visits.visit_id"), index=True)
    assessment_date = Column(DateTime, default=datetime.utcnow)
    symptoms = Column(Text)
    physical_exam = Column(Text)
    diagnosis = Column(Text)
    treatment_plan = Column(Text)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    visit = relationship("Visit", back_populates="clinical_assessments")


class LabResult(Base):
    __tablename__ = "lab_results"

    result_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    visit_id = Column(String, ForeignKey("visits.visit_id"), index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    test_name = Column(String(100), nullable=False, index=True)
    value = Column(Float)
    unit = Column(String(20))
    reference_range_low = Column(Float)
    reference_range_high = Column(Float)
    collected_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    visit = relationship("Visit", back_populates="lab_results")
    patient = relationship("Patient", back_populates="lab_results")


class ImagingStudy(Base):
    __tablename__ = "imaging_studies"

    study_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    visit_id = Column(String, ForeignKey("visits.visit_id"), index=True)
    study_type = Column(String(50), nullable=False)
    modality = Column(String(50))
    study_date = Column(DateTime, nullable=False)
    accession_number = Column(String(100))
    study_instance_uid = Column(String(255), unique=True)
    file_path = Column(String(500))
    storage_location = Column(String(100))
    status = Column(String(50), default=SegmentationStatus.PENDING, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    visit = relationship("Visit", back_populates="imaging_studies")


class GeneExpression(Base):
    __tablename__ = "gene_expressions"

    expression_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    visit_id = Column(String, ForeignKey("visits.visit_id"), index=True)
    sample_id = Column(String(100), nullable=False)
    gene_symbol = Column(String(100), nullable=False, index=True)
    expression_value = Column(Float)
    mutation_status = Column(String(50))
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    patient = relationship("Patient", back_populates="gene_expressions")
    visit = relationship("Visit", back_populates="gene_expressions")


class Biomarker(Base):
    __tablename__ = "biomarkers"

    biomarker_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    visit_id = Column(String, ForeignKey("visits.visit_id"), index=True)
    biomarker_name = Column(String(50), nullable=False, index=True)
    value = Column(Float)
    unit = Column(String(20))
    reference_range_low = Column(Float)
    reference_range_high = Column(Float)
    collected_at = Column(DateTime, nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    patient = relationship("Patient", back_populates="biomarkers")


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    visit_id = Column(String, ForeignKey("visits.visit_id"), index=True)
    prediction_type = Column(String(50), nullable=False, index=True)
    model_version = Column(String(50))
    prediction_value = Column(Float)
    confidence_score = Column(Float)
    risk_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    meta_data = Column(JSON)

    # Relationships
    patient = relationship("Patient", back_populates="predictions")


class TwinState(Base):
    __tablename__ = "twin_states"

    twin_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    visit_id = Column(String, ForeignKey("visits.visit_id"), index=True)
    organ_risks = Column(JSON)
    tumour_volume = Column(Float)
    health_score = Column(Float)
    risk_score = Column(Float)
    simulation_months = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="twin_states")
    visit = relationship("Visit", back_populates="twin_states")
    simulation_results = relationship("SimulationResult", back_populates="twin_state")


class SimulationResult(Base):
    __tablename__ = "simulation_results"

    simulation_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    twin_id = Column(String, ForeignKey("twin_states.twin_id"), index=True)
    treatment = Column(String(100))
    duration_months = Column(Integer)
    outcome_data = Column(JSON)
    intervention_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    twin_state = relationship("TwinState", back_populates="simulation_results")


class Treatment(Base):
    __tablename__ = "treatments"

    treatment_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    visit_id = Column(String, ForeignKey("visits.visit_id"), index=True)
    treatment_name = Column(String(100), nullable=False)
    recommended_by_ai = Column(Boolean, default=False)
    status = Column(String(50), default="scheduled", index=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    dosage = Column(Text)
    side_effects = Column(Text)
    efficacy_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="treatments")


class ClinicalReport(Base):
    __tablename__ = "clinical_reports"

    report_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    visit_id = Column(String, ForeignKey("visits.visit_id"), index=True)
    report_type = Column(String(50), nullable=False, index=True)
    content = Column(JSON)
    format = Column(String(20), default='json')
    generated_at = Column(DateTime, default=datetime.utcnow, index=True)
    file_path = Column(String(500))
    storage_location = Column(String(100))
    exported_formats = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    patient = relationship("Patient", back_populates="clinical_reports")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    log_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, ForeignKey("users.id"), index=True)
    action = Column(String(255), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(String(255), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    details = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(Text)

    # Relationships
    reviewer = relationship("User", back_populates="created_reviews")


class WorkflowState(Base):
    __tablename__ = "workflow_states"

    state_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    current_step = Column(String(50), nullable=False, index=True)
    step_data = Column(JSON)
    completed_steps = Column(JSON, default=list)
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(50), default="in_progress", index=True)

    # Relationships
    patient = relationship("Patient", back_populates="workflow_states")
    visit = relationship("Visit", back_populates="workflow_states")
    user = relationship("User", foreign_keys=[user_id])