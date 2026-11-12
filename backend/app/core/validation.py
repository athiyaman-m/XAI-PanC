"""
JSON Schema definitions for API contracts in ExAI-PanC v2.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class APIResponse(BaseModel):
    """Base API response model"""
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")


class PaginatedResponse(BaseModel):
    """Paginated response model"""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")


class ValidationErrorDetail(BaseModel):
    """Validation error detail model"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    code: str = Field(..., description="Error code")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response model with field details"""
    errors: List[ValidationErrorDetail] = Field(..., description="List of validation errors")


class PatientBase(BaseModel):
    """Base patient model"""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")
    mrn: str = Field(..., min_length=1, max_length=50, description="Medical Record Number")
    date_of_birth: Optional[datetime] = Field(None, description="Date of birth")
    sex: Optional[str] = Field(None, description="Sex (Male/Female/Other)")
    blood_type: Optional[str] = Field(None, description="Blood type")
    emergency_contact: Optional[str] = Field(None, description="Emergency contact")

    @validator('sex')
    def validate_sex(cls, v):
        if v and v.lower() not in ['male', 'female', 'other']:
            raise ValueError('Sex must be Male, Female, or Other')
        return v


class ClinicalAssessmentBase(BaseModel):
    """Base clinical assessment model"""
    symptoms: Optional[List[str]] = Field(None, description="List of symptoms")
    physical_exam: Optional[str] = Field(None, description="Physical examination findings")
    diagnosis: Optional[str] = Field(None, description="Primary diagnosis")
    treatment_plan: Optional[str] = Field(None, description="Treatment plan")
    notes: Optional[str] = Field(None, description="Additional notes")


class LabResultBase(BaseModel):
    """Base lab result model"""
    test_name: str = Field(..., min_length=1, max_length=100, description="Test name")
    value: float = Field(..., description="Test value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    reference_range_low: Optional[float] = Field(None, description="Lower reference range")
    reference_range_high: Optional[float] = Field(None, description="Upper reference range")
    collected_at: Optional[datetime] = Field(None, description="Collection timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

    @validator('value')
    def validate_value(cls, v):
        if v is not None and v < 0:
            raise ValueError('Value must be non-negative')
        return v


class ImagingStudyBase(BaseModel):
    """Base imaging study model"""
    study_type: str = Field(..., min_length=1, max_length=50, description="Study type")
    modality: Optional[str] = Field(None, description="Modality (CT/MRI/PET/etc.)")
    study_date: datetime = Field(..., description="Study date")
    accession_number: Optional[str] = Field(None, description="Accession number")
    study_instance_uid: Optional[str] = Field(None, description="Study instance UID")
    status: str = Field("pending", description="Study status")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GeneExpressionBase(BaseModel):
    """Base gene expression model"""
    sample_id: str = Field(..., min_length=1, max_length=100, description="Sample ID")
    gene_symbol: str = Field(..., min_length=1, max_length=100, description="Gene symbol")
    expression_value: Optional[float] = Field(None, description="Expression value")
    mutation_status: Optional[str] = Field(None, description="Mutation status")


class PredictionBase(BaseModel):
    """Base prediction model"""
    prediction_type: str = Field(..., min_length=1, max_length=50, description="Prediction type")
    model_version: Optional[str] = Field(None, description="Model version")
    prediction_value: float = Field(..., description="Prediction value")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator('prediction_value')
    def validate_prediction_value(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Prediction value must be between 0 and 100')
        return v


class TwinStateBase(BaseModel):
    """Base digital twin state model"""
    organ_risks: Dict[str, Any] = Field(..., description="Organ risk scores")
    tumour_volume: Optional[float] = Field(None, description="Tumor volume in cm³")
    health_score: float = Field(..., ge=0, le=100, description="Health score (0-100)")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100)")
    simulation_months: Optional[int] = Field(None, description="Simulation months")

    @validator('tumour_volume')
    def validate_tumour_volume(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Tumor volume must be positive')
        return v


class SimulationResultBase(BaseModel):
    """Base simulation result model"""
    treatment: str = Field(..., min_length=1, max_length=100, description="Treatment name")
    duration_months: int = Field(..., description="Duration in months")
    outcome_data: Dict[str, Any] = Field(..., description="Simulation outcomes")
    intervention_data: Dict[str, Any] = Field(..., description="Intervention data")

    @validator('duration_months')
    def validate_duration_months(cls, v):
        if v <= 0 or v > 24:
            raise ValueError('Duration must be between 1 and 24 months')
        return v


class TreatmentBase(BaseModel):
    """Base treatment model"""
    treatment_name: str = Field(..., min_length=1, max_length=100, description="Treatment name")
    recommended_by_ai: bool = Field(False, description="Whether recommended by AI")
    status: str = Field("scheduled", description="Treatment status")
    start_date: Optional[datetime] = Field(None, description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")
    dosage: Optional[str] = Field(None, description="Dosage information")
    side_effects: Optional[str] = Field(None, description="Side effects")
    efficacy_score: Optional[float] = Field(None, ge=0, le=100, description="Efficacy score (0-100)")


class ClinicalReportBase(BaseModel):
    """Base clinical report model"""
    report_type: str = Field(..., min_length=1, max_length=50, description="Report type")
    content: Dict[str, Any] = Field(..., description="Report content")
    format: str = Field("json", description="Report format")
    file_path: Optional[str] = Field(None, description="File path (if applicable)")
    storage_location: Optional[str] = Field(None, description="Storage location")
    exported_formats: Optional[List[str]] = Field(None, description="Exported formats")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkflowStateBase(BaseModel):
    """Base workflow state model"""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")
    user_id: str = Field(..., min_length=1, max_length=255, description="User ID")
    current_step: str = Field(..., min_length=1, max_length=50, description="Current workflow step")
    step_data: Dict[str, Any] = Field(..., description="Step-specific data")
    completed_steps: List[str] = Field(default_factory=list, description="Completed steps")
    status: str = Field("in_progress", description="Workflow status")


class BiomarkerBase(BaseModel):
    """Base biomarker model"""
    biomarker_name: str = Field(..., min_length=1, max_length=50, description="Biomarker name")
    value: float = Field(..., description="Biomarker value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    reference_range_low: Optional[float] = Field(None, description="Lower reference range")
    reference_range_high: Optional[float] = Field(None, description="Upper reference range")
    collected_at: datetime = Field(..., description="Collection timestamp")

    @validator('value')
    def validate_value(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CreatePatientRequest(PatientBase):
    """Request model for creating a patient"""
    user_id: str = Field(..., min_length=1, max_length=255, description="User ID")
    display_name: Optional[str] = Field(None, description="Display name for patient")

    @validator('user_id')
    def validate_user_id(cls, v):
        if len(v) < 36 or len(v) > 255:
            raise ValueError('User ID must be between 36 and 255 characters')
        return v


class UpdatePatientRequest(BaseModel):
    """Request model for updating a patient"""
    medical_history: Optional[str] = Field(None, description="Medical history")
    risk_factors: Optional[List[str]] = Field(None, description="Risk factors")
    emergency_contact: Optional[str] = Field(None, description="Emergency contact")


class CreateVisitRequest(BaseModel):
    """Request model for creating a visit"""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")
    visit_date: datetime = Field(..., description="Visit date")
    visit_type: Optional[str] = Field(None, description="Visit type")
    admitting_doctor: Optional[str] = Field(None, description="Admitting doctor")
    primary_diagnosis: Optional[str] = Field(None, description="Primary diagnosis")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CreateClinicalAssessmentRequest(ClinicalAssessmentBase):
    """Request model for creating a clinical assessment"""
    visit_id: str = Field(..., min_length=1, max_length=255, description="Visit ID")
    assessment_date: Optional[datetime] = Field(None, description="Assessment date")


class CreateLabResultRequest(LabResultBase):
    """Request model for creating a lab result"""
    visit_id: str = Field(..., min_length=1, max_length=255, description="Visit ID")
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")


class CreateImagingStudyRequest(ImagingStudyBase):
    """Request model for creating an imaging study"""
    visit_id: str = Field(..., min_length=1, max_length=255, description="Visit ID")


class CreateGeneExpressionRequest(GeneExpressionBase):
    """Request model for creating gene expression data"""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")
    visit_id: Optional[str] = Field(None, min_length=1, max_length=255, description="Visit ID")


class CreatePredictionRequest(PredictionBase):
    """Request model for creating a prediction"""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")
    visit_id: Optional[str] = Field(None, min_length=1, max_length=255, description="Visit ID")


class CreateTwinStateRequest(TwinStateBase):
    """Request model for creating a digital twin state"""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")
    visit_id: Optional[str] = Field(None, min_length=1, max_length=255, description="Visit ID")


class CreateSimulationRequest(SimulationResultBase):
    """Request model for creating a simulation"""
    twin_id: str = Field(..., min_length=1, max_length=255, description="Twin ID")


class CreateTreatmentRequest(TreatmentBase):
    """Request model for creating a treatment"""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")
    visit_id: Optional[str] = Field(None, min_length=1, max_length=255, description="Visit ID")


class CreateReportRequest(ClinicalReportBase):
    """Request model for creating a report"""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")
    visit_id: Optional[str] = Field(None, min_length=1, max_length=255, description="Visit ID")


class CreateBiomarkerRequest(BiomarkerBase):
    """Request model for creating a biomarker"""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")
    visit_id: Optional[str] = Field(None, min_length=1, max_length=255, description="Visit ID")


class WorkflowStepRequest(BaseModel):
    """Request model for workflow step completion"""
    step: str = Field(..., min_length=1, max_length=50, description="Workflow step to complete")
    step_data: Dict[str, Any] = Field(..., description="Step-specific data")
    comments: Optional[str] = Field(None, description="Additional comments")


class TreatmentComparisonRequest(BaseModel):
    """Request model for treatment comparison"""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")
    treatment_options: List[str] = Field(..., description="List of treatments to compare")
    simulation_params: Optional[Dict[str, Any]] = Field(None, description="Simulation parameters")


class RunAIRequest(BaseModel):
    """Request model for running AI analysis"""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")
    include_predefined_models: bool = Field(True, description="Whether to include predefined models")
    confidence_threshold: float = Field(0.5, ge=0, le=1, description="Confidence threshold")


class ValidationResponse(BaseModel):
    """Response model for validation"""
    is_valid: bool = Field(..., description="Whether the data is valid")
    errors: List[ValidationErrorDetail] = Field(default_factory=list, description="List of validation errors")


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status"""
    patient_id: str = Field(..., description="Patient ID")
    current_step: str = Field(..., description="Current workflow step")
    completed_steps: List[str] = Field(..., description="Completed steps")
    next_step: str = Field(..., description="Next workflow step")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion date")
    is_workflow_complete: bool = Field(..., description="Whether workflow is complete")


class ExportRequest(BaseModel):
    """Request model for exporting reports"""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient ID")
    report_types: List[str] = Field(..., description="Types of reports to export")
    format: str = Field("pdf", description="Export format")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range for export")


class ExportResponse(BaseModel):
    """Response model for export"""
    export_id: str = Field(..., description="Export ID")
    status: str = Field(..., description="Export status")
    file_path: Optional[str] = Field(None, description="File path (if ready)")
    download_url: Optional[str] = Field(None, description="Download URL (if available)")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")