export type UserRole = 'Administrator' | 'Radiologist' | 'Oncologist' | 'Researcher' | 'Clinician' | 'Demo User';

export type User = {
  id: string;
  username: string;
  email: string;
  full_name: string;
  role: UserRole;
  permissions: string[];
};

export type Sex = '' | 'Male' | 'Female' | 'Other';

export type SmokingStatus = 'Never' | 'Former' | 'Current';
export type AlcoholStatus = 'None / Minimal' | 'Moderate (<14/week)' | 'Heavy (>14/week)';
export type DiabetesStatus = 'None' | 'New-onset (<2 years)' | 'Long-standing (>2 years)' | 'Type 1' | 'Type 2' | 'Prediabetes';
export type MutationStatus = 'Mutated' | 'Wildtype' | 'Altered' | 'Intact' | 'Unknown';

export type OrganName =
  | 'Pancreas'
  | 'Liver'
  | 'Bile Duct'
  | 'Duodenum'
  | 'Stomach'
  | 'Spleen'
  | 'Lungs'
  | 'Lymph Nodes'
  | 'Heart'
  | 'Kidneys';

export type OrganHealth = {
  organ: OrganName;
  healthScore: number;
  riskScore: number;
  tumorBurden: number;
  inflammation: number;
  perfusion: number;
  fibrosis: number;
  necrosis: number;
  metastasisRisk: number;
  obstruction: number;
  functionScore: number;
  color: string;
  status: 'healthy' | 'mild' | 'moderate' | 'severe' | 'critical';
  affected: boolean;
  lesions: Lesion[];
  biomarkers: Record<string, number>;
};

export type Lesion = {
  id: string;
  position: [number, number, number];
  size: number;
  type: 'primary' | 'metastasis' | 'inflammation';
  growthRate: number;
  vascularity: number;
};

export type PatientState = {
  patientId: string;
  name: string;
  dateOfBirth: string;
  sex: Sex;
  age: number;
  diagnosis: string;
  symptoms: string;
  ecog: string;
  ca199: number;
  cea: number;
  bilirubin: number;
  albumin: number;
  amylase: number;
  weightLoss: number;
  bmi: number;
  smoking: SmokingStatus;
  alcohol: AlcoholStatus;
  diabetes: DiabetesStatus;
  familyHistory: boolean;
  pancreatitis: boolean;
  geneticMutation: boolean;
  kras: MutationStatus;
  tp53: MutationStatus;
  smad4: MutationStatus;
  erbb2: MutationStatus;
  imagingFiles: File[];
  imagingStudy: string;
  imagingStatus: string;
  imagingStudyId?: string;
  imagingMeshUrl?: string;
  uploadedPayload?: any;
  imagingSegmentation?: {
    tumour_size_cm: number;
    organ_masks: Record<string, boolean>;
    quality_score: number;
    segmentation_summary: string;
  };
};

export type PredictionResult = {
  probability: number;
  stage: string;
  stageColour: string;
  stageDetail: string;
  risk: number;
  survival: number;
  survivalData: Record<string, unknown>;
  treatmentData: Record<string, unknown>;
  bestTreatment: string;
  shapValues: Record<string, number>;
  organRisks: Record<string, [number, string, string]>;
  biomarkerTimeline: Record<string, unknown>[];
  confidence: number;
  uncertainty: number;
  modelVersion: string;
};

export type TwinSnapshot = {
  id: string;
  timestamp: string;
  month: number;
  patient: PatientState;
  organs: Record<OrganName, OrganHealth>;
  prediction: PredictionResult | null;
  treatment?: string;
  notes?: string;
};

export type TimelineEvent = {
  id: string;
  type: 'visit' | 'lab' | 'imaging' | 'prediction' | 'treatment' | 'surgery' | 'progression' | 'twin_snapshot';
  date: string;
  title: string;
  description: string;
  data?: Record<string, unknown>;
  icon: string;
  color: string;
};

export type TreatmentOption = {
  id: string;
  name: string;
  description: string;
  responseProbability: number;
  sideEffects: string[];
  complications: string[];
  survivalBenefit: number;
  tumorShrinkage: number;
  organRecovery: number;
  contraindications: string[];
  bestFor: string[];
};

export type SimulationResult = {
  month: number;
  treatment: string;
  tumorVolume: number;
  tumorVolumeChange: number;
  organHealth: Record<OrganName, Partial<OrganHealth>>;
  biomarkers: Record<string, number>;
  survivalProbability: number;
  responseRate: number;
  notes: string;
};

export type XAIExplanation = {
  feature: string;
  shapValue: number;
  contribution: 'positive' | 'negative';
  magnitude: number;
  description: string;
  clinicalSignificance: string;
  counterfactual: {
    feature: string;
    currentValue: number;
    thresholdValue: number;
    impact: number;
  };
};

export type ClinicalFinding = {
  id: string;
  category: 'detection' | 'staging' | 'survival' | 'treatment' | 'risk' | 'prognosis';
  title: string;
  description: string;
  evidence: string;
  confidence: number;
  recommendation: string;
  relatedFeatures: string[];
};

export type StepId =
  | 'registry'
  | 'clinical'
  | 'labs'
  | 'imaging'
  | 'genomics'
  | 'validation'
  | 'analysis'
  | 'twin'
  | 'progression'
  | 'simulation'
  | 'comparison'
  | 'report'
  | 'export';

export type WorkflowState = {
  workflowId: string;
  patientId: string;
  currentStep: StepId;
  completedSteps: StepId[];
  progressPct: number;
  isComplete: boolean;
  status: string;
};

export type ValidationItem = {
  label: string;
  valid: boolean;
  message: string;
};

export type ThemeMode = 'dark' | 'light' | 'clinical';

export type RenderMode = 'realistic' | 'xray' | 'wireframe' | 'heatmap' | 'segmentation' | 'ct';

export type SimulationMode = 'progression' | 'treatment' | 'whatif';

export type CameraPosition = {
  position: [number, number, number];
  target: [number, number, number];
  fov: number;
};

export type SyntheticOrganMesh = {
  organ: OrganName;
  vertices: Float32Array;
  normals: Float32Array;
  indices: Uint16Array;
  color: string;
  opacity: number;
  emissive: string;
  emissiveIntensity: number;
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
  tumorMesh?: {
    vertices: Float32Array;
    normals: Float32Array;
    indices: Uint16Array;
    color: string;
    emissive: string;
    emissiveIntensity: number;
  };
};

export type SyntheticModel = {
  generated: boolean;
  generating: boolean;
  progress: number;
  timestamp: string;
  meshes: SyntheticOrganMesh[];
  tumorPresent: boolean;
  tumorSize: number;
  tumorLocation: string;
};
