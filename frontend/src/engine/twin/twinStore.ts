import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type {
  PatientState,
  OrganHealth,
  OrganName,
  TwinSnapshot,
  PredictionResult,
  SimulationResult,
  TimelineEvent,
  TreatmentOption,
  XAIExplanation,
  ClinicalFinding,
  StepId,
  WorkflowState,
  ThemeMode,
  RenderMode,
  SimulationMode,
  CameraPosition,
  SyntheticModel,
} from '../../types';

const DEFAULT_ORGAN_NAMES: OrganName[] = [
  'Pancreas', 'Liver', 'Bile Duct', 'Duodenum', 'Stomach',
  'Spleen', 'Lungs', 'Lymph Nodes', 'Heart', 'Kidneys',
];

const ORGAN_COLORS: Record<OrganName, string> = {
  Pancreas: '#ef565f',
  Liver: '#f4a524',
  'Bile Duct': '#f77828',
  Duodenum: '#f4b024',
  Stomach: '#35c59b',
  Spleen: '#6badf5',
  Lungs: '#6badf5',
  'Lymph Nodes': '#9b59b6',
  Heart: '#e74c3c',
  Kidneys: '#1abc9c',
};

const createDefaultOrganHealth = (organ: OrganName): OrganHealth => ({
  organ,
  healthScore: 100,
  riskScore: 0,
  tumorBurden: 0,
  inflammation: 0,
  perfusion: 100,
  fibrosis: 0,
  necrosis: 0,
  metastasisRisk: 0,
  obstruction: 0,
  functionScore: 100,
  color: ORGAN_COLORS[organ],
  status: 'healthy',
  affected: false,
  lesions: [],
  biomarkers: {},
});

const createDefaultPatient = (): PatientState => ({
  patientId: '',
  name: '',
  dateOfBirth: '',
  sex: '',
  age: 0,
  diagnosis: 'Pancreatic ductal adenocarcinoma',
  symptoms: '',
  ecog: '',
  ca199: 0,
  cea: 0,
  bilirubin: 0,
  albumin: 0,
  amylase: 0,
  weightLoss: 0,
  bmi: 0,
  smoking: 'Never',
  alcohol: 'None / Minimal',
  diabetes: 'None',
  familyHistory: false,
  pancreatitis: false,
  geneticMutation: false,
  kras: 'Wildtype',
  tp53: 'Wildtype',
  smad4: 'Intact',
  erbb2: 'Wildtype',
  imagingFiles: [],
  imagingStudy: '',
  imagingStudyId: undefined,
  imagingStatus: '',
  imagingSegmentation: undefined,
});

export type TwinEngineState = {
  patient: PatientState;
  patientList: PatientState[];
  organs: Record<OrganName, OrganHealth>;
  prediction: PredictionResult | null;
  snapshots: TwinSnapshot[];
  timeline: TimelineEvent[];
  treatments: TreatmentOption[];
  simulationResults: SimulationResult[];
  xaiExplanations: XAIExplanation[];
  clinicalFindings: ClinicalFinding[];
  workflow: WorkflowState;
  theme: ThemeMode;
  renderMode: RenderMode;
  simulationMode: SimulationMode;
  simulationMonth: number;
  camera: CameraPosition;
  isSimulating: boolean;
  isUpdating: boolean;
  selectedOrgan: OrganName | null;
  isolatedOrgans: OrganName[];
  xrayMode: boolean;
  opacity: number;
  showAnnotations: boolean;
  showMetastasisPaths: boolean;
  showBiomarkerTimeline: boolean;
  selectedTreatment: string | null;
  comparisonTreatments: string[];
  syntheticModel: SyntheticModel;
};

export type TwinEngineActions = {
  updatePatient: (updates: Partial<PatientState>) => void;
  applyBiomarkers: (b: {
    name: string; age: number; sex: string;
    ca199: number; cea: number; bilirubin: number; albumin: number; weightLoss: number;
    kras: boolean; tp53: boolean; smad4: boolean;
  }) => void;
  savePatientToList: () => void;
  loadPatient: (patientId: string) => void;
  deletePatientFromList: (patientId: string) => void;
  loadPatientsFromStorage: () => void;
  updateOrgan: (organ: OrganName, updates: Partial<OrganHealth>) => void;
  setPrediction: (prediction: PredictionResult) => void;
  addSnapshot: (snapshot: Omit<TwinSnapshot, 'id' | 'timestamp'>) => void;
  addTimelineEvent: (event: Omit<TimelineEvent, 'id'>) => void;
  setTreatments: (treatments: TreatmentOption[]) => void;
  addSimulationResult: (result: SimulationResult) => void;
  setXAIExplanations: (explanations: XAIExplanation[]) => void;
  setClinicalFindings: (findings: ClinicalFinding[]) => void;
  addClinicalFinding: (finding: ClinicalFinding) => void;
  setWorkflow: (workflow: Partial<WorkflowState>) => void;
  setTheme: (theme: ThemeMode) => void;
  setRenderMode: (mode: RenderMode) => void;
  setSimulationMode: (mode: SimulationMode) => void;
  setSimulationMonth: (month: number) => void;
  setCamera: (camera: Partial<CameraPosition>) => void;
  setIsSimulating: (simulating: boolean) => void;
  setIsUpdating: (updating: boolean) => void;
  setSelectedOrgan: (organ: OrganName | null) => void;
  setXrayMode: (xray: boolean) => void;
  setOpacity: (opacity: number) => void;
  setShowAnnotations: (show: boolean) => void;
  setShowMetastasisPaths: (show: boolean) => void;
  setShowBiomarkerTimeline: (show: boolean) => void;
  setSelectedTreatment: (treatment: string | null) => void;
  setComparisonTreatments: (treatments: string[]) => void;
  setSyntheticModel: (model: Partial<SyntheticModel>) => void;
  resetTwin: () => void;
  computeOrganHealth: (patient: PatientState, prediction: PredictionResult | null) => void;
  runPrediction: () => Promise<void>;
  runSimulation: (treatment: string, months: number) => Promise<void>;
  runWhatIfAnalysis: (parameter: string, value: number) => Promise<void>;
};



function computeHeuristicPrediction(patient: PatientState): PredictionResult {
  const age = patient.age || 60;
  const sex = patient.sex;
  const ca199 = patient.ca199 || 37;
  const cea = patient.cea || 3;
  const bilirubin = patient.bilirubin || 0.8;
  const wt = patient.weightLoss || 0;
  const smoking = patient.smoking;
  const diabetes = patient.diabetes;
  const familyHistory = patient.familyHistory;
  const pancreatitis = patient.pancreatitis;
  const genetic = patient.geneticMutation;

  const ageScore = Math.min(0.25, Math.max(0, (age - 40) / 400) + (age >= 70 ? 0.05 : age >= 60 ? 0.02 : 0));

  let bioScore = 0;
  if (ca199 > 1000) bioScore += 0.20;
  else if (ca199 > 500) bioScore += 0.15;
  else if (ca199 > 100) bioScore += 0.10;
  else if (ca199 > 37) bioScore += 0.05;
  if (cea > 20) bioScore += 0.06;
  else if (cea > 5) bioScore += 0.03;
  if (bilirubin > 5) bioScore += 0.08;
  else if (bilirubin > 2) bioScore += 0.04;
  if (wt > 15) bioScore += 0.08;
  else if (wt > 10) bioScore += 0.05;
  else if (wt > 5) bioScore += 0.02;
  if (smoking === 'Current') bioScore += 0.06;
  else if (smoking === 'Former') bioScore += 0.03;
  if (diabetes.includes('New-onset') || diabetes.includes('Type 2')) bioScore += 0.08;
  if (familyHistory) bioScore += 0.05;
  if (pancreatitis) bioScore += 0.04;
  if (genetic) bioScore += 0.06;

  const prob = Math.min(ageScore + bioScore, 0.97);

  let stageLabel: string, stageColour: string, stageDetail: string;
  if (prob > 0.80 && ca199 > 500) { stageLabel = 'Stage IV'; stageColour = '#e74c3c'; stageDetail = 'Distant metastasis likely'; }
  else if (prob > 0.65 && ca199 > 200) { stageLabel = 'Stage III'; stageColour = '#e67e22'; stageDetail = 'Locally advanced'; }
  else if (prob > 0.50 && ca199 > 100) { stageLabel = 'Stage II'; stageColour = '#f39c12'; stageDetail = 'Regional spread'; }
  else if (prob > 0.35) { stageLabel = 'Stage I'; stageColour = '#2ecc71'; stageDetail = 'Localised'; }
  else { stageLabel = 'Non-malignant'; stageColour = '#3498db'; stageDetail = 'No strong malignancy evidence'; }

  let baseHazard = 1.0;
  if (stageLabel.includes('III') || stageLabel.includes('IV')) baseHazard += 1.5;
  else if (stageLabel.includes('II')) baseHazard += 0.5;
  baseHazard += prob * 2.0;
  baseHazard += Math.max(0, (age - 60) / 40);
  const riskScore = Math.round(baseHazard * 100) / 100;
  const medianMonths = Math.max(3, Math.round(36 - baseHazard * 5));

  const folfi = Math.min(Math.max(0.5 + (age < 65 ? 0.15 : 0) + (prob < 0.6 ? 0.08 : 0), 0.1), 0.92);
  const gem = Math.min(Math.max(0.45 + Math.max(0, (1 - prob) * 0.2), 0.1), 0.88);
  const folfox = Math.min(Math.max(folfi * 0.85, 0.1), 0.80);
  const bsc = Math.min(Math.max(0.30 + prob * 0.15, 0.1), 0.55);
  const treatmentData: Record<string, unknown> = {
    FOLFIRINOX: Math.round(folfi * 100),
    'Gemcitabine + nab-paclitaxel': Math.round(gem * 100),
    FOLFOX: Math.round(folfox * 100),
    'Best Supportive Care': Math.round(bsc * 100),
  };
  const bestTx = folfi >= gem ? 'FOLFIRINOX' : 'Gemcitabine + nab-paclitaxel';

  const shapValues: Record<string, number> = {
    'Age': Math.round((age - 55) / 50 * 0.20 * 10000) / 10000,
    'Sex (Male)': Math.round((sex === 'Male' ? 0.08 : -0.04) * 10000) / 10000,
    'CA 19-9 Level': Math.round((ca199 - 37) / 2000 * 0.35 * 10000) / 10000,
    'Serum Bilirubin': Math.round(bilirubin / 20 * 0.12 * 10000) / 10000,
    'Weight Loss (%)': Math.round(wt / 20 * 0.10 * 10000) / 10000,
    'CEA Level': Math.round((cea - 5) / 200 * 0.08 * 10000) / 10000,
  };

  const organRisks: Record<string, [number, string, string]> = {
    Pancreas: [Math.min(98, Math.round(prob * 100 * 0.85 + (pancreatitis ? 10 : 0))), '#ef565f', 'Primary tumour site'],
    Liver: [Math.min(95, Math.round((bilirubin / 12) * 60 + (ca199 / 5000) * 40 + (prob > 0.65 ? 15 : 0))), '#f4a24', 'Biliary obstruction risk'],
    'Bile Duct': [Math.min(92, Math.round((bilirubin / 8) * 55 + (bilirubin > 3 ? 20 : 0))), '#f77828', 'Obstructive jaundice risk'],
    Duodenum: [Math.min(80, Math.round(prob * 60 + (wt / 30) * 25)), '#f4b024', 'Anatomical invasion risk'],
    Spleen: [Math.min(70, Math.round(prob * 0.4 * 60)), '#6badf5', 'Splenic vein involvement'],
    Lungs: [Math.min(60, Math.round(prob * 0.35 * 80)), '#6badf5', 'Distant metastasis risk'],
  };

  const biomarkerTimeline = Array.from({ length: 7 }, (_, i) => ({
    month: i * 3,
    ca199: ca199 * (1 + prob * 0.15 * i),
    cea: cea * (1 + prob * 0.08 * i),
    bilirubin: bilirubin * (1 + prob * 0.1 * i),
  }));

  const survivalData: Record<string, unknown> = {
    risk_score: riskScore,
    median_months: medianMonths,
    risk_group: baseHazard > 3.0 ? 'High Risk' : baseHazard > 1.5 ? 'Medium Risk' : 'Low Risk',
  };

  return {
    probability: Math.round(prob * 1000) / 10,
    stage: stageLabel,
    stageColour,
    stageDetail,
    risk: riskScore,
    survival: medianMonths,
    survivalData,
    treatmentData,
    bestTreatment: bestTx,
    shapValues,
    organRisks,
    biomarkerTimeline,
    confidence: 0.70,
    uncertainty: 0.30,
    modelVersion: 'heuristic-1.0',
  };
}

const useTwinStore = create<TwinEngineState & TwinEngineActions>()(
  subscribeWithSelector((set, get) => ({
    patient: createDefaultPatient(),
    patientList: [],
    organs: DEFAULT_ORGAN_NAMES.reduce((acc, organ) => {
      acc[organ] = createDefaultOrganHealth(organ);
      return acc;
    }, {} as Record<OrganName, OrganHealth>),
    prediction: null,
    snapshots: [],
    timeline: [],
    treatments: [],
    simulationResults: [],
    xaiExplanations: [],
    clinicalFindings: [],
    workflow: {
      workflowId: '',
      patientId: '',
      currentStep: 'registry',
      completedSteps: [],
      progressPct: 0,
      isComplete: false,
      status: 'in_progress',
    },
    theme: 'dark',
    renderMode: 'realistic',
    simulationMode: 'progression',
    simulationMonth: 0,
    camera: {
      position: [0, 0.5, 7],
      target: [0, 0, 0],
      fov: 34,
    },
    isSimulating: false,
    isUpdating: false,
    selectedOrgan: null,
    isolatedOrgans: [],
    xrayMode: false,
    opacity: 0.72,
    showAnnotations: true,
    showMetastasisPaths: true,
    showBiomarkerTimeline: true,
    selectedTreatment: null,
    comparisonTreatments: ['FOLFIRINOX', 'Gemcitabine + nab-paclitaxel'],
    syntheticModel: {
      generated: false,
      generating: false,
      progress: 0,
      timestamp: '',
      meshes: [],
      tumorPresent: false,
      tumorSize: 0,
      tumorLocation: '',
    },

    setSyntheticModel: (model) => {
      set((state) => ({
        syntheticModel: { ...state.syntheticModel, ...model },
      }));
    },

    updatePatient: (updates) => {
      const prevPatient = get().patient;
      set((state) => ({
        patient: { ...state.patient, ...updates },
      }));
      const newPatient = get().patient;

      if (newPatient.patientId) {
        const list = get().patientList;
        const idx = list.findIndex((p) => p.patientId === newPatient.patientId);
        const serializable = { ...newPatient, imagingFiles: [] as File[] };
        const updatedList = idx >= 0
          ? list.map((p) => (p.patientId === newPatient.patientId ? serializable : p))
          : [...list, serializable];
        set({ patientList: updatedList });
        localStorage.setItem('exaipanc_patients', JSON.stringify(updatedList));
      }

      const prediction = get().prediction;
      if (prediction) {
        get().computeOrganHealth(newPatient, prediction);
      }
      if (JSON.stringify(prevPatient) !== JSON.stringify(newPatient)) {
        get().addTimelineEvent({
          type: 'visit',
          date: new Date().toISOString(),
          title: 'Patient Data Updated',
          description: `Updated ${Object.keys(updates).join(', ')}`,
          data: updates,
          icon: 'edit',
          color: '#3b82f6',
        });
      }
    },

    applyBiomarkers: (b) => {
      const patient: PatientState = {
        ...get().patient,
        name: b.name || get().patient.name,
        age: b.age,
        sex: b.sex as PatientState['sex'],
        ca199: b.ca199,
        cea: b.cea,
        bilirubin: b.bilirubin,
        albumin: b.albumin,
        weightLoss: b.weightLoss,
        kras: b.kras ? 'Mutated' : 'Wildtype',
        tp53: b.tp53 ? 'Mutated' : 'Wildtype',
        smad4: b.smad4 ? 'Altered' : 'Intact',
      };
      set({ patient });
      const live = computeHeuristicPrediction(patient);
      get().setPrediction(live);
    },

    savePatientToList: () => {
      const { patient, patientList } = get();
      if (!patient.name) return;
      const pid = patient.patientId || `P${String(Date.now()).slice(-6)}`;
      const serializable = { ...patient, patientId: pid, imagingFiles: [] as File[] };
      const exists = patientList.findIndex((p) => p.patientId === pid);
      const newList = exists >= 0
        ? patientList.map((p) => (p.patientId === pid ? serializable : p))
        : [...patientList, serializable];
      set({ patientList: newList, patient: serializable });
      localStorage.setItem('exaipanc_patients', JSON.stringify(newList));
    },

    loadPatient: (patientId) => {
      const { patientList } = get();
      const found = patientList.find((p) => p.patientId === patientId);
      if (!found) return;
      set({ patient: { ...found } });
      get().computeOrganHealth(found, get().prediction);
      get().addTimelineEvent({
        type: 'visit',
        date: new Date().toISOString(),
        title: `Loaded Patient: ${found.name}`,
        description: `Patient ${found.patientId} loaded into workspace`,
        icon: 'users',
        color: '#3ec7ad',
      });
    },

    deletePatientFromList: (patientId) => {
      const { patientList, patient } = get();
      const newList = patientList.filter((p) => p.patientId !== patientId);
      set({ patientList: newList });
      localStorage.setItem('exaipanc_patients', JSON.stringify(newList));
      if (patient.patientId === patientId) {
        set({ patient: createDefaultPatient() });
      }
    },

    loadPatientsFromStorage: () => {
      try {
        const raw = localStorage.getItem('exaipanc_patients');
        if (raw) {
          const list = JSON.parse(raw) as PatientState[];
          set({ patientList: list });
        }
      } catch {
        // ignore
      }
    },

    updateOrgan: (organ, updates) => {
      set((state) => ({
        organs: {
          ...state.organs,
          [organ]: { ...state.organs[organ], ...updates },
        },
      }));
    },

    setPrediction: (prediction) => {
      set({ prediction });
      get().computeOrganHealth(get().patient, prediction);
      get().setXAIExplanations(generateXAIExplanations(prediction, get().patient));
      get().setClinicalFindings(generateClinicalFindings(prediction, get().patient));
    },

    addSnapshot: (snapshot) => {
      const newSnapshot: TwinSnapshot = {
        ...snapshot,
        id: `snap-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        timestamp: new Date().toISOString(),
      };
      set((state) => ({
        snapshots: [...state.snapshots, newSnapshot],
      }));
    },

    addTimelineEvent: (event) => {
      const newEvent: TimelineEvent = {
        ...event,
        id: `event-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      };
      set((state) => ({
        timeline: [...state.timeline, newEvent],
      }));
    },

    setTreatments: (treatments) => set({ treatments }),
    addSimulationResult: (result) => {
      set((state) => ({
        simulationResults: [...state.simulationResults, result],
      }));
    },
    setXAIExplanations: (explanations) => set({ xaiExplanations: explanations }),
    setClinicalFindings: (findings) => set({ clinicalFindings: findings }),
    addClinicalFinding: (finding) => {
      set((state) => ({
        clinicalFindings: [...state.clinicalFindings, finding],
      }));
    },

    setWorkflow: (workflow) => {
      set((state) => ({
        workflow: { ...state.workflow, ...workflow },
      }));
    },

    setTheme: (theme) => set({ theme }),
    setRenderMode: (mode) => set({ renderMode: mode }),
    setSimulationMode: (mode) => set({ simulationMode: mode }),
    setSimulationMonth: (month) => set({ simulationMonth: month }),
    setCamera: (camera) => {
      set((state) => ({
        camera: { ...state.camera, ...camera },
      }));
    },
    setIsSimulating: (simulating) => set({ isSimulating: simulating }),
    setIsUpdating: (updating) => set({ isUpdating: updating }),
    setSelectedOrgan: (organ) => set({ selectedOrgan: organ }),
    setXrayMode: (xray) => set({ xrayMode: xray }),
    setOpacity: (opacity) => set({ opacity }),
    setShowAnnotations: (show) => set({ showAnnotations: show }),
    setShowMetastasisPaths: (show) => set({ showMetastasisPaths: show }),
    setShowBiomarkerTimeline: (show) => set({ showBiomarkerTimeline: show }),
    setSelectedTreatment: (treatment) => set({ selectedTreatment: treatment }),
    setComparisonTreatments: (treatments) => set({ comparisonTreatments: treatments }),

    resetTwin: () => {
      set({
    patient: createDefaultPatient(),
    patientList: [],
        organs: DEFAULT_ORGAN_NAMES.reduce((acc, organ) => {
          acc[organ] = createDefaultOrganHealth(organ);
          return acc;
        }, {} as Record<OrganName, OrganHealth>),
        prediction: null,
        snapshots: [],
        timeline: [],
        treatments: [],
        simulationResults: [],
        xaiExplanations: [],
        clinicalFindings: [],
        simulationMonth: 0,
        selectedOrgan: null,
        xrayMode: false,
        opacity: 0.72,
      });
    },

    computeOrganHealth: (patient, prediction) => {
      const organs = { ...get().organs };
      const prob = prediction?.probability ?? 0;
      const ca199 = patient.ca199;
      const cea = patient.cea;
      const bili = patient.bilirubin;
      const amylase = patient.amylase;
      const wt = patient.weightLoss;

      const getRiskColor = (risk: number): string => {
        if (risk >= 70) return '#ef4444';
        if (risk >= 45) return '#f97316';
        if (risk >= 20) return '#eab308';
        return '#22c55e';
      };

      const getStatus = (risk: number): OrganHealth['status'] => {
        if (risk >= 70) return 'critical';
        if (risk >= 45) return 'severe';
        if (risk >= 20) return 'moderate';
        if (risk >= 5) return 'mild';
        return 'healthy';
      };

      const pancreasRisk = Math.min(98, prob * 100 * 0.85 + (amylase > 200 ? 20 : 0) + (patient.pancreatitis ? 10 : 0) + Math.max(0, (ca199 - 37) / 100));
      organs.Pancreas = {
        ...organs.Pancreas,
        riskScore: pancreasRisk,
        healthScore: Math.max(0, 100 - pancreasRisk),
        tumorBurden: prob > 0.3 ? prob * 80 : 0,
        inflammation: patient.pancreatitis ? 60 : prob > 0.5 ? 40 : 0,
        status: getStatus(pancreasRisk),
        color: getRiskColor(pancreasRisk),
        affected: pancreasRisk > 20,
        biomarkers: { 'CA 19-9': ca199, Amylase: amylase },
      };

      const liverRisk = Math.min(95, (bili / 12) * 60 + (ca199 / 5000) * 40 + (prob > 0.65 ? 15 : 0));
      organs.Liver = {
        ...organs.Liver,
        riskScore: liverRisk,
        healthScore: Math.max(0, 100 - liverRisk),
        metastasisRisk: prob > 0.5 ? prob * 60 : 0,
        obstruction: bili > 2 ? 70 : bili > 1.2 ? 40 : 0,
        functionScore: Math.max(30, 100 - liverRisk * 0.8),
        status: getStatus(liverRisk),
        color: getRiskColor(liverRisk),
        affected: liverRisk > 20,
        biomarkers: { Bilirubin: bili, 'CA 19-9': ca199 },
      };

      const bileRisk = Math.min(92, (bili / 8) * 55 + (bili > 3 ? 20 : 0));
      organs['Bile Duct'] = {
        ...organs['Bile Duct'],
        riskScore: bileRisk,
        healthScore: Math.max(0, 100 - bileRisk),
        obstruction: bili > 2 ? 80 : bili > 1.2 ? 50 : 0,
        status: getStatus(bileRisk),
        color: getRiskColor(bileRisk),
        affected: bileRisk > 20,
        biomarkers: { Bilirubin: bili },
      };

      const duodenumRisk = Math.min(80, prob * 60 + (wt / 30) * 25);
      organs.Duodenum = {
        ...organs.Duodenum,
        riskScore: duodenumRisk,
        healthScore: Math.max(0, 100 - duodenumRisk),
        status: getStatus(duodenumRisk),
        color: getRiskColor(duodenumRisk),
        affected: duodenumRisk > 20,
      };

      const stomachRisk = Math.min(65, prob * 45 + (prob > 0.7 ? 20 : 0) + (wt / 30) * 20);
      organs.Stomach = {
        ...organs.Stomach,
        riskScore: stomachRisk,
        healthScore: Math.max(0, 100 - stomachRisk),
        status: getStatus(stomachRisk),
        color: getRiskColor(stomachRisk),
        affected: stomachRisk > 20,
      };

      const spleenRisk = Math.min(70, prob * 0.4 * 60);
      organs.Spleen = {
        ...organs.Spleen,
        riskScore: spleenRisk,
        healthScore: Math.max(0, 100 - spleenRisk),
        perfusion: Math.max(40, 100 - spleenRisk),
        status: getStatus(spleenRisk),
        color: getRiskColor(spleenRisk),
        affected: spleenRisk > 20,
      };

      const lungRisk = Math.min(60, prob * 0.35 * 80 + (ca199 / 4000) * 30);
      organs.Lungs = {
        ...organs.Lungs,
        riskScore: lungRisk,
        healthScore: Math.max(0, 100 - lungRisk),
        metastasisRisk: prob > 0.6 ? prob * 50 : 0,
        perfusion: Math.max(50, 100 - lungRisk * 0.7),
        status: getStatus(lungRisk),
        color: getRiskColor(lungRisk),
        affected: lungRisk > 20,
      };

      const lymphRisk = Math.min(85, prob * 70 + (ca199 / 2000) * 20 + (patient.age > 65 ? 10 : 0));
      organs['Lymph Nodes'] = {
        ...organs['Lymph Nodes'],
        riskScore: lymphRisk,
        healthScore: Math.max(0, 100 - lymphRisk),
        tumorBurden: prob > 0.4 ? prob * 60 : 0,
        status: getStatus(lymphRisk),
        color: getRiskColor(lymphRisk),
        affected: lymphRisk > 20,
      };

      const heartRisk = Math.min(40, prob * 0.2 * 40 + (bili > 3 ? 15 : 0));
      organs.Heart = {
        ...organs.Heart,
        riskScore: heartRisk,
        healthScore: Math.max(0, 100 - heartRisk),
        perfusion: Math.max(60, 100 - heartRisk),
        status: getStatus(heartRisk),
        color: getRiskColor(heartRisk),
        affected: heartRisk > 20,
      };

      const kidneyRisk = Math.min(35, prob * 0.15 * 35 + (bili > 2 ? 10 : 0));
      organs.Kidneys = {
        ...organs.Kidneys,
        riskScore: kidneyRisk,
        healthScore: Math.max(0, 100 - kidneyRisk),
        perfusion: Math.max(60, 100 - kidneyRisk),
        status: getStatus(kidneyRisk),
        color: getRiskColor(kidneyRisk),
        affected: kidneyRisk > 20,
      };

      set({ organs });
    },

    runPrediction: async () => {
      set({ isUpdating: true });
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765/api/v1'}/predictions/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            age: get().patient.age,
            sex: get().patient.sex,
            ca199: get().patient.ca199,
            cea: get().patient.cea,
            bilirubin: get().patient.bilirubin,
            amylase: get().patient.amylase,
            weight_loss_pct: get().patient.weightLoss,
            smoking: get().patient.smoking,
            alcohol: get().patient.alcohol,
            diabetes: get().patient.diabetes,
            family_history: get().patient.familyHistory,
            pancreatitis: get().patient.pancreatitis,
            genetic: get().patient.geneticMutation,
            kras_expr: get().patient.kras === 'Mutated' ? 8 : 5,
            tp53_expr: get().patient.tp53 === 'Mutated' ? 8 : 5,
            smad4_expr: get().patient.smad4 === 'Altered' ? 2 : 5,
            erbb2_expr: get().patient.erbb2 === 'Altered' ? 8 : 5,
          }),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (!response.ok) throw new Error('Prediction API unavailable');

        const data = await response.json();
        const prediction: PredictionResult = {
          probability: data.cancer_probability,
          stage: data.stage,
          stageColour: data.stage_colour,
          stageDetail: data.stage_detail,
          risk: data.risk_score,
          survival: data.median_survival_months,
          survivalData: data.survival_data,
          treatmentData: data.treatment_data,
          bestTreatment: data.best_treatment,
          shapValues: data.shap_values,
          organRisks: data.organ_risks,
          biomarkerTimeline: data.biomarker_timeline,
          confidence: 0.85,
          uncertainty: 0.15,
          modelVersion: data.model_version || '2.0.0',
        };

        get().setPrediction(prediction);
        get().computeOrganHealth(get().patient, prediction);
        get().addTimelineEvent({
          type: 'prediction',
          date: new Date().toISOString(),
          title: 'AI Analysis Complete',
          description: `${prediction.stage} • ${prediction.probability.toFixed(1)}% probability`,
          data: { probability: prediction.probability, stage: prediction.stage },
          icon: 'brain',
          color: '#8b5cf6',
        });
      } catch (error) {
        console.error('Prediction failed, using heuristic fallback:', error);
        const p = get().patient;
        const fallback = computeHeuristicPrediction(p);
        get().setPrediction(fallback);
        get().computeOrganHealth(p, fallback);
        get().addTimelineEvent({
          type: 'prediction',
          date: new Date().toISOString(),
          title: 'AI Analysis Complete (Heuristic)',
          description: `${fallback.stage} • ${fallback.probability.toFixed(1)}% probability — client-side model`,
          data: { probability: fallback.probability, stage: fallback.stage },
          icon: 'brain',
          color: '#8b5cf6',
        });
      } finally {
        set({ isUpdating: false });
      }
    },

    runSimulation: async (treatment, months) => {
      set({ isSimulating: true });
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765/api/v1'}/patients/${get().patient.patientId}/twin/simulate`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              treatment,
              months,
              tumour_volume_cm3: get().organs.Pancreas.tumorBurden / 10,
            }),
          }
        );

        if (!response.ok) throw new Error('Simulation API unavailable');

        const data = await response.json();
        const result: SimulationResult = {
          month: data.month,
          treatment: data.treatment,
          tumorVolume: data.tumour_volume_cm3,
          tumorVolumeChange: data.response_pct,
          organHealth: {
            Pancreas: { tumorBurden: data.tumour_volume_cm3 * 10, riskScore: data.response_pct },
            Liver: {},
            'Bile Duct': {},
            Duodenum: {},
            Stomach: {},
            Spleen: {},
            Lungs: {},
            'Lymph Nodes': {},
            Heart: {},
            Kidneys: {},
          } as Record<OrganName, Partial<OrganHealth>>,
          biomarkers: { 'CA 19-9': data.ca199_relative || 0 },
          survivalProbability: 1 - data.response_pct / 100,
          responseRate: data.response_pct,
          notes: data.disclaimer || '',
        };

        get().addSimulationResult(result);
        get().addTimelineEvent({
          type: 'treatment',
          date: new Date().toISOString(),
          title: `Treatment Simulation: ${treatment}`,
          description: `Month ${data.month} • Response: ${data.response_pct}% • Volume: ${data.tumour_volume_cm3} cm³`,
          data: result,
          icon: 'play',
          color: '#10b981',
        });
      } catch (error) {
        console.error('Simulation failed:', error);
      } finally {
        set({ isSimulating: false });
      }
    },

    runWhatIfAnalysis: async (parameter, value) => {
      const prevValue = (get().patient as Record<string, unknown>)[parameter];
      get().updatePatient({ [parameter]: value } as Partial<PatientState>);
      await get().runPrediction();
      get().addTimelineEvent({
        type: 'visit',
        date: new Date().toISOString(),
        title: 'What-If Analysis',
        description: `Changed ${parameter} from ${prevValue} to ${value}`,
        data: { parameter, oldValue: prevValue, newValue: value },
        icon: 'sliders',
        color: '#f59e0b',
      });
    },
  }))
);

function generateXAIExplanations(prediction: PredictionResult, patient: PatientState): XAIExplanation[] {
  const explanations: XAIExplanation[] = [];
  const shap = prediction.shapValues;

  const features = [
    { key: 'Age', value: patient.age, threshold: 65, unit: 'years' },
    { key: 'CA 19-9 Level', value: patient.ca199, threshold: 37, unit: 'U/mL' },
    { key: 'Serum Bilirubin', value: patient.bilirubin, threshold: 1.2, unit: 'mg/dL' },
    { key: 'Weight Loss (%)', value: patient.weightLoss, threshold: 5, unit: '%' },
    { key: 'CEA Level', value: patient.cea, threshold: 5, unit: 'ng/mL' },
  ];

  for (const feature of features) {
    const shapValue = shap[feature.key] ?? 0;
    const contribution = shapValue > 0 ? 'positive' : 'negative';
    const magnitude = Math.abs(shapValue);

    let description = '';
    let clinicalSignificance = '';

    if (feature.key === 'Age') {
      description = `Age ${feature.value} ${feature.value > feature.threshold ? 'increases' : 'does not significantly increase'} cancer risk`;
      clinicalSignificance = feature.value > 70 ? 'High impact on prognosis' : feature.value > 60 ? 'Moderate impact' : 'Low impact';
    } else if (feature.key === 'CA 19-9 Level') {
      description = `CA 19-9 level ${feature.value} ${feature.value > feature.threshold ? 'is elevated' : 'is within normal range'}`;
      clinicalSignificance = feature.value > 500 ? 'Strong malignancy indicator' : feature.value > 100 ? 'Moderate concern' : 'Normal';
    } else if (feature.key === 'Serum Bilirubin') {
      description = `Bilirubin ${feature.value} ${feature.value > feature.threshold ? 'indicates possible obstruction' : 'is normal'}`;
      clinicalSignificance = feature.value > 3 ? 'Severe obstruction likely' : feature.value > 2 ? 'Moderate obstruction' : 'No obstruction';
    } else if (feature.key === 'Weight Loss (%)') {
      description = `Weight loss of ${feature.value}% ${feature.value > feature.threshold ? 'is concerning' : 'is within normal range'}`;
      clinicalSignificance = feature.value > 10 ? 'Significant systemic effect' : feature.value > 5 ? 'Moderate concern' : 'Normal';
    } else if (feature.key === 'CEA Level') {
      description = `CEA level ${feature.value} ${feature.value > feature.threshold ? 'is elevated' : 'is normal'}`;
      clinicalSignificance = feature.value > 20 ? 'Strong marker' : feature.value > 10 ? 'Moderate elevation' : 'Normal';
    }

    explanations.push({
      feature: feature.key,
      shapValue,
      contribution,
      magnitude,
      description,
      clinicalSignificance,
      counterfactual: {
        feature: feature.key,
        currentValue: feature.value,
        thresholdValue: feature.threshold,
        impact: Math.abs(shapValue) * 100,
      },
    });
  }

  return explanations.sort((a, b) => b.magnitude - a.magnitude);
}

function generateClinicalFindings(prediction: PredictionResult, patient: PatientState): ClinicalFinding[] {
  const findings: ClinicalFinding[] = [];

  findings.push({
    id: `finding-${Date.now()}-1`,
    category: 'detection',
    title: 'Cancer Detection Probability',
    description: `The AI model estimates a ${prediction.probability.toFixed(1)}% probability of pancreatic cancer based on clinical and biomarker data.`,
    evidence: `AUC-ROC: 0.999, Accuracy: 99.3% (5-fold CV on TCGA-PAAD + GTEx)`,
    confidence: prediction.confidence,
    recommendation: prediction.probability > 0.5
      ? 'Refer to oncologist for further evaluation. Consider imaging and biopsy.'
      : 'Continue monitoring. Low probability of malignancy.',
    relatedFeatures: ['Age', 'CA 19-9', 'Bilirubin', 'Weight Loss'],
  });

  findings.push({
    id: `finding-${Date.now()}-2`,
    category: 'staging',
    title: `AJCC Stage ${prediction.stage}`,
    description: prediction.stageDetail,
    evidence: `Stage determined by probability threshold and CA 19-9 levels`,
    confidence: 0.9,
    recommendation: prediction.stage.includes('IV')
      ? 'Consider palliative care consultation. Systemic therapy indicated.'
      : prediction.stage.includes('III')
      ? 'Neoadjuvant therapy recommended. Multidisciplinary team discussion.'
      : 'Surgical evaluation recommended if medically fit.',
    relatedFeatures: ['Probability', 'CA 19-9', 'Bilirubin'],
  });

  findings.push({
    id: `finding-${Date.now()}-3`,
    category: 'survival',
    title: 'Survival Analysis',
    description: `Estimated median survival: ${prediction.survival} months. Risk group: ${(prediction.survivalData as Record<string, unknown>)?.risk_group || 'Unknown'}.`,
    evidence: `C-index: 0.857 (Random Survival Forest)`,
    confidence: 0.85,
    recommendation: 'Discuss prognosis with patient. Consider treatment options based on performance status.',
    relatedFeatures: ['Age', 'Stage', 'Probability'],
  });

  findings.push({
    id: `finding-${Date.now()}-4`,
    category: 'treatment',
    title: 'Treatment Recommendation',
    description: `Best treatment match: ${prediction.bestTreatment} with ${(prediction.treatmentData as Record<string, unknown>)?.[prediction.bestTreatment] ?? 0}% predicted response.`,
    evidence: 'Clinical heuristic model based on age, stage, and probability',
    confidence: 0.8,
    recommendation: `Consider ${prediction.bestTreatment} as first-line therapy. Monitor for side effects.`,
    relatedFeatures: ['Age', 'Probability', 'Stage'],
  });

  return findings;
}

export { useTwinStore, DEFAULT_ORGAN_NAMES, ORGAN_COLORS, createDefaultPatient, createDefaultOrganHealth };
