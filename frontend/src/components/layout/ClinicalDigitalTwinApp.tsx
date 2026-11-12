'use client';

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  Activity, BrainCircuit, Calendar, ClipboardList, Database, Dna,
  FileDown, FlaskConical, ImageIcon, Layers3, Play, ShieldCheck,
  Stethoscope, TrendingUp, Users, Zap, Settings,
  BarChart3, AlertTriangle, Check, X, RefreshCw, SlidersHorizontal,
  Download, Share2, ZoomOut, Split, UserRound, ChevronDown, ChevronRight, ChevronUp,
  Scan, Upload, Bone,
} from 'lucide-react';
import { useTwinStore } from '../../engine/twin/twinStore';
import { DiseaseProgressionSimulator, getTreatmentOptions } from '../../engine/simulation/simulationEngine';
import { XAIEngine } from '../../engine/xai/xaiEngine';
import { TimelineEngine } from '../../engine/timeline/timelineEngine';
import { ClinicalTwinWorkspace, predictFromBiomarkers, biomarkersFromPatient, type BiomarkerForm } from '../workspace/ClinicalTwinWorkspace';
import { TwinMeshViewer } from '../workspace/TwinMeshViewer';
import { AppLogo } from '../ui/AppLogo';
import type { CtVolumePayload } from '../workspace/CtVolumeViewer';
import type { StepId, PatientState, PredictionResult, OrganName, OrganHealth, SimulationResult, TreatmentOption, TimelineEvent, TwinSnapshot, XAIExplanation, ClinicalFinding } from '../../types';

const STEPS: { id: StepId; label: string; icon: typeof Activity; description: string }[] = [
  { id: 'registry', label: 'Patient Registry', icon: Users, description: 'Create or select a patient record' },
  { id: 'clinical', label: 'Clinical Assessment', icon: Stethoscope, description: 'Capture clinical encounter data' },
  { id: 'labs', label: 'Lab Results', icon: FlaskConical, description: 'Record biomarker values' },
  { id: 'imaging', label: 'Medical Imaging', icon: ImageIcon, description: 'Attach DICOM/NIfTI studies' },
  { id: 'genomics', label: 'Genomics', icon: Dna, description: 'Record molecular findings' },
  { id: 'validation', label: 'Data Validation', icon: ShieldCheck, description: 'Resolve data quality checks' },
  { id: 'analysis', label: 'AI Analysis', icon: BrainCircuit, description: 'Generate prediction snapshot' },
  { id: 'twin', label: 'Clinical Digital Twin', icon: Layers3, description: '3D anatomical model with risk overlay' },
  { id: 'progression', label: 'Disease Progression', icon: TrendingUp, description: 'Explore disease trajectory' },
  { id: 'simulation', label: 'Treatment Simulation', icon: Play, description: 'Run treatment scenarios' },
  { id: 'comparison', label: 'Treatment Comparison', icon: ClipboardList, description: 'Compare response and safety' },
  { id: 'report', label: 'Clinical Report', icon: FileDown, description: 'Review structured report' },
  { id: 'export', label: 'Export', icon: Download, description: 'Export research record' },
];

const timelineEngine = new TimelineEngine();

interface OrganRadarProps {
  risks: Record<string, [number, string, string]>;
}
function OrganRadar({ risks }: OrganRadarProps) {
  const entries = Object.entries(risks);
  const n = entries.length;
  if (n === 0) {
    return <div style={{ fontSize: 11, color: '#607389' }}>No organ risk data</div>;
  }
  const cx = 100, cy = 100, R = 66, max = 100;
  const pts = entries.map(([, d]) => {
    const v = Number((d as [number, string, string])[0]);
    return { angle: 0, r: (Math.min(v, max) / max) * R };
  });
  const polygon = pts
    .map((p, i) => {
      const a = (2 * Math.PI * i) / n - Math.PI / 2;
      return `${cx + p.r * Math.cos(a)},${cy + p.r * Math.sin(a)}`;
    })
    .join(' ');
  const grid = [25, 50, 75, 100];
  return (
    <svg width="100%" height="200" viewBox="0 0 200 200" style={{ maxWidth: '100%' }}>
      {grid.map((g) => {
        const r = (g / max) * R;
        const d = Array.from({ length: n + 1 }, (_, i) => {
          const a = (2 * Math.PI * i) / n - Math.PI / 2;
          return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
        }).join(' ');
        return <polygon key={g} points={d} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="1" />;
      })}
      {entries.map(([, d], i) => {
        const a = (2 * Math.PI * i) / n - Math.PI / 2;
        return (
          <line
            key={i}
            x1={cx} y1={cy}
            x2={cx + R * Math.cos(a)} y2={cy + R * Math.sin(a)}
            stroke="rgba(255,255,255,0.08)" strokeWidth="1"
          />
        );
      })}
      <polygon points={polygon} fill="rgba(98, 206, 192, 0.25)" stroke="#62dec0" strokeWidth="2" />
      {entries.map(([name, d], i) => {
        const a = (2 * Math.PI * i) / n - Math.PI / 2;
        const v = Number(d[0]);
        const r = (Math.min(v, max) / max) * R;
        const x = cx + (r + 12) * Math.cos(a);
        const y = cy + (r + 12) * Math.sin(a);
        return (
          <g key={name}>
            <text
              x={cx + R * Math.cos(a)} y={cy + R * Math.sin(a)}
              fill="#8293a7" fontSize="9" textAnchor="middle"
              dy={a > 0 && a < Math.PI ? 12 : -4}
            >
              {name}
            </text>
            <text x={x} y={y} fill={d[1]} fontSize="10" fontWeight={600} textAnchor="middle">
              {v}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}

interface SurvivalProjection {
  median: number;
  riskScore: number;
  riskGroup: string;
  pts: { t: number; s: number }[];
  p1y: number;
  p2y: number;
}
function computeSurvivalProjection(prediction: PredictionResult | null): SurvivalProjection {
  const sd = prediction?.survivalData as Record<string, unknown> | undefined;
  const median = Number(sd?.median_months ?? prediction?.survival ?? 0);
  const lam = Math.log(2) / Math.max(median, 1);
  const pts = [3, 6, 12, 18, 24, 36].map((t) => ({ t, s: Math.exp(-((lam * t) ** 1.15)) }));
  return {
    median,
    riskScore: Number(sd?.risk_score ?? 0),
    riskGroup: String(sd?.risk_group ?? 'Unknown'),
    pts,
    p1y: pts.find((p) => p.t === 12)?.s ?? 0,
    p2y: pts.find((p) => p.t === 24)?.s ?? 0,
  };
}

export function ClinicalDigitalTwinApp() {
  const {
    patient,
    patientList,
    organs,
    prediction,
    workflow,
    theme,
    renderMode,
    simulationMode,
    simulationMonth,
    isSimulating,
    isUpdating,
    selectedOrgan,
    showAnnotations,
    showMetastasisPaths,
    showBiomarkerTimeline,
    selectedTreatment,
    comparisonTreatments,
    treatments,
    simulationResults,
    xaiExplanations,
    clinicalFindings,
    timeline,
    snapshots,
    updatePatient,
    applyBiomarkers,
    setPrediction,
    setWorkflow,
    setTheme,
    setRenderMode,
    setSimulationMode,
    setSimulationMonth,
    setIsSimulating,
    setSelectedOrgan,
    setShowAnnotations,
    setShowMetastasisPaths,
    setShowBiomarkerTimeline,
    setSelectedTreatment,
    setComparisonTreatments,
    setTreatments,
    addSimulationResult,
    setXAIExplanations,
    addClinicalFinding,
    addTimelineEvent,
    addSnapshot,
    savePatientToList,
    loadPatient,
    deletePatientFromList,
    loadPatientsFromStorage,
    computeOrganHealth,
    runPrediction,
    runSimulation,
    runWhatIfAnalysis,
    syntheticModel,
    setSyntheticModel,
  } = useTwinStore();

  useEffect(() => {
    loadPatientsFromStorage();
  }, [loadPatientsFromStorage]);

  const imagingInputRef = useRef<HTMLInputElement | null>(null);
  const ctInputRef = useRef<HTMLInputElement | null>(null);
  const [defaultPatientId, setDefaultPatientId] = useState('');
  useEffect(() => {
    if (!defaultPatientId) {
      setDefaultPatientId(`P${String(Date.now()).slice(-6)}`);
    }
  }, [defaultPatientId]);

  useEffect(() => {
    const input = imagingInputRef.current;
    if (!input) return;
    (input as any).webkitdirectory = true;
    (input as any).directory = true;
  }, []);

  useEffect(() => {
    const input = ctInputRef.current;
    if (!input) return;
    (input as any).webkitdirectory = true;
    (input as any).directory = true;
  }, []);

  const [activePanel, setActivePanel] = useState<'patient' | 'xai' | 'treatment' | 'timeline' | 'settings'>('patient');
  const [treatmentScenario, setTreatmentScenario] = useState('FOLFIRINOX');
  const [timeHorizon, setTimeHorizon] = useState(6);
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  const [comparisonResult, setComparisonResult] = useState<Record<string, unknown> | null>(null);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [segmentationLoading, setSegmentationLoading] = useState(false);
  const [segmentationError, setSegmentationError] = useState<string | null>(null);
  const [twinView, setTwinView] = useState<'realct' | 'ct' | 'library' | 'vtk-ct'>('vtk-ct');
  const [ctCase, setCtCase] = useState('414');
  const [ctCases, setCtCases] = useState<string[]>(['414', '409', '010', '219', '001']);
  const [ctVolume, setCtVolume] = useState<CtVolumePayload | null>(null);
  const [ctVolumeLoading, setCtVolumeLoading] = useState(false);
  const [ctVolumeError, setCtVolumeError] = useState<string | null>(null);
  const [twinPanelOpen, setTwinPanelOpen] = useState(true);
  const [twinShowBiomarkers, setTwinShowBiomarkers] = useState(true);

  // Single Patient Case Folder Upload state (contains imagesTr, labelsTr, meshes.npz)
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  const [folderUploading, setFolderUploading] = useState(false);
  const [folderUploadError, setFolderUploadError] = useState<string | null>(null);

  const handleUploadFolder = useCallback(async (files: File[]) => {
    if (!files || files.length === 0) return;
    setFolderFiles(files);
    setFolderUploading(true);
    setFolderUploadError(null);
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765/api/v1';
    const formData = new FormData();
    files.forEach((f) => {
      const relPath = (f as any).webkitRelativePath || f.name;
      formData.append('files', f, relPath);
    });

    try {
      const response = await fetch(`${apiUrl}/twin/mesh/upload-folder`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Folder upload failed (${response.status})`);
      }
      const payload = await response.json();
      const folderName = (files[0] as any)?.webkitRelativePath?.split('/')[0] || 'Patient Case Folder';
      updatePatient({
        uploadedPayload: payload,
        imagingFiles: files,
        imagingStudy: `Folder: ${folderName} (${files.length} items)`,
        imagingStatus: `Combined 3D Anatomy Extracted (imagesTr + labelsTr + meshes.npz)`,
      });
      setTwinView('vtk-ct');
    } catch (err: any) {
      setFolderUploadError(err.message || 'Failed to process patient folder via VTK pipeline');
    } finally {
      setFolderUploading(false);
    }
  }, [updatePatient]);

  const twinSummary = useMemo(() => {
    if (!patient) return null;
    try {
      return predictFromBiomarkers(biomarkersFromPatient(patient));
    } catch {
      return null;
    }
  }, [patient]);

  const [biomarkers, setBiomarkers] = useState<BiomarkerForm>(() => biomarkersFromPatient(patient ?? ({} as PatientState)));
  const [predictionRunning, setPredictionRunning] = useState(false);
  const [biomarkerNotice, setBiomarkerNotice] = useState('');

  useEffect(() => {
    setBiomarkers(biomarkersFromPatient(patient ?? ({} as PatientState)));
  }, [patient]);

  const updateBiomarker = useCallback((key: keyof BiomarkerForm, value: string | boolean) => {
    const next = {
      ...biomarkers,
      [key]: typeof value === 'boolean' ? value : Number.isNaN(Number(value)) ? value : Number(value),
    };
    setBiomarkers(next);
    applyBiomarkers({
      name: next.name, age: next.age, sex: next.sex,
      ca199: next.ca199, cea: next.cea, bilirubin: next.bilirubin, albumin: next.albumin,
      weightLoss: next.weight_loss_pct, kras: next.kras, tp53: next.tp53, smad4: next.smad4,
    });
  }, [biomarkers, applyBiomarkers]);

  const runBiomarkerPrediction = useCallback(async () => {
    setPredictionRunning(true);
    setBiomarkerNotice('Running clinical prediction…');
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765/api/v1';
      const r = await fetch(`${apiUrl}/predictions/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(biomarkers),
      });
      if (!r.ok) throw Error('Prediction API unavailable');
      const d = await r.json();
      setBiomarkerNotice(`Prediction complete • ${d.stage} • ${d.cancer_probability}% probability`);
    } catch {
      setBiomarkerNotice('Fallback: using local heuristic prediction');
    } finally {
      setPredictionRunning(false);
    }
  }, [biomarkers]);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765/api/v1';
    fetch(`${apiUrl}/twin/mesh/options`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d: { ct_cases?: string[]; default_case?: string }) => {
        if (d.ct_cases?.length) setCtCases(d.ct_cases);
        if (d.default_case) setCtCase(d.default_case);
      })
      .catch(() => {});
  }, []);

  const activeIndex = STEPS.findIndex((s) => s.id === workflow.currentStep);
  const pct = Math.round((workflow.completedSteps.length / STEPS.length) * 100);

  const canOpen = useCallback(
    (id: StepId) => id === 'registry' || workflow.completedSteps.includes(STEPS[Math.max(0, STEPS.findIndex((s) => s.id === id) - 1)].id),
    [workflow.completedSteps]
  );

  const handleNext = useCallback(() => {
    setWorkflow({
      completedSteps: [...workflow.completedSteps, workflow.currentStep],
      currentStep: STEPS[activeIndex + 1]?.id || workflow.currentStep,
    });
  }, [workflow, activeIndex, setWorkflow]);

  const handlePrev = useCallback(() => {
    if (activeIndex > 0) {
      setWorkflow({ currentStep: STEPS[activeIndex - 1].id });
    }
  }, [activeIndex, setWorkflow]);

  const handleRunAnalysis = useCallback(async () => {
    await runPrediction();
  }, [runPrediction]);

  const handleProcessImagingStudy = useCallback(async () => {
    const patientId = patient.patientId || defaultPatientId;
    if (!patient.imagingStudyId) return;

    setSegmentationLoading(true);
    setSegmentationError(null);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765/api/v1';
      const response = await fetch(
        `${apiUrl}/patients/${patientId}/imaging/studies/${patient.imagingStudyId}/process`,
        { method: 'POST' }
      );
      if (!response.ok) {
        throw new Error(`Segmentation processing failed (${response.status})`);
      }
      const study = await response.json();
      updatePatient({
        imagingSegmentation: study.segmentation,
        imagingStatus: `Segmentation processed (${study.segmentation.quality_score}% quality)`,
      });
    } catch (error) {
      console.error('Segmentation processing failed', error);
      setSegmentationError('Unable to process imaging study. Please retry or refresh.');
      updatePatient({ imagingStatus: 'Segmentation processing failed' });
    } finally {
      setSegmentationLoading(false);
    }
  }, [patient, defaultPatientId, updatePatient]);

  const loadCtVolume = useCallback(async (caseId: string) => {
    setCtVolumeLoading(true);
    setCtVolumeError(null);
    setCtVolume(null);
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765/api/v1';
    try {
      const r = await fetch(`${apiUrl}/twin/mesh/ct-volume?case=${encodeURIComponent(caseId)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setCtVolume((await r.json()) as CtVolumePayload);
    } catch (e) {
      setCtVolumeError(e instanceof Error ? e.message : String(e));
    } finally {
      setCtVolumeLoading(false);
    }
  }, []);

  const handleCtUpload = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setCtVolumeLoading(true);
    setCtVolumeError(null);
    setCtVolume(null);
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765/api/v1';
    const formData = new FormData();
    Array.from(files).forEach((file) => formData.append('files', file));
    formData.append('patient_id', patient?.patientId || defaultPatientId);
    try {
      const r = await fetch(`${apiUrl}/twin/mesh/ct-volume`, { method: 'POST', body: formData });
      if (!r.ok) throw new Error(`Upload failed (${r.status})`);
      setCtVolume((await r.json()) as CtVolumePayload);
    } catch (e) {
      setCtVolumeError(e instanceof Error ? e.message : String(e));
    } finally {
      setCtVolumeLoading(false);
    }
  }, [patient, defaultPatientId]);

  const handleSelectCtCase = useCallback((value: string) => {
    setCtCase(value);
    if (twinView === 'realct' || twinView === 'ct') loadCtVolume(value);
  }, [twinView, loadCtVolume]);

  const handleSimulate = useCallback(async () => {
    const responseProb = (treatments.find((t: TreatmentOption) => t.name === treatmentScenario)?.responseProbability ?? 0.5);
    await runSimulation(treatmentScenario, timeHorizon);

    const simulator = new DiseaseProgressionSimulator(patient, prediction);
    const result = simulator.simulate(timeHorizon, treatmentScenario, responseProb);
    setSimulationResult(result);
    addSnapshot({ patient, organs, prediction, month: timeHorizon, treatment: treatmentScenario, notes: `Simulation: ${treatmentScenario}` });
  }, [patient, prediction, organs, treatmentScenario, timeHorizon, treatments, runSimulation, addSnapshot]);

  const handleCompareTreatments = useCallback(async () => {
    if (!prediction) return;

    const simulator = new DiseaseProgressionSimulator(patient, prediction);
    const results: Record<string, unknown> = {};

    for (const tx of comparisonTreatments) {
      const txOption = treatments.find((t: TreatmentOption) => t.name === tx);
      const responseProb = txOption?.responseProbability ?? 0.5;
      const result = simulator.simulate(6, tx, responseProb);
      results[tx] = result;
    }

    setComparisonResult(results);
  }, [patient, prediction, comparisonTreatments, treatments]);

  const handleWhatIf = useCallback(async (parameter: string, value: number) => {
    await runWhatIfAnalysis(parameter, value);
  }, [runWhatIfAnalysis]);

  const handleSavePatient = useCallback(() => {
    savePatientToList();
    setWorkflow({ currentStep: 'clinical' });
  }, [savePatientToList, setWorkflow]);

  const handleRenderModeChange = useCallback((mode: string) => {
    setRenderMode(mode as any);
  }, [setRenderMode]);

  const currentStep = STEPS[activeIndex];

  return (
    <div className="clinical-twin-app" style={{
      minHeight: '100vh',
      background: theme === 'dark'
        ? 'radial-gradient(circle at 55% 28%, #142333 0%, #0a1018 37%, #080d14 70%)'
        : 'radial-gradient(circle at 55% 28%, #e8f0f5 0%, #d0e0ea 37%, #b8d0e0 70%)',
      color: theme === 'dark' ? '#e5edf8' : '#2c3e50',
      fontFamily: 'Manrope, Arial, sans-serif',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Top Bar */}
      <header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 24px',
        background: 'rgba(15, 26, 38, 0.7)',
        borderBottom: '1px solid #1a2a39',
        backdropFilter: 'blur(10px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <AppLogo size={44} />
          <div>
            <div style={{ fontSize: 20, fontWeight: 700, letterSpacing: -0.2 }}>ExAI-PanC Clinical Digital Twin</div>
            <div style={{ fontSize: 11, color: '#607389', letterSpacing: 1.4, textTransform: 'uppercase' }}>
              AI-Powered Clinical Decision Support Platform
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '6px 14px',
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: 8,
            fontSize: 12,
            color: '#8293a7',
          }}>
            <span className="status-dot" style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: prediction ? '#22c55e' : '#f59e0b',
              animation: prediction ? 'pulse 2s infinite' : 'none',
            }} />
            {prediction ? `Analysis Complete • ${prediction.stage}` : 'Ready for Analysis'}
          </div>

          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            style={{
              padding: 8,
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid #1a2a39',
              borderRadius: 6,
              color: '#8293a7',
              cursor: 'pointer',
            }}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>

          <button
            onClick={() => setIsFullScreen(!isFullScreen)}
            style={{
              padding: 8,
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid #1a2a39',
              borderRadius: 6,
              color: '#8293a7',
              cursor: 'pointer',
            }}
          >
            {isFullScreen ? '⤢' : '⤢'}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div style={{
        flex: 1,
        display: 'flex',
        overflow: 'hidden',
        height: isFullScreen ? 'calc(100vh - 56px)' : 'calc(100vh - 56px - 80px)',
      }}>
        {/* Left Sidebar - Workflow Navigation */}
        <aside style={{
          width: 280,
          background: 'rgba(15, 26, 38, 0.5)',
          borderRight: '1px solid #1a2a39',
          display: 'flex',
          flexDirection: 'column',
          overflowY: 'auto',
          backdropFilter: 'blur(10px)',
        }}>
          <div style={{
            padding: '16px',
            borderBottom: '1px solid #1a2a39',
          }}>
            <div style={{ fontSize: 12, color: '#607389', letterSpacing: 1, textTransform: 'uppercase' }}>
              Clinical Workflow
            </div>
            <div style={{
              marginTop: 8,
              height: 6,
              background: 'rgba(255, 255, 255, 0.05)',
              borderRadius: 3,
              overflow: 'hidden',
            }}>
              <div style={{
                width: `${pct}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #3ec7ad, #62dec0)',
                borderRadius: 3,
                transition: 'width 0.3s ease',
              }} />
            </div>
            <div style={{ fontSize: 11, color: '#607389', marginTop: 4 }}>
              {pct}% Complete • Step {activeIndex + 1} of {STEPS.length}
            </div>
          </div>

          <nav style={{ flex: 1, padding: 8 }}>
            {STEPS.map((step, index) => {
              const Icon = step.icon;
              const locked = !canOpen(step.id);
              const isComplete = workflow.completedSteps.includes(step.id);
              const isCurrent = workflow.currentStep === step.id;

              return (
                <button
                  key={step.id}
                  onClick={() => canOpen(step.id) && setWorkflow({ currentStep: step.id })}
                  disabled={locked}
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    margin: '4px 0',
                    background: isCurrent
                      ? 'rgba(62, 199, 173, 0.15)'
                      : isComplete
                      ? 'rgba(255, 255, 255, 0.03)'
                      : 'transparent',
                    border: isCurrent
                      ? '1px solid #3ec7ad'
                      : '1px solid transparent',
                    borderRadius: 8,
                    color: locked ? '#334155' : isCurrent ? '#3ec7ad' : '#b0c8d8',
                    cursor: locked ? 'not-allowed' : 'pointer',
                    transition: 'all 0.2s ease',
                    textAlign: 'left',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                  }}
                >
                  <span style={{
                  display: 'grid',
                  placeItems: 'center',
                  width: 32,
                  height: 32,
                  borderRadius: 9,
                  background: isCurrent
                    ? 'rgba(62, 199, 173, 0.18)'
                    : isComplete
                      ? 'rgba(62, 199, 173, 0.1)'
                      : locked
                        ? 'rgba(255, 255, 255, 0.03)'
                        : 'rgba(255, 255, 255, 0.05)',
                  border: isCurrent
                    ? '1px solid #3ec7ad'
                    : isComplete
                      ? '1px solid #3ec7ad'
                      : '1px solid #1a2a39',
                  color: isCurrent ? '#3ec7ad' : locked ? '#334155' : isComplete ? '#3ec7ad' : '#62dec0',
                  flexShrink: 0,
                  position: 'relative',
                }}>
                  <Icon size={17} />
                  {isComplete && <Check size={11} style={{ position: 'absolute', bottom: 3, right: 3, color: '#22c55e' }} />}
                </span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: isCurrent ? 600 : 500 }}>{step.label}</div>
                  <div style={{ fontSize: 10, color: '#607389', marginTop: 2 }}>{step.description}</div>
                </div>
              </button>
              );
            })}
          </nav>

          <div style={{
            padding: 16,
            borderTop: '1px solid #1a2a39',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}>
            <ShieldCheck size={15} color="#8293a7" />
            <span style={{ fontSize: 11, color: '#8293a7' }}>
              Research environment • Audit trail enabled
            </span>
          </div>
        </aside>

        {/* Main Workspace */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Workspace Header */}
          <div style={{
            padding: '16px 24px',
            borderBottom: '1px solid #1a2a39',
            background: 'rgba(15, 26, 38, 0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 16,
          }}>
            <div>
              <div style={{ fontSize: 11, color: '#607389', letterSpacing: 1, textTransform: 'uppercase' }}>
                WORKSPACE / {String(activeIndex + 1).padStart(2, '0')} OF {STEPS.length}
              </div>
              <h1 style={{ fontSize: 20, margin: '4px 0', fontWeight: 600 }}>{currentStep.label}</h1>
              <p style={{ fontSize: 13, color: '#8293a7', margin: 0 }}>{currentStep.description}</p>
            </div>

            {workflow.currentStep === 'twin' && (
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                {twinSummary && (
                  <span style={{
                    fontSize: 11, fontWeight: 600, padding: '7px 10px', borderRadius: 6,
                    background: twinSummary.stage.includes('IV') || twinSummary.stage.includes('III')
                      ? 'rgba(231,76,60,0.15)'
                      : twinSummary.stage === 'Non-malignant' ? 'rgba(52,152,219,0.15)' : 'rgba(243,156,18,0.15)',
                    border: '1px solid #1a2a39', color: twinSummary.stageCol,
                  }}>
                    {twinSummary.stage}
                  </span>
                )}
                <button
                  onClick={() => { setTwinPanelOpen(true); setTwinShowBiomarkers(true); }}
                  style={{
                    padding: '8px 14px',
                    background: 'rgba(62, 199, 173, 0.2)',
                    border: '1px solid #3ec7ad',
                    borderRadius: 6, color: '#3ec7ad', fontSize: 11, cursor: 'pointer',
                    display: 'flex', alignItems: 'center', gap: 6,
                  }}
                >
                  <UserRound size={13} /> Patient
                </button>
                <button
                  onClick={() => setTwinPanelOpen((v) => !v)}
                  style={{
                    padding: '8px 14px',
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid #1a2a39',
                    borderRadius: 6, color: '#8293a7', fontSize: 11, cursor: 'pointer',
                    display: 'flex', alignItems: 'center', gap: 6,
                  }}
                >
                  {twinPanelOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />} Panel
                </button>
                <button
                  onClick={() => { setRenderMode('heatmap'); setTwinView('vtk-ct'); }}
                  style={{
                    padding: '8px 14px',
                    background: twinView === 'vtk-ct' ? 'rgba(62, 199, 173, 0.2)' : 'rgba(255, 255, 255, 0.03)',
                    border: twinView === 'vtk-ct' ? '1px solid #3ec7ad' : '1px solid #1a2a39',
                    borderRadius: 6,
                    color: twinView === 'vtk-ct' ? '#3ec7ad' : '#8293a7',
                    fontSize: 11,
                    cursor: 'pointer',
                  }}
                >
                  VTK CT Twin
                </button>
                <button
                  onClick={() => { setRenderMode('heatmap'); setTwinView('realct'); }}
                  style={{
                    padding: '8px 14px',
                    background: twinView === 'realct' ? 'rgba(62, 199, 173, 0.2)' : 'rgba(255, 255, 255, 0.03)',
                    border: twinView === 'realct' ? '1px solid #3ec7ad' : '1px solid #1a2a39',
                    borderRadius: 6,
                    color: twinView === 'realct' ? '#3ec7ad' : '#8293a7',
                    fontSize: 11,
                    cursor: 'pointer',
                  }}
                >
                  Real CT Twin
                </button>
                <button
                  onClick={() => {
                    setRenderMode('ct');
                    setTwinView('ct');
                    if (!ctVolume && !ctVolumeLoading) loadCtVolume(ctCase);
                  }}
                  style={{
                    padding: '8px 14px',
                    background: twinView === 'ct' ? 'rgba(62, 199, 173, 0.2)' : 'rgba(255, 255, 255, 0.03)',
                    border: twinView === 'ct' ? '1px solid #3ec7ad' : '1px solid #1a2a39',
                    borderRadius: 6,
                    color: twinView === 'ct' ? '#3ec7ad' : '#8293a7',
                    fontSize: 11,
                    cursor: 'pointer',
                    display: 'flex', alignItems: 'center', gap: 6,
                  }}
                >
                  <Scan size={13} /> Patient CT
                </button>
                <button
                  onClick={() => { setRenderMode('heatmap'); setTwinView('library'); }}
                  style={{
                    padding: '8px 14px',
                    background: twinView === 'library' ? 'rgba(62, 199, 173, 0.2)' : 'rgba(255, 255, 255, 0.03)',
                    border: twinView === 'library' ? '1px solid #3ec7ad' : '1px solid #1a2a39',
                    borderRadius: 6,
                    color: twinView === 'library' ? '#3ec7ad' : '#8293a7',
                    fontSize: 11,
                    cursor: 'pointer',
                  }}
                >
                  <Bone size={13} /> Anatomy Library
                </button>
                {(twinView === 'realct' || twinView === 'vtk-ct') && (
                  patient.uploadedPayload ? (
                    <span style={{
                      padding: '6px 12px',
                      background: 'rgba(62, 199, 173, 0.15)',
                      border: '1px solid #3ec7ad',
                      borderRadius: 6,
                      color: '#3ec7ad',
                      fontSize: 11,
                      fontWeight: 600,
                    }}>
                      ✓ Uploaded Patient Case Active
                    </span>
                  ) : (
                    <select
                      value={ctCase}
                      onChange={(e) => setCtCase(e.target.value)}
                      style={{
                        padding: '7px 8px',
                        background: 'rgba(0, 0, 0, 0.3)',
                        border: '1px solid #3ec7ad',
                        borderRadius: 6,
                        color: '#3ec7ad',
                        fontSize: 11,
                        cursor: 'pointer',
                      }}
                    >
                      {ctCases.map((c) => <option key={c} value={c}>Case {c}</option>)}
                    </select>
                  )
                )}
                {twinView === 'ct' && (
                  <>
                    <select
                      value={ctCase}
                      onChange={(e) => handleSelectCtCase(e.target.value)}
                      style={{
                        padding: '7px 8px',
                        background: 'rgba(0, 0, 0, 0.3)',
                        border: '1px solid #3ec7ad',
                        borderRadius: 6,
                        color: '#3ec7ad',
                        fontSize: 11,
                        cursor: 'pointer',
                      }}
                    >
                      {ctCases.map((c) => <option key={c} value={c}>Case {c}</option>)}
                    </select>
                    <button
                      onClick={() => ctInputRef.current?.click()}
                      disabled={ctVolumeLoading}
                      style={{
                        padding: '8px 14px',
                        background: ctVolumeLoading ? 'rgba(255,255,255,0.05)' : 'rgba(62, 199, 173, 0.2)',
                        border: '1px solid #3ec7ad',
                        borderRadius: 6,
                        color: ctVolumeLoading ? '#607389' : '#3ec7ad',
                        fontSize: 11,
                        cursor: ctVolumeLoading ? 'not-allowed' : 'pointer',
                        display: 'flex', alignItems: 'center', gap: 6,
                      }}
                    >
                      <Upload size={13} /> {ctVolumeLoading ? 'Loading CT…' : 'Load CT DICOM…'}
                    </button>
                    <input
                      ref={ctInputRef}
                      type="file"
                      multiple
                      style={{ display: 'none' }}
                      // Folder selection is supported by Chromium-based clinical workstations.
                      // @ts-ignore non-standard folder input attribute
                      webkitdirectory=""
                      // @ts-ignore non-standard folder input attribute
                      directory=""
                      accept=".dcm,.ima,.dicom,.nii,.nii.gz"
                      onChange={(e) => {
                        handleCtUpload(e.target.files);
                        e.target.value = '';
                      }}
                    />
                  </>
                )}
              </div>
            )}

            {workflow.currentStep === 'analysis' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>
                <div>
                  <strong style={{ fontSize: 14 }}>Prediction Snapshot</strong>
                  <div style={{ fontSize: 12, color: '#8293a7', marginTop: 2 }}>
                    Detection • Staging • Survival • Organ Risk • SHAP
                  </div>
                </div>
                <button
                  onClick={handleRunAnalysis}
                  disabled={isUpdating}
                  style={{
                    padding: '10px 24px', background: 'linear-gradient(135deg, #62dec0, #3ec7ad)',
                    border: '1px solid #54dfc4', borderRadius: 8, color: '#08221f',
                    fontSize: 13, fontWeight: 700,
                    cursor: isUpdating ? 'not-allowed' : 'pointer',
                    display: 'flex', alignItems: 'center', gap: 8,
                  }}
                >
                  <BrainCircuit size={16} />
                  {isUpdating ? 'Running Pipeline…' : 'Run AI Analysis'}
                </button>
              </div>
            )}
          </div>

          {/* Workspace Content */}
          <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
            {renderWorkspaceContent()}
          </div>

          {/* Bottom Navigation */}
          <div style={{
            padding: '14px 24px',
            borderTop: '1px solid #1a2a39',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 8,
            background: 'rgba(15, 26, 38, 0.3)',
          }}>
            {activeIndex > 0 ? (
              <button
                onClick={handlePrev}
                style={{
                  padding: '8px 14px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid #1a2a39',
                  borderRadius: 6,
                  color: '#8293a7',
                  fontSize: 12,
                  cursor: 'pointer',
                }}
              >
                ← Previous
              </button>
            ) : <span />}
            {activeIndex < STEPS.length - 1 && (
              <button
                onClick={handleNext}
                style={{
                  padding: '8px 14px',
                  background: 'linear-gradient(135deg, #3ec7ad, #62dec0)',
                  border: '1px solid #54dfc4',
                  borderRadius: 6,
                  color: '#08221f',
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Next →
              </button>
            )}
          </div>
        </main>

        {/* Right Panel - Contextual Tools */}
        <aside style={{
          width: 360,
          background: 'rgba(15, 26, 38, 0.5)',
          borderLeft: '1px solid #1a2a39',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          backdropFilter: 'blur(10px)',
        }}>
          {/* Patient & Biomarker Inputs — always visible on every page */}
          <div style={{
            padding: 12,
            borderBottom: '1px solid #1a2a39',
            flexShrink: 0,
            maxHeight: '52%',
            overflow: 'auto',
          }}>
            {renderBiomarkerForm()}
          </div>

          <div style={{
            padding: '12px 16px',
            borderBottom: '1px solid #1a2a39',
            display: 'flex',
            gap: 4,
            flexShrink: 0,
          }}>
            {(['patient', 'xai', 'treatment', 'timeline', 'settings'] as const).map((panel) => (
              <button
                key={panel}
                onClick={() => setActivePanel(panel)}
                style={{
                  flex: 1,
                  padding: '10px 8px',
                  background: activePanel === panel
                    ? 'rgba(62, 199, 173, 0.15)'
                    : 'rgba(255, 255, 255, 0.03)',
                  border: activePanel === panel
                    ? '1px solid #3ec7ad'
                    : '1px solid transparent',
                  borderRadius: 6,
                  color: activePanel === panel ? '#3ec7ad' : '#8293a7',
                  fontSize: 11,
                  fontWeight: activePanel === panel ? 600 : 500,
                  cursor: 'pointer',
                  textTransform: 'uppercase',
                  letterSpacing: 0.5,
                }}
              >
                {panel}
              </button>
            ))}
          </div>

          <div style={{ flex: 1, overflow: 'auto', padding: 16 }}>
            {renderRightPanel()}
          </div>
        </aside>
      </div>

      {/* Footer */}
      <footer style={{
        padding: '12px 24px',
        borderTop: '1px solid #1a2a39',
        background: 'rgba(15, 26, 38, 0.3)',
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: 11,
        color: '#607389',
      }}>
        <div style={{ display: 'flex', gap: 16 }}>
          <ShieldCheck size={14} />
          <span>Research use only. Not for clinical decisions.</span>
        </div>
        <div style={{ display: 'flex', gap: 16 }}>
          <span>Model v2.0.0</span>
          <span>•</span>
          <span>TCGA-PAAD + GTEx</span>
        </div>
      </footer>
    </div>
  );

  function renderWorkspaceContent() {
    switch (workflow.currentStep) {
      case 'registry':
        return renderRegistryStep();
      case 'clinical':
        return renderClinicalStep();
      case 'labs':
        return renderLabsStep();
      case 'imaging':
        return renderImagingStep();
      case 'genomics':
        return renderGenomicsStep();
      case 'validation':
        return renderValidationStep();
      case 'analysis':
        return renderAnalysisStep();
      case 'twin':
        return renderTwinStep();
      case 'progression':
        return renderProgressionStep();
      case 'simulation':
        return renderSimulationStep();
      case 'comparison':
        return renderComparisonStep();
      case 'report':
        return renderReportStep();
      case 'export':
        return renderExportStep();
      default:
        return renderRegistryStep();
    }
  }

  function renderRightPanel() {
    switch (activePanel) {
      case 'patient':
        return renderPatientPanel();
      case 'xai':
        return renderXAIPanel();
      case 'treatment':
        return renderTreatmentPanel();
      case 'timeline':
        return renderTimelinePanel();
      case 'settings':
        return renderSettingsPanel();
      default:
        return renderPatientPanel();
    }
  }

  function renderRegistryStep() {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div className="content-card" style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid #1a2a39',
          borderRadius: 12,
          padding: 24,
        }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
            <div>
              <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
                Patient ID
              </label>
              <input
                readOnly
                value={patient.patientId || defaultPatientId}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  background: 'rgba(0, 0, 0, 0.2)',
                  border: '1px solid #1a2a39',
                  borderRadius: 6,
                  color: '#62dec0',
                  fontSize: 13,
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
                Full Name *
              </label>
              <input
                value={patient.name}
                onChange={(e) => updatePatient({ name: e.target.value })}
                placeholder="Enter patient name"
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  background: 'rgba(0, 0, 0, 0.2)',
                  border: '1px solid #1a2a39',
                  borderRadius: 6,
                  color: '#e5edf8',
                  fontSize: 13,
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
                Age *
              </label>
              <input
                type="number"
                min={0}
                max={120}
                value={patient.age || ''}
                onChange={(e) => updatePatient({ age: parseInt(e.target.value) || 0 })}
                placeholder="Enter age"
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  background: 'rgba(0, 0, 0, 0.2)',
                  border: '1px solid #1a2a39',
                  borderRadius: 6,
                  color: '#e5edf8',
                  fontSize: 13,
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
                Sex *
              </label>
              <select
                value={patient.sex}
                onChange={(e) => updatePatient({ sex: e.target.value as any })}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  background: 'rgba(0, 0, 0, 0.2)',
                  border: '1px solid #1a2a39',
                  borderRadius: 6,
                  color: patient.sex ? '#e5edf8' : '#607389',
                  fontSize: 13,
                }}
              >
                <option value="">Select sex</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
          </div>

          <div style={{ marginTop: 24, display: 'flex', gap: 12 }}>
            <button
              onClick={handleSavePatient}
              disabled={!patient.name || !patient.age || !patient.sex}
              style={{
                padding: '12px 24px',
                background: patient.name && patient.age && patient.sex
                  ? 'linear-gradient(135deg, #3ec7ad, #62dec0)'
                  : 'rgba(255, 255, 255, 0.05)',
                border: patient.name && patient.age && patient.sex
                  ? '1px solid #54dfc4'
                  : '1px solid #1a2a39',
                borderRadius: 8,
                color: patient.name && patient.age && patient.sex ? '#08221f' : '#607389',
                fontSize: 13,
                fontWeight: 600,
                cursor: patient.name && patient.age && patient.sex ? 'pointer' : 'not-allowed',
              }}
            >
              Save Patient & Continue
            </button>
          </div>

          <div style={{
            marginTop: 24,
            padding: '12px 16px',
            background: 'rgba(13, 42, 66, 0.3)',
            border: '1px solid #1a2a39',
            borderRadius: 8,
            display: 'flex',
            gap: 8,
          }}>
            <ShieldCheck size={16} color="#8293a7" />
            <span style={{ fontSize: 12, color: '#8293a7' }}>
              Use a local study identifier. Do not enter directly identifying information in this research environment.
            </span>
          </div>
        </div>

        {patientList.length > 0 && (
          <div className="content-card" style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: '1px solid #1a2a39',
            borderRadius: 12,
            padding: 24,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
              <Database size={20} color="#8293a7" />
              <div>
                <h3 style={{ fontSize: 15, margin: 0, fontWeight: 600 }}>Saved Patients</h3>
                <p style={{ fontSize: 12, color: '#607389', margin: 0 }}>
                  Click a patient to load their data into the workspace.
                </p>
              </div>
            </div>

            <div style={{ display: 'grid', gap: 8 }}>
              {patientList.map((p) => (
                <div
                  key={p.patientId}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    padding: '12px 16px',
                    background: patient.patientId === p.patientId
                      ? 'rgba(62, 199, 173, 0.1)'
                      : 'rgba(255, 255, 255, 0.02)',
                    border: patient.patientId === p.patientId
                      ? '1px solid #3ec7ad'
                      : '1px solid #1a2a39',
                    borderRadius: 8,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                  onClick={() => {
                    loadPatient(p.patientId);
                    setWorkflow({
                      currentStep: 'clinical',
                      completedSteps: ['registry'],
                    });
                  }}
                >
                  <div style={{
                    width: 40,
                    height: 40,
                    borderRadius: 10,
                    background: 'rgba(62, 199, 173, 0.15)',
                    display: 'grid',
                    placeItems: 'center',
                    flexShrink: 0,
                  }}>
                    <Users size={18} color="#3ec7ad" />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <strong style={{ fontSize: 14 }}>{p.name || 'Unnamed'}</strong>
                      <span style={{
                        fontSize: 10,
                        padding: '2px 8px',
                        borderRadius: 4,
                        background: 'rgba(98, 222, 192, 0.1)',
                        color: '#62dec0',
                      }}>
                        {p.patientId}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: '#8293a7', marginTop: 2 }}>
                      {p.age ? `${p.age}y` : '—'} · {p.sex || '—'} · {p.diagnosis || 'No diagnosis'}
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deletePatientFromList(p.patientId);
                    }}
                    style={{
                      padding: '6px 10px',
                      background: 'rgba(239, 68, 68, 0.1)',
                      border: '1px solid rgba(239, 68, 68, 0.3)',
                      borderRadius: 6,
                      color: '#ef4444',
                      fontSize: 11,
                      cursor: 'pointer',
                    }}
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  function renderClinicalStep() {
    return (
      <div className="content-card" style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #1a2a39',
        borderRadius: 12,
        padding: 24,
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
          <div>
            <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
              ECOG Performance Status
            </label>
            <select
              value={patient.ecog}
              onChange={(e) => updatePatient({ ecog: e.target.value })}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid #1a2a39',
                borderRadius: 6,
                color: '#e5edf8',
                fontSize: 13,
              }}
            >
              <option value="">Select</option>
              {[0, 1, 2, 3, 4].map((v) => (
                <option key={v} value={String(v)}>{v}</option>
              ))}
            </select>
            <div style={{ marginTop: 6, padding: '8px 10px', background: 'rgba(62,199,173,0.04)', border: '1px solid rgba(62,199,173,0.1)', borderRadius: 6 }}>
              {[
                { level: '0', text: 'Fully active — no restrictions' },
                { level: '1', text: 'Light activity — restricted strenuous work' },
                { level: '2', text: 'Ambulatory — up >50% of waking hours, limited work' },
                { level: '3', text: 'Limited self-care — confined to bed/chair >50% of waking hours' },
                { level: '4', text: 'Fully disabled — no self-care, bed/chair confined' },
              ].map((item) => (
                <div key={item.level} style={{ display: 'flex', gap: 6, fontSize: 11, color: '#7a95a8', lineHeight: '1.5' }}>
                  <span style={{ color: '#3ec7ad', fontWeight: 600, minWidth: 14 }}>{item.level}:</span>
                  <span>{item.text}</span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
              Smoking History
            </label>
            <select
              value={patient.smoking}
              onChange={(e) => updatePatient({ smoking: e.target.value as any })}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid #1a2a39',
                borderRadius: 6,
                color: '#e5edf8',
                fontSize: 13,
              }}
            >
              <option value="Never">Never</option>
              <option value="Former">Former</option>
              <option value="Current">Current</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
              Diabetes
            </label>
            <select
              value={patient.diabetes}
              onChange={(e) => updatePatient({ diabetes: e.target.value as any })}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid #1a2a39',
                borderRadius: 6,
                color: '#e5edf8',
                fontSize: 13,
              }}
            >
              <option value="None">No</option>
              <option value="New-onset (<2 years)">New-onset</option>
              <option value="Long-standing (>2 years)">Long-standing</option>
            </select>
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={patient.familyHistory}
                onChange={(e) => updatePatient({ familyHistory: e.target.checked })}
                style={{ width: 'auto' }}
              />
              <span style={{ fontSize: 13 }}>Family History</span>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={patient.pancreatitis}
                onChange={(e) => updatePatient({ pancreatitis: e.target.checked })}
                style={{ width: 'auto' }}
              />
              <span style={{ fontSize: 13 }}>Pancreatitis</span>
            </label>
          </div>
        </div>

        <div style={{ marginTop: 16 }}>
          <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
            Presenting Symptoms
          </label>
          <textarea
            value={patient.symptoms}
            onChange={(e) => updatePatient({ symptoms: e.target.value })}
            placeholder="Document symptoms, duration, and relevant findings"
            rows={4}
            style={{
              width: '100%',
              padding: '12px',
              background: 'rgba(0, 0, 0, 0.2)',
              border: '1px solid #1a2a39',
              borderRadius: 6,
              color: '#e5edf8',
              fontSize: 13,
              resize: 'vertical',
            }}
          />
          <div style={{ marginTop: 6, padding: '8px 10px', background: 'rgba(62,199,173,0.04)', border: '1px solid rgba(62,199,173,0.1)', borderRadius: 6 }}>
            <span style={{ fontSize: 11, color: '#3ec7ad', fontWeight: 600 }}>Common presenting symptoms: </span>
            <span style={{ fontSize: 11, color: '#7a95a8' }}>
              Abdominal/epigastric pain radiating to back, painless jaundice, unintentional weight loss, new-onset diabetes, nausea/vomiting, anorexia, dark urine, pale stools, fatigue
            </span>
          </div>
        </div>
      </div>
    );
  }

  function renderLabsStep() {
    return (
      <div className="content-card" style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #1a2a39',
        borderRadius: 12,
        padding: 24,
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16 }}>
          {[
            { label: 'CA 19-9 (U/mL)', key: 'ca199', placeholder: '37', normal: '0-37' },
            { label: 'CEA (ng/mL)', key: 'cea', placeholder: '3.0', normal: '0-5' },
            { label: 'Bilirubin (mg/dL)', key: 'bilirubin', placeholder: '0.8', normal: '0-1.2' },
            { label: 'Albumin (g/dL)', key: 'albumin', placeholder: '4.0', normal: '3.5-5.0' },
            { label: 'Amylase (U/L)', key: 'amylase', placeholder: '95', normal: '0-110' },
            { label: 'Weight Loss (%)', key: 'weightLoss', placeholder: '0', normal: '0-5' },
          ].map((field) => (
            <div key={field.key}>
              <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
                {field.label}
              </label>
              <input
                type="number"
                value={String((patient as Record<string, unknown>)[field.key] || '')}
                onChange={(e) => updatePatient({ [field.key]: parseFloat(e.target.value) || 0 } as Partial<PatientState>)}
                placeholder={field.placeholder}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  background: 'rgba(0, 0, 0, 0.2)',
                  border: '1px solid #1a2a39',
                  borderRadius: 6,
                  color: '#e5edf8',
                  fontSize: 13,
                }}
              />
              <div style={{ fontSize: 10, color: '#607389', marginTop: 2 }}>Normal: {field.normal}</div>
            </div>
          ))}
        </div>

        {isUpdating && (
          <div style={{ marginTop: 16, padding: '12px 16px', background: 'rgba(56, 189, 248, 0.1)', border: '1px solid #38bdf8', borderRadius: 8, display: 'flex', gap: 8, alignItems: 'center' }}>
            <div style={{ width: 16, height: 16, border: '2px solid #38bdf8', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
            <span style={{ fontSize: 12, color: '#38bdf8' }}>Updating Digital Twin...</span>
          </div>
        )}
      </div>
    );
  }

  function renderImagingStep() {
    return (
      <div className="content-card" style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #1a2a39',
        borderRadius: 12,
        padding: 24,
        display: 'flex',
        flexDirection: 'column',
        gap: 20,
      }}>
        {/* Upload Container */}
        <div style={{
          border: '2px dashed #1f354d',
          borderRadius: 12,
          padding: '24px',
          background: 'rgba(0, 0, 0, 0.2)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <ImageIcon size={32} color="#3ec7ad" />
            <div>
              <div style={{ fontSize: 16, fontWeight: 600, color: '#e5edf8' }}>
                Upload Patient Medical Imaging & 3D Segmentation
              </div>
              <div style={{ fontSize: 12, color: '#8293a7' }}>
                Upload a single patient case: NIfTI CT + Label pair (.nii.gz) or DICOM series (.dcm) for VTK 3D surface extraction.
              </div>
            </div>
          </div>

          {/* Section 1: Single Patient Case Folder Upload (imagesTr + labelsTr + meshes.npz) */}
          <div style={{
            padding: 18,
            background: 'rgba(10, 18, 28, 0.85)',
            border: '1px solid #3ec7ad',
            borderRadius: 8,
            marginBottom: 16,
          }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#3ec7ad', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Layers3 size={18} /> Upload Patient Case Folder (imagesTr + labelsTr + meshes.npz)
            </div>
            <div style={{ fontSize: 12, color: '#8293a7', marginBottom: 14, lineHeight: 1.4 }}>
              Select a single patient folder containing <code>imagesTr/pancreas_XXX.nii.gz</code> (CT volume), <code>labelsTr/pancreas_XXX.nii.gz</code> (Label mask), and/or <code>meshes.npz</code>.
            </div>

            {folderUploadError && (
              <div style={{
                padding: '10px 12px', background: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid #ef4444', borderRadius: 6, color: '#fca5a5',
                fontSize: 12, marginBottom: 12,
              }}>
                ⚠️ {folderUploadError}
              </div>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <label style={{
                padding: '9px 18px',
                background: folderUploading ? 'rgba(62,199,173,0.2)' : 'linear-gradient(135deg, #3ec7ad, #62dec0)',
                border: '1px solid #54dfc4', borderRadius: 6,
                color: folderUploading ? '#607389' : '#08221f',
                fontSize: 12, fontWeight: 600, cursor: folderUploading ? 'not-allowed' : 'pointer',
                display: 'inline-flex', alignItems: 'center', gap: 6,
              }}>
                <Upload size={14} />
                {folderUploading ? 'Processing Patient Folder (VTK)...' : 'Select Patient Folder...'}
                <input
                  type="file"
                  multiple
                  // @ts-ignore
                  webkitdirectory=""
                  // @ts-ignore
                  directory=""
                  disabled={folderUploading}
                  onChange={(e) => {
                    const files = Array.from(e.target.files || []);
                    if (files.length > 0) {
                      handleUploadFolder(files);
                    }
                  }}
                  style={{ display: 'none' }}
                />
              </label>

              {folderUploading && (
                <div style={{ fontSize: 12, color: '#3ec7ad', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <RefreshCw size={14} className="animate-spin" /> Extracting CT + Label + Meshes via VTK...
                </div>
              )}
            </div>

            {folderFiles.length > 0 && !folderUploading && (
              <div style={{ marginTop: 10, fontSize: 11, color: '#62dec0', background: 'rgba(62,199,173,0.06)', padding: '6px 10px', borderRadius: 4 }}>
                ✓ Attached {folderFiles.length} file(s) from patient folder
              </div>
            )}
          </div>

          {/* Section 2: DICOM Folder Upload */}
          <div style={{ paddingTop: 12, borderTop: '1px solid #1f354d', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
            <div style={{ fontSize: 12, color: '#8293a7' }}>
              Or select DICOM folder containing .dcm slices:
            </div>
            <input
              ref={imagingInputRef}
              type="file"
              multiple
              // @ts-ignore
              webkitdirectory=""
              // @ts-ignore
              directory=""
              accept=".dcm,.nii,.nii.gz,.zip,.glb,.gltf"
              onChange={async (e) => {
                const files = e.target.files;
                if (!files || files.length === 0) return;

                const fileList = Array.from(files);
                updatePatient({
                  imagingFiles: fileList,
                  imagingStudy: fileList.map((f) => f.name).join(', '),
                  imagingStatus: `Attached ${fileList.length} file(s) - validation pending`,
                });

                const patientId = patient.patientId || defaultPatientId;
                updatePatient({ patientId });
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765/api/v1';
                const formData = new FormData();
                fileList.forEach((file) => formData.append('files', file));

                try {
                  const response = await fetch(`${apiUrl}/patients/${patientId}/imaging/studies`, {
                    method: 'POST',
                    body: formData,
                  });
                  if (!response.ok) {
                    throw new Error(`Upload failed ${response.status}`);
                  }
                  const study = await response.json();
                  updatePatient({
                    imagingStudyId: study.study_id,
                    imagingStatus: `Study uploaded (${study.file_count} files) — validating DICOM…`,
                  });
                } catch (error) {
                  console.error('Imaging upload failed', error);
                  updatePatient({ imagingStatus: 'Upload failed — retry or refresh' });
                }
              }}
              style={{ fontSize: 12 }}
            />
          </div>
        </div>

        {/* Small 3D Preview Viewport Card (Confirms Successful Upload right on this page) */}
        {patient.uploadedPayload && (
          <div style={{
            background: 'rgba(10, 18, 28, 0.9)',
            border: '1px solid rgba(62, 199, 173, 0.3)',
            borderRadius: 12,
            padding: 20,
            boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Check size={18} color="#3ec7ad" />
                <span style={{ fontSize: 14, fontWeight: 600, color: '#3ec7ad' }}>
                  ✓ Uploaded Patient 3D Anatomy Ready (VTK Pipeline)
                </span>
              </div>
              <span style={{ fontSize: 11, color: '#8293a7', background: 'rgba(0,0,0,0.4)', padding: '2px 10px', borderRadius: 4 }}>
                {patient.imagingStudy || 'Single Uploaded Patient Case'}
              </span>
            </div>

            {/* Small 3D Preview Viewport (Pure 1:1 Original Uploaded Patient Anatomy) */}
            <div style={{ width: '100%', height: 340, borderRadius: 8, overflow: 'hidden', border: '1px solid #1f354d', marginBottom: 16 }}>
              <TwinMeshViewer directPayload={patient.uploadedPayload} />
            </div>

            {/* Volume Diagnostics & Metrics */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, fontSize: 11, background: 'rgba(0,0,0,0.3)', padding: 12, borderRadius: 8 }}>
              <div>
                <span style={{ color: '#607389', display: 'block' }}>Pancreas Volume</span>
                <span style={{ color: '#e0a468', fontWeight: 600, fontSize: 13 }}>
                  {patient.uploadedPayload.meta?.pancreas_diagnostics?.volume_cm3 ?? '—'} cm³
                </span>
              </div>
              <div>
                <span style={{ color: '#607389', display: 'block' }}>Cancer / Tumor Volume</span>
                <span style={{ color: '#ef565f', fontWeight: 600, fontSize: 13 }}>
                  {patient.uploadedPayload.meta?.tumor_diagnostics?.volume_cm3 ?? '—'} cm³
                </span>
              </div>
              <div>
                <span style={{ color: '#607389', display: 'block' }}>Pancreas 3D Mesh</span>
                <span style={{ color: '#b0c8d8', fontWeight: 500 }}>
                  {patient.uploadedPayload.meta?.pancreas_diagnostics?.vertices ?? 0} points
                </span>
              </div>
              <div>
                <span style={{ color: '#607389', display: 'block' }}>Cancer 3D Mesh</span>
                <span style={{ color: '#b0c8d8', fontWeight: 500 }}>
                  {patient.uploadedPayload.meta?.tumor_diagnostics?.vertices ?? 0} points
                </span>
              </div>
            </div>

            <div style={{ marginTop: 12, fontSize: 11, color: '#62dec0', background: 'rgba(62,199,173,0.08)', padding: '8px 12px', borderRadius: 6, border: '1px solid rgba(62,199,173,0.2)' }}>
              🔒 <strong>State Preserved:</strong> This uploaded patient anatomy is active across all workflow steps (Clinical, Lab Results, AI Analysis, Clinical Digital Twin, Progression, Simulation, Report). Navigating away and returning will NOT reset or close this model.
            </div>
          </div>
        )}
      </div>
    );
  }

  function renderGenomicsStep() {
    return (
      <div className="content-card" style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #1a2a39',
        borderRadius: 12,
        padding: 24,
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16 }}>
          {[
            { label: 'KRAS', key: 'kras' },
            { label: 'TP53', key: 'tp53' },
            { label: 'SMAD4', key: 'smad4' },
            { label: 'ERBB2', key: 'erbb2' },
          ].map((field) => (
            <div key={field.key}>
              <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
                {field.label}
              </label>
              <select
                value={String((patient as Record<string, unknown>)[field.key] || '')}
                onChange={(e) => updatePatient({ [field.key]: e.target.value } as Partial<PatientState>)}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  background: 'rgba(0, 0, 0, 0.2)',
                  border: '1px solid #1a2a39',
                  borderRadius: 6,
                  color: '#e5edf8',
                  fontSize: 13,
                }}
              >
                <option value="">Select</option>
                <option value="Mutated">Mutated</option>
                <option value="Wildtype">Wildtype</option>
                <option value="Altered">Altered</option>
                <option value="Intact">Intact</option>
                <option value="Unknown">Unknown</option>
              </select>
            </div>
          ))}
        </div>

        <div style={{ marginTop: 16 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={patient.geneticMutation}
              onChange={(e) => updatePatient({ geneticMutation: e.target.checked })}
              style={{ width: 'auto' }}
            />
            <span style={{ fontSize: 13 }}>Known germline mutation (e.g. BRCA1/2, PALB2)</span>
          </label>
        </div>
      </div>
    );
  }

  function renderValidationStep() {
    const validation = [
      { label: 'Patient identity', valid: Boolean(patient.name && patient.patientId), message: 'Name and ID required' },
      { label: 'Clinical assessment', valid: Boolean(patient.ecog || patient.symptoms), message: 'ECOG or symptoms required' },
      { label: 'Laboratory results', valid: Boolean(patient.ca199 > 0 || patient.bilirubin > 0), message: 'At minimum CA 19-9 required' },
      { label: 'Imaging study', valid: Boolean(patient.imagingStudy), message: 'Attach a study' },
      { label: 'Genomic profile', valid: Boolean(patient.kras), message: 'At minimum KRAS status required' },
    ];

    return (
      <div className="content-card" style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #1a2a39',
        borderRadius: 12,
        padding: 24,
      }}>
        <div style={{ display: 'grid', gap: 8 }}>
          {validation.map((item) => (
            <div key={item.label} style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              padding: '12px 16px',
              background: item.valid ? 'rgba(34, 197, 94, 0.1)' : 'rgba(245, 158, 11, 0.1)',
              border: `1px solid ${item.valid ? '#22c55e' : '#f59e0b'}`,
              borderRadius: 8,
            }}>
              {item.valid ? <Check size={17} color="#22c55e" /> : <AlertTriangle size={17} color="#f59e0b" />}
              <span style={{ fontSize: 13, fontWeight: 500 }}>{item.label}</span>
              <small style={{ flex: 1, color: '#7a9aa5', fontSize: 11 }}>{item.message}</small>
              <strong style={{ fontSize: 11, color: item.valid ? '#22c55e' : '#f59e0b' }}>
                {item.valid ? 'Ready' : 'Required'}
              </strong>
            </div>
          ))}
        </div>
      </div>
    );
  }

  function renderAnalysisStep() {
    const surv = computeSurvivalProjection(prediction);
    const biomarkers = patient
      ? [
          { name: 'CA 19-9 (U/mL)', value: patient.ca199 ?? 0, threshold: 37 },
          { name: 'CEA (ng/mL)', value: patient.cea ?? 0, threshold: 5 },
          { name: 'Bilirubin (mg/dL)', value: patient.bilirubin ?? 0, threshold: 1.2 },
          { name: 'Weight Loss (%)', value: patient.weightLoss ?? 0, threshold: 5 },
        ]
      : [];
    const riskFactors: string[] = [];
    if (patient) {
      if (patient.ca199 > 1000) riskFactors.push(`CA 19-9 markedly elevated (${patient.ca199.toFixed(0)})`);
      else if (patient.ca199 > 500) riskFactors.push(`CA 19-9 elevated (${patient.ca199.toFixed(0)})`);
      else if (patient.ca199 > 100) riskFactors.push(`CA 19-9 elevated (${patient.ca199.toFixed(0)})`);
      if (patient.cea > 20) riskFactors.push(`CEA elevated (${patient.cea.toFixed(1)})`);
      else if (patient.cea > 5) riskFactors.push(`CEA elevated (${patient.cea.toFixed(1)})`);
      if (patient.bilirubin > 5) riskFactors.push('Hyperbilirubinaemia');
      else if (patient.bilirubin > 2) riskFactors.push('Elevated bilirubin');
      if (patient.weightLoss > 15) riskFactors.push('Severe weight loss');
      else if (patient.weightLoss > 10) riskFactors.push('Moderate weight loss');
      if (patient.smoking === 'Current') riskFactors.push('Current smoker');
      else if (patient.smoking === 'Former') riskFactors.push('Former smoker');
      if (patient.diabetes.includes('New-onset') || patient.diabetes.includes('Type 2')) riskFactors.push(`Diabetes (${patient.diabetes})`);
      if (patient.familyHistory) riskFactors.push('Family history of PDAC');
      if (patient.pancreatitis) riskFactors.push('Chronic pancreatitis');
      if (patient.geneticMutation) riskFactors.push('BRCA1/2 or PALB2 variant');
    }
    return (
      <div className="content-card" style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #1a2a39',
        borderRadius: 12,
        padding: 24,
      }}>

        {!prediction && (
          <div style={{ marginTop: 24, fontSize: 13, color: '#607389' }}>
            Run the AI analysis to populate the prediction outcome below.
          </div>
        )}

        {prediction && (
          <div style={{ marginTop: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* Risk banner */}
            {prediction.probability > 50 ? (
              <div className="res-hi" style={{ padding: '14px 20px', borderRadius: 10 }}>
                <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f87171' }}>HIGH RISK DETECTED</div>
                <div style={{ fontSize: '0.8rem', opacity: 0.9, marginTop: 2 }}>
                  P(Cancer) = <b>{prediction.probability.toFixed(1)}%</b> | Stage: <b style={{ color: prediction.stageColour }}>{prediction.stage}</b> | Risk: <b>{surv.riskGroup}</b> | Recommended: <b style={{ color: '#34d399' }}>{prediction.bestTreatment}</b>
                </div>
              </div>
            ) : (
              <div className="res-lo" style={{ padding: '14px 20px', borderRadius: 10 }}>
                <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#34d399' }}>LOW RISK - MONITORING RECOMMENDED</div>
                <div style={{ fontSize: '0.8rem', opacity: 0.9, marginTop: 2 }}>
                  P(Cancer) = <b>{prediction.probability.toFixed(1)}%</b> | Stage: <b style={{ color: prediction.stageColour }}>{prediction.stage}</b> | Risk: <b>{surv.riskGroup}</b>
                </div>
              </div>
            )}

            {/* Row 1: Gauge | Biomarker Status | Risk Factors | Key Metrics */}
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr 1fr', gap: 16 }}>
              <div style={{ padding: 16, background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>Model Confidence</div>
                <svg width="100%" viewBox="0 0 160 130" style={{ maxWidth: 150, margin: '0 auto', display: 'block' }}>
                  <circle cx="80" cy="70" r="54" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="12" />
                  <circle
                    cx="80" cy="70" r="54" fill="none"
                    stroke="url(#gaugeGrad)" strokeWidth="12"
                    strokeDasharray={`${Math.min(1, prediction.confidence) * 2 * Math.PI * 54} ${2 * Math.PI * 54}`}
                    strokeLinecap="round"
                    transform="rotate(-90 80 70)"
                  />
                  <defs>
                    <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop stopColor="#34d399" />
                      <stop offset="0.5" stopColor="#fbbf24" />
                      <stop offset="1" stopColor="#ef4444" />
                    </linearGradient>
                  </defs>
                  <text x="80" y="78" textAnchor="middle" fill="#e5edf8" fontSize="22" fontWeight={700}>{Math.round(prediction.confidence * 100)}%</text>
                  <text x="80" y="96" textAnchor="middle" fill="#94a3b8" fontSize="10">Confidence</text>
                </svg>
                <div style={{ fontSize: 10, color: '#607389', marginTop: 4, textAlign: 'center' }}>
                  {prediction.confidence > 0.8 ? 'Very High' : prediction.confidence > 0.5 ? 'Moderate' : 'Low'} confidence
                </div>
              </div>

              <div style={{ padding: 16, background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 }}>Biomarker Status</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {biomarkers.map((b) => {
                    const hi = b.value > b.threshold;
                    return (
                      <div key={b.name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: 11, color: '#cbd5e1' }}>{b.name}</span>
                        <span style={{
                          fontSize: 10, fontWeight: 700,
                          color: hi ? '#f87171' : '#34d399',
                        }}>{hi ? 'ELEVATED' : 'NORMAL'}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div style={{ padding: 16, background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>Risk Factors</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 150, overflow: 'auto' }}>
                  {riskFactors.length > 0 ? riskFactors.map((f) => (
                    <div key={f} style={{ fontSize: 10, color: '#fbbf24', padding: '2px 0' }}>[!] {f}</div>
                  )) : (
                    <div style={{ fontSize: 10, color: '#34d399' }}>[OK] No major risk factors</div>
                  )}
                </div>
              </div>

              <div style={{ padding: 16, background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 }}>Key Metrics</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 11 }}>
                  <span style={{ color: '#607389' }}>Risk Score</span><span style={{ color: '#62dec0', fontWeight: 600 }}>{surv.riskScore.toFixed(3)}</span>
                  <span style={{ color: '#607389' }}>Risk Group</span><span style={{ color: prediction.stageColour, fontWeight: 600 }}>{surv.riskGroup}</span>
                  <span style={{ color: '#607389' }}>Median Survival</span><span style={{ color: '#e5edf8', fontWeight: 600 }}>{surv.median} months</span>
                  <span style={{ color: '#607389' }}>1-Year Survival</span><span style={{ color: '#34d399', fontWeight: 600 }}>{(surv.p1y * 100).toFixed(0)}%</span>
                  <span style={{ color: '#607389' }}>2-Year Survival</span><span style={{ color: surv.p2y > 0.3 ? '#34d399' : '#f87171', fontWeight: 600 }}>{(surv.p2y * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>

            {/* Row 2: SHAP | Treatment Response */}
            <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16 }}>
              <div style={{ padding: 16, background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 }}>SHAP Feature Contributions</div>
                <div style={{ fontSize: 10, color: '#607389', marginBottom: 6 }}>Red pushes toward cancer. Blue toward normal.</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {Object.entries(prediction.shapValues).map(([feature, value]) => {
                    const v = Number(value);
                    const isPositive = v > 0;
                    const maxAbs = Math.max(...Object.values(prediction.shapValues).map((x) => Math.abs(Number(x))), 1e-9);
                    const barWidth = (Math.abs(v) / maxAbs) * 100;
                    return (
                      <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontSize: 10, color: '#8293a7', minWidth: 110, textTransform: 'capitalize' }}>{feature}</span>
                        <div style={{ flex: 1, height: 16, background: 'rgba(255,255,255,0.03)', borderRadius: 4, overflow: 'hidden', position: 'relative' }}>
                          <div style={{
                            width: `${barWidth}%`, height: '100%',
                            background: isPositive ? 'linear-gradient(90deg, #ef4444, #dc2626)' : 'linear-gradient(90deg, #22c55e, #16a34a)',
                            borderRadius: 4, transition: 'width 0.3s ease',
                          }} />
                        </div>
                        <span style={{ fontSize: 10, fontWeight: 600, color: isPositive ? '#ef4444' : '#22c55e', minWidth: 50, textAlign: 'right' }}>
                          {isPositive ? '+' : ''}{v.toFixed(3)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div style={{ padding: 16, background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 }}>Treatment Response</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {Object.entries(prediction.treatmentData).map(([treatment, response]) => {
                    const respPct = Number(response) || 0;
                    const isBest = treatment === prediction.bestTreatment;
                    return (
                      <div key={treatment} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ fontSize: 11, color: isBest ? '#3ec7ad' : '#e5edf8', fontWeight: isBest ? 600 : 400 }}>{treatment}</span>
                          <span style={{ fontSize: 10, color: '#62dec0', fontWeight: 600 }}>{respPct}%</span>
                        </div>
                        <div style={{ width: '100%', height: 8, background: 'rgba(255,255,255,0.05)', borderRadius: 4, overflow: 'hidden' }}>
                          <div style={{
                            width: `${respPct}%`, height: '100%',
                            background: isBest ? 'linear-gradient(90deg, #62dec0, #3ec7ad)' : 'rgba(100,150,200,0.5)',
                            borderRadius: 4,
                          }} />
                        </div>
                        {isBest && <span style={{ fontSize: 9, color: '#3ec7ad', fontWeight: 700 }}>RECOMMENDED</span>}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Row 3: Survival Curve | Organ Radar */}
            <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: 16 }}>
              <div style={{ padding: 16, background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 }}>Survival Projection</div>
                <svg width="100%" height="150" viewBox="0 0 360 150" style={{ maxWidth: '100%' }}>
                  <line x1="30" y1="125" x2="330" y2="125" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                  <line x1="30" y1="20" x2="30" y2="125" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                  {[0, 25, 50, 75, 100].map((p) => (
                    <text key={p} x="6" y={125 - (p / 100) * 105} fill="#607389" fontSize="9">{p}%</text>
                  ))}
                  {surv.pts.map((p) => (
                    <text key={p.t} x={30 + (p.t / 36) * 300} y="138" fill="#607389" fontSize="9" textAnchor="middle">{p.t}m</text>
                  ))}
                  <polyline
                    points={surv.pts.map((p) => `${30 + (p.t / 36) * 300},${125 - p.s * 105}`).join(' ')}
                    fill="none" stroke="#34d399" strokeWidth="2.5"
                    strokeLinejoin="round" strokeLinecap="round"
                  />
                  {surv.pts.map((p) => (
                    <circle key={p.t} cx={30 + (p.t / 36) * 300} cy={125 - p.s * 105} r="3" fill="#34d399" />
                  ))}
                  <text x="330" y="18" fill="#62dec0" fontSize="11" textAnchor="end">Median: {surv.median} mo</text>
                </svg>
              </div>

              <div style={{ padding: 16, background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 }}>Organ Risk Radar</div>
                <OrganRadar risks={prediction.organRisks} />
              </div>
            </div>

            {/* Row 4: Tables */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div style={{ padding: 16, background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 }}>Survival by Timepoint</div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr>
                      {[{ t: 'Timepoint' }, { t: 'Survival' }, { t: 'Risk Group' }].map((h) => (
                        <th key={h.t} style={{ textAlign: 'left', color: '#607389', padding: '4px 0' }}>{h.t}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {surv.pts.map((p) => (
                      <tr key={p.t} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        <td style={{ color: '#8293a7', padding: '3px 0' }}>{p.t}m</td>
                        <td style={{ color: '#34d399', fontWeight: 600, padding: '3px 0' }}>{(p.s * 100).toFixed(1)}%</td>
                        <td style={{ color: '#94a3b8', padding: '3px 0' }}>{surv.riskGroup}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div style={{ padding: 16, background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 }}>Organ Risk Summary</div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr>
                      {['Organ', 'Risk', 'Concern'].map((h) => (
                        <th key={h} style={{ textAlign: 'left', color: '#607389', padding: '4px 0' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(prediction.organRisks).map(([organ, data]) => {
                      const d = data as [number, string, string];
                      return (
                        <tr key={organ} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ color: '#8293a7', padding: '3px 0', textTransform: 'capitalize' }}>{organ}</td>
                          <td style={{ color: d[0] > 60 ? '#ef4444' : d[0] > 30 ? '#f59e0b' : '#34d399', fontWeight: 600, padding: '3px 0' }}>{d[0]}%</td>
                          <td style={{ color: '#94a3b8', padding: '3px 0', fontSize: 10 }}>{d[2]}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  function renderTwinStep() {
    return (
      <div style={{
        height: '100%',
        display: 'flex',
        gap: 16,
        overflow: 'hidden',
      }}>
        <div style={{ flex: 2, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ flex: 1, borderRadius: 12, overflow: 'hidden', border: '1px solid #1a2a39' }}>
            <ClinicalTwinWorkspace
              patient={patient}
              imagingSegmentation={patient.imagingSegmentation}
              twinView={twinView}
              ctCase={ctCase}
              ctVolume={ctVolume}
              ctVolumeLoading={ctVolumeLoading}
              ctVolumeError={ctVolumeError}
              panelOpen={twinPanelOpen}
              biomarkers={biomarkers}
            />
          </div>
        </div>
      </div>
    );
  }

  function renderProgressionStep() {
    return (
      <div className="content-card" style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #1a2a39',
        borderRadius: 12,
        padding: 24,
      }}>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 24 }}>
          <label style={{ fontSize: 12, color: '#8293a7' }}>Time Horizon:</label>
          <select
            value={timeHorizon}
            onChange={(e) => setTimeHorizon(Number(e.target.value))}
            style={{
              padding: '8px 12px',
              background: 'rgba(0, 0, 0, 0.2)',
              border: '1px solid #1a2a39',
              borderRadius: 6,
              color: '#e5edf8',
              fontSize: 13,
            }}
          >
            <option value={3}>3 months</option>
            <option value={6}>6 months</option>
            <option value={12}>12 months</option>
            <option value={24}>24 months</option>
          </select>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 16 }}>
          <div style={{ padding: '16px', background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
            <small style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase' }}>CURRENT STAGE</small>
            <strong style={{ fontSize: 18, display: 'block', marginTop: 4 }}>{prediction?.stage || '—'}</strong>
          </div>
          <div style={{ padding: '16px', background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
            <small style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase' }}>RISK SCORE</small>
            <strong style={{ fontSize: 18, display: 'block', marginTop: 4 }}>{prediction ? `${prediction.risk.toFixed(0)}` : '—'}</strong>
          </div>
          <div style={{ padding: '16px', background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
            <small style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase' }}>EST. SURVIVAL</small>
            <strong style={{ fontSize: 18, display: 'block', marginTop: 4 }}>{prediction ? `${prediction.survival} mo` : '—'}</strong>
          </div>
          <div style={{ padding: '16px', background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
            <small style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase' }}>TUMOR VOLUME</small>
            <strong style={{ fontSize: 18, display: 'block', marginTop: 4 }}>{organs.Pancreas ? `${Math.round(organs.Pancreas.tumorBurden / 20)} cm³` : '—'}</strong>
          </div>
        </div>

        <div style={{
          marginTop: 24,
          padding: '16px',
          background: 'rgba(56, 189, 248, 0.1)',
          border: '1px solid #38bdf8',
          borderRadius: 8,
          display: 'flex',
          gap: 12,
          alignItems: 'center',
        }}>
          <BarChart3 size={20} color="#38bdf8" />
          <span style={{ fontSize: 13, color: '#38bdf8' }}>
            Run a treatment simulation to see how progression changes with intervention.
          </span>
        </div>
      </div>
    );
  }

  function renderSimulationStep() {
    return (
      <div className="content-card" style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #1a2a39',
        borderRadius: 12,
        padding: 24,
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 24 }}>
          <div>
            <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
              Treatment Scenario
            </label>
            <select
              value={treatmentScenario}
              onChange={(e) => setTreatmentScenario(e.target.value)}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid #1a2a39',
                borderRadius: 6,
                color: '#e5edf8',
                fontSize: 13,
              }}
            >
              <option>FOLFIRINOX</option>
              <option>Gemcitabine + nab-paclitaxel</option>
              <option>FOLFOX</option>
              <option>Radiotherapy</option>
              <option>Targeted therapy</option>
              <option>Immunotherapy</option>
              <option>Whipple surgery</option>
              <option>Best Supportive Care</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
              Time Horizon
            </label>
            <select
              value={timeHorizon}
              onChange={(e) => setTimeHorizon(Number(e.target.value))}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid #1a2a39',
                borderRadius: 6,
                color: '#e5edf8',
                fontSize: 13,
              }}
            >
              <option value={3}>3 months</option>
              <option value={6}>6 months</option>
              <option value={12}>12 months</option>
              <option value={24}>24 months</option>
            </select>
          </div>
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          padding: '16px',
          background: 'rgba(0, 0, 0, 0.1)',
          borderRadius: 8,
          border: '1px solid #1a2a39',
        }}>
          <Play size={24} color="#62dec0" />
          <div>
            <strong style={{ fontSize: 14 }}>Run Simulation</strong>
            <div style={{ fontSize: 12, color: '#8293a7' }}>Simulates tumor volume, biomarker response, and organ risk changes.</div>
          </div>
          <button
            onClick={handleSimulate}
            disabled={isSimulating || !prediction}
            style={{
              padding: '10px 24px',
              background: 'linear-gradient(135deg, #62dec0, #3ec7ad)',
              border: '1px solid #54dfc4',
              borderRadius: 6,
              color: '#08221f',
              fontSize: 13,
              fontWeight: 600,
              cursor: isSimulating || !prediction ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              marginLeft: 'auto',
            }}
          >
            <Play size={14} /> {isSimulating ? 'Running…' : 'Simulate'}
          </button>
        </div>

        {simulationResult && (
          <div style={{
            marginTop: 20,
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: 16,
          }}>
            <div style={{ padding: '16px', background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
              <small style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase' }}>FINAL VOLUME</small>
              <strong style={{ fontSize: 18, display: 'block', marginTop: 4 }}>{simulationResult.tumorVolume.toFixed(1)} cm³</strong>
            </div>
            <div style={{ padding: '16px', background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
              <small style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase' }}>RESPONSE</small>
              <strong style={{ fontSize: 18, display: 'block', marginTop: 4 }}>{simulationResult.responseRate.toFixed(1)}%</strong>
            </div>
            <div style={{ padding: '16px', background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
              <small style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase' }}>SURVIVAL PROB</small>
              <strong style={{ fontSize: 18, display: 'block', marginTop: 4 }}>{(simulationResult.survivalProbability * 100).toFixed(1)}%</strong>
            </div>
            <div style={{ padding: '16px', background: 'rgba(0, 0, 0, 0.1)', borderRadius: 8, border: '1px solid #1a2a39' }}>
              <small style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase' }}>TREATMENT</small>
              <strong style={{ fontSize: 14, display: 'block', marginTop: 4 }}>{simulationResult.treatment}</strong>
            </div>
          </div>
        )}
      </div>
    );
  }

  function renderComparisonStep() {
    return (
      <div className="content-card" style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #1a2a39',
        borderRadius: 12,
        padding: 24,
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 24 }}>
          <div>
            <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
              Treatment A
            </label>
            <select
              value={comparisonTreatments[0] || 'FOLFIRINOX'}
              onChange={(e) => setComparisonTreatments([e.target.value, comparisonTreatments[1] || 'Gemcitabine + nab-paclitaxel'])}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid #1a2a39',
                borderRadius: 6,
                color: '#e5edf8',
                fontSize: 13,
              }}
            >
              <option>FOLFIRINOX</option>
              <option>Gemcitabine + nab-paclitaxel</option>
              <option>FOLFOX</option>
              <option>Best Supportive Care</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: 11, color: '#607389', textTransform: 'uppercase', marginBottom: 4, display: 'block' }}>
              Treatment B
            </label>
            <select
              value={comparisonTreatments[1] || 'Gemcitabine + nab-paclitaxel'}
              onChange={(e) => setComparisonTreatments([comparisonTreatments[0] || 'FOLFIRINOX', e.target.value])}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid #1a2a39',
                borderRadius: 6,
                color: '#e5edf8',
                fontSize: 13,
              }}
            >
              <option>Gemcitabine + nab-paclitaxel</option>
              <option>FOLFIRINOX</option>
              <option>FOLFOX</option>
              <option>Best Supportive Care</option>
            </select>
          </div>
        </div>

        <button
          onClick={handleCompareTreatments}
          disabled={!prediction}
          style={{
            padding: '10px 24px',
            background: 'linear-gradient(135deg, #62dec0, #3ec7ad)',
            border: '1px solid #54dfc4',
            borderRadius: 6,
            color: '#08221f',
            fontSize: 13,
            fontWeight: 600,
            cursor: !prediction ? 'not-allowed' : 'pointer',
            marginBottom: 16,
          }}
        >
          Compare Treatments
        </button>

        {comparisonResult && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {Object.entries(comparisonResult).map(([tx, result]) => {
              const r = result as SimulationResult;
              return (
                <div key={tx} style={{
                  padding: '16px',
                  background: 'rgba(0, 0, 0, 0.1)',
                  borderRadius: 8,
                  border: '1px solid #1a2a39',
                }}>
                  <h3 style={{ fontSize: 14, margin: '0 0 12px', fontWeight: 600 }}>{tx}</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    <div><small style={{ fontSize: 10, color: '#607389' }}>Volume</small><strong style={{ fontSize: 14, display: 'block' }}>{r.tumorVolume.toFixed(1)} cm³</strong></div>
                    <div><small style={{ fontSize: 10, color: '#607389' }}>Response</small><strong style={{ fontSize: 14, display: 'block' }}>{r.responseRate.toFixed(1)}%</strong></div>
                    <div><small style={{ fontSize: 10, color: '#607389' }}>Survival</small><strong style={{ fontSize: 14, display: 'block' }}>{(r.survivalProbability * 100).toFixed(1)}%</strong></div>
                    <div><small style={{ fontSize: 10, color: '#607389' }}>Volume Δ</small><strong style={{ fontSize: 14, display: 'block' }}>{r.tumorVolumeChange > 0 ? '+' : ''}{r.tumorVolumeChange.toFixed(2)} cm³</strong></div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  function renderReportStep() {
    return (
      <div className="content-card" style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #1a2a39',
        borderRadius: 12,
        padding: 24,
      }}>
        {prediction ? (
          <div style={{
            padding: '24px',
            background: 'rgba(0, 0, 0, 0.1)',
            borderRadius: 8,
            border: '1px solid #1a2a39',
            fontFamily: 'monospace',
            fontSize: 12,
            lineHeight: 1.6,
            whiteSpace: 'pre-wrap',
            overflow: 'auto',
            maxHeight: 400,
          }}>
            EXAI-PANC CLINICAL REPORT
            ========================================

            Patient: {patient.name || 'Unknown'}
            Patient ID: {patient.patientId || 'N/A'}
            Date: {new Date().toISOString().split('T')[0]}

            DIAGNOSIS SUMMARY
            ------------------------------------------
            Cancer Probability: {prediction.probability.toFixed(1)}%
            AJCC Stage: {prediction.stage}
            Stage Detail: {prediction.stageDetail}
            Confidence: {(prediction.confidence * 100).toFixed(1)}%

            SURVIVAL ANALYSIS
            ------------------------------------------
            Median Survival: {prediction.survival} months
            Risk Score: {prediction.risk.toFixed(2)}
            Risk Group: {String((prediction.survivalData as Record<string, unknown>)?.risk_group || 'Unknown')}

            TREATMENT RECOMMENDATION
            ------------------------------------------
            Best Treatment: {prediction.bestTreatment}
            Response Probability: {String((prediction.treatmentData as Record<string, unknown>)?.[prediction.bestTreatment] ?? 0)}%

            Organ Risks:
            {Object.entries(prediction.organRisks || {}).map(([organ, data]) => {
              const d = data as [number, string, string];
              return `  ${organ}: ${d[0]}% (${d[2]})`;
            }).join('\n')}

            SHAP EXPLANATIONS
            ------------------------------------------
            {xaiExplanations.map((exp: XAIExplanation) =>
              `${exp.feature}: ${exp.contribution === 'positive' ? '+' : ''}${exp.shapValue.toFixed(4)} (${exp.clinicalSignificance})`
            ).join('\n')}

            ========================================
            Report generated by ExAI-PanC v2.0.0
            Research use only. Not for clinical decisions.
          </div>
        ) : (
          <div style={{
            padding: '40px',
            textAlign: 'center',
            color: '#8293a7',
            border: '1px dashed #1a2a39',
            borderRadius: 8,
          }}>
            <FileDown size={48} color="#3a6a7a" />
            <p style={{ marginTop: 16, fontSize: 14 }}>No prediction available. Run AI analysis first.</p>
          </div>
        )}

        <div style={{ marginTop: 16, display: 'flex', gap: 12 }}>
          <button
            style={{
              padding: '10px 20px',
              background: 'rgba(62, 199, 173, 0.15)',
              border: '1px solid #3ec7ad',
              borderRadius: 6,
              color: '#3ec7ad',
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            <Download size={14} /> Download TXT
          </button>
          <button
            style={{
              padding: '10px 20px',
              background: 'rgba(62, 199, 173, 0.15)',
              border: '1px solid #3ec7ad',
              borderRadius: 6,
              color: '#3ec7ad',
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            <Download size={14} /> Download CSV
          </button>
        </div>
      </div>
    );
  }

  function renderExportStep() {
    return (
      <div className="content-card" style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #1a2a39',
        borderRadius: 12,
        padding: 24,
      }}>
        <div style={{ display: 'grid', gap: 12 }}>
          <button style={{
            padding: '16px',
            background: 'rgba(0, 0, 0, 0.1)',
            border: '1px solid #1a2a39',
            borderRadius: 8,
            color: '#e5edf8',
            fontSize: 14,
            cursor: 'pointer',
            textAlign: 'left',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <FileDown size={16} />
              <strong>Clinical Report (TXT)</strong>
            </div>
            <small style={{ color: '#8293a7' }}>Full clinical report with predictions and recommendations</small>
          </button>
          <button style={{
            padding: '16px',
            background: 'rgba(0, 0, 0, 0.1)',
            border: '1px solid #1a2a39',
            borderRadius: 8,
            color: '#e5edf8',
            fontSize: 14,
            cursor: 'pointer',
            textAlign: 'left',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Download size={16} />
              <strong>Organ Risks (CSV)</strong>
            </div>
            <small style={{ color: '#8293a7' }}>Organ risk scores and health metrics</small>
          </button>
          <button style={{
            padding: '16px',
            background: 'rgba(0, 0, 0, 0.1)',
            border: '1px solid #1a2a39',
            borderRadius: 8,
            color: '#e5edf8',
            fontSize: 14,
            cursor: 'pointer',
            textAlign: 'left',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Share2 size={16} />
              <strong>Digital Twin State (JSON)</strong>
            </div>
            <small style={{ color: '#8293a7' }}>Complete twin state for research sharing</small>
          </button>
        </div>
      </div>
    );
  }

  function renderBiomarkerForm() {
    return (
      <div style={{
        border: '1px solid #1a2a39', borderRadius: 8,
        background: 'rgba(255,255,255,0.02)', overflow: 'hidden',
      }}>
        <button onClick={() => setTwinShowBiomarkers((v) => !v)} style={{
          width: '100%', display: 'flex', alignItems: 'center', gap: 8, padding: '9px 12px',
          background: 'transparent', border: 'none', cursor: 'pointer', textAlign: 'left',
          color: '#e5edf8',
        }}>
          <SlidersHorizontal size={14} color="#3ec7ad" />
          <span style={{ flex: 1, fontSize: 12, fontWeight: 600 }}>Patient & Biomarker Inputs</span>
          {twinShowBiomarkers ? <ChevronUp size={14} color="#607389" /> : <ChevronDown size={14} color="#607389" />}
        </button>
        {twinShowBiomarkers && (
          <div style={{ padding: '0 12px 12px' }}>
            <div style={{ fontSize: 10, color: '#3ec7ad', margin: '2px 0 8px' }}>
              Live — tumor updates as you type
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {([
                ['name', 'Patient Name'],
                ['age', 'Age'],
                ['sex', 'Sex'],
                ['ca199', 'CA 19-9'],
                ['cea', 'CEA'],
                ['bilirubin', 'Bilirubin'],
                ['albumin', 'Albumin'],
                ['weight_loss_pct', 'Weight Loss %'],
              ] as const).map(([key, label]) => (
                <label key={key} style={{ fontSize: 10, color: '#607389' }}>
                  {label}
                  {key === 'sex' ? (
                    <select
                      value={biomarkers[key] as string}
                      onChange={(e) => updateBiomarker(key as keyof BiomarkerForm, e.target.value)}
                      style={{
                        width: '100%', padding: '5px 6px', fontSize: 11, marginTop: 2,
                        background: 'rgba(0,0,0,0.3)', border: '1px solid #152235',
                        borderRadius: 4, color: '#e5edf8',
                      }}
                    >
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                    </select>
                  ) : (
                    <input
                      type={key === 'age' ? 'number' : key === 'name' ? 'text' : 'number'}
                      step={key === 'age' ? '1' : '0.1'}
                      value={biomarkers[key] as string | number}
                      onChange={(e) => updateBiomarker(key as keyof BiomarkerForm, e.target.value)}
                      style={{
                        width: '100%', padding: '5px 6px', fontSize: 11, marginTop: 2,
                        background: 'rgba(0,0,0,0.3)', border: '1px solid #152235',
                        borderRadius: 4, color: '#e5edf8',
                      }}
                    />
                  )}
                </label>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 10, marginTop: 10, flexWrap: 'wrap' }}>
              {(['kras', 'tp53', 'smad4'] as const).map((k) => (
                <label key={k} style={{ fontSize: 10, display: 'flex', alignItems: 'center', gap: 4 }}>
                  <input
                    type="checkbox"
                    checked={biomarkers[k] as boolean}
                    onChange={(e) => updateBiomarker(k as keyof BiomarkerForm, e.target.checked)}
                  />
                  {k.toUpperCase()} mutation
                </label>
              ))}
            </div>
            <button
              onClick={runBiomarkerPrediction}
              disabled={predictionRunning}
              style={{
                marginTop: 12, width: '100%', padding: '7px 0', fontSize: 11, fontWeight: 600,
                background: predictionRunning ? 'rgba(255,255,255,0.05)' : 'linear-gradient(135deg,#3ec7ad,#62dec0)',
                border: '1px solid #54dfc4', borderRadius: 6,
                color: predictionRunning ? '#607389' : '#08221f', cursor: predictionRunning ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
              }}
            >
              <FlaskConical size={11} /> {predictionRunning ? 'Running…' : 'Run Clinical Prediction'}
            </button>
            {biomarkerNotice && (
              <div style={{
                marginTop: 8, padding: '6px 8px', background: 'rgba(0,0,0,0.3)', borderRadius: 4,
                fontSize: 10, color: '#b7f6e4', border: '1px solid rgba(62,199,173,0.3)',
              }}>
                {biomarkerNotice}
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  function renderPatientPanel() {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{
          padding: '16px',
          background: 'rgba(0, 0, 0, 0.1)',
          borderRadius: 8,
          border: '1px solid #1a2a39',
        }}>
          <h3 style={{ fontSize: 14, margin: '0 0 12px', fontWeight: 600 }}>Patient Summary</h3>
          <div style={{ display: 'grid', gap: 8 }}>
            <div><small style={{ fontSize: 10, color: '#607389' }}>NAME</small><strong style={{ fontSize: 14, display: 'block' }}>{patient.name || '—'}</strong></div>
            <div><small style={{ fontSize: 10, color: '#607389' }}>AGE / SEX</small><strong style={{ fontSize: 14, display: 'block' }}>{patient.age || '—'} / {patient.sex || '—'}</strong></div>
            <div><small style={{ fontSize: 10, color: '#607389' }}>DIAGNOSIS</small><strong style={{ fontSize: 12, display: 'block' }}>{patient.diagnosis}</strong></div>
          </div>
        </div>

        {prediction && (
          <div style={{
            padding: '16px',
            background: 'rgba(0, 0, 0, 0.1)',
            borderRadius: 8,
            border: '1px solid #1a2a39',
          }}>
            <h3 style={{ fontSize: 14, margin: '0 0 12px', fontWeight: 600 }}>AI Findings</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <div><small style={{ fontSize: 10, color: '#607389' }}>PROBABILITY</small><strong style={{ fontSize: 18, color: prediction.stageColour, display: 'block' }}>{prediction.probability.toFixed(1)}%</strong></div>
              <div><small style={{ fontSize: 10, color: '#607389' }}>STAGE</small><strong style={{ fontSize: 16, display: 'block' }}>{prediction.stage}</strong></div>
              <div><small style={{ fontSize: 10, color: '#607389' }}>BEST TREATMENT</small><strong style={{ fontSize: 14, display: 'block' }}>{prediction.bestTreatment}</strong></div>
            </div>
          </div>
        )}
      </div>
    );
  }

  function renderXAIPanel() {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{
          padding: '16px',
          background: 'rgba(0, 0, 0, 0.1)',
          borderRadius: 8,
          border: '1px solid #1a2a39',
        }}>
          <h3 style={{ fontSize: 14, margin: '0 0 12px', fontWeight: 600 }}>SHAP Explanations</h3>
          <div style={{ display: 'grid', gap: 8 }}>
            {xaiExplanations.map((exp: XAIExplanation) => (
              <div key={exp.feature} style={{
                padding: '8px 12px',
                background: 'rgba(255, 255, 255, 0.03)',
                borderRadius: 6,
                borderLeft: `3px solid ${exp.contribution === 'positive' ? '#ef4444' : '#22c55e'}`,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 12, fontWeight: 500 }}>{exp.feature}</span>
                  <span style={{ fontSize: 11, color: exp.contribution === 'positive' ? '#ef4444' : '#22c55e' }}>
                    {exp.contribution === 'positive' ? '+' : ''}{exp.shapValue.toFixed(4)}
                  </span>
                </div>
                <small style={{ fontSize: 10, color: '#8293a7' }}>{exp.clinicalSignificance}</small>
              </div>
            ))}
          </div>
        </div>

        {clinicalFindings.length > 0 && (
          <div style={{
            padding: '16px',
            background: 'rgba(0, 0, 0, 0.1)',
            borderRadius: 8,
            border: '1px solid #1a2a39',
          }}>
            <h3 style={{ fontSize: 14, margin: '0 0 12px', fontWeight: 600 }}>Clinical Findings</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              {clinicalFindings.slice(0, 3).map((finding: ClinicalFinding) => (
                <div key={finding.id} style={{
                  padding: '10px 14px',
                  background: 'rgba(255, 255, 255, 0.03)',
                  borderRadius: 6,
                  border: '1px solid #1a2a39',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                    <strong style={{ fontSize: 12 }}>{finding.title}</strong>
                    <span style={{ fontSize: 10, color: '#607389' }}>{finding.category}</span>
                  </div>
                  <small style={{ fontSize: 10, color: '#8293a7', display: 'block', marginBottom: 4 }}>{finding.description}</small>
                  <small style={{ fontSize: 9, color: '#607389' }}>{finding.recommendation}</small>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  function renderTreatmentPanel() {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{
          padding: '16px',
          background: 'rgba(0, 0, 0, 0.1)',
          borderRadius: 8,
          border: '1px solid #1a2a39',
        }}>
          <h3 style={{ fontSize: 14, margin: '0 0 12px', fontWeight: 600 }}>Treatment Options</h3>
          <div style={{ display: 'grid', gap: 8 }}>
            {treatments.map((tx: TreatmentOption) => (
              <div key={tx.id} style={{
                padding: '10px 14px',
                background: 'rgba(255, 255, 255, 0.03)',
                borderRadius: 6,
                border: selectedTreatment === tx.name ? '1px solid #3ec7ad' : '1px solid #1a2a39',
                cursor: 'pointer',
              }} onClick={() => setSelectedTreatment(tx.name)}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <strong style={{ fontSize: 12 }}>{tx.name}</strong>
                  <span style={{ fontSize: 11, color: '#62dec0' }}>{(tx.responseProbability * 100).toFixed(1)}%</span>
                </div>
                <small style={{ fontSize: 10, color: '#8293a7', display: 'block' }}>{tx.description}</small>
              </div>
            ))}
          </div>
        </div>

        {simulationResult && (
          <div style={{
            padding: '16px',
            background: 'rgba(0, 0, 0, 0.1)',
            borderRadius: 8,
            border: '1px solid #1a2a39',
          }}>
            <h3 style={{ fontSize: 14, margin: '0 0 12px', fontWeight: 600 }}>Latest Simulation</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <div><small style={{ fontSize: 10, color: '#607389' }}>TUMOR VOLUME</small><strong style={{ fontSize: 16, display: 'block' }}>{simulationResult.tumorVolume.toFixed(1)} cm³</strong></div>
              <div><small style={{ fontSize: 10, color: '#607389' }}>RESPONSE</small><strong style={{ fontSize: 16, display: 'block' }}>{simulationResult.responseRate.toFixed(1)}%</strong></div>
              <div><small style={{ fontSize: 10, color: '#607389' }}>SURVIVAL</small><strong style={{ fontSize: 16, display: 'block' }}>{(simulationResult.survivalProbability * 100).toFixed(1)}%</strong></div>
            </div>
          </div>
        )}
      </div>
    );
  }

  function renderTimelinePanel() {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <h3 style={{ fontSize: 14, margin: 0, fontWeight: 600 }}>Patient Timeline</h3>
        {timeline.length === 0 ? (
          <div style={{ padding: '20px', textAlign: 'center', color: '#607389', fontSize: 12 }}>
            No timeline events yet. Start by entering patient data.
          </div>
        ) : (
          <div style={{ display: 'grid', gap: 8 }}>
            {timeline.slice().reverse().map((event: TimelineEvent) => (
              <div key={event.id} style={{
                padding: '10px 14px',
                background: 'rgba(255, 255, 255, 0.03)',
                borderRadius: 6,
                borderLeft: `3px solid ${event.color}`,
                border: '1px solid #1a2a39',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                  <strong style={{ fontSize: 12 }}>{event.title}</strong>
                  <small style={{ fontSize: 9, color: '#607389' }}>
                    {new Date(event.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </small>
                </div>
                <small style={{ fontSize: 10, color: '#8293a7' }}>{event.description}</small>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  function renderSettingsPanel() {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div>
          <h3 style={{ fontSize: 14, margin: '0 0 12px', fontWeight: 600 }}>Theme</h3>
          <div style={{ display: 'grid', gap: 8 }}>
            {(['dark', 'light', 'clinical'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTheme(t)}
                style={{
                  padding: '10px 14px',
                  background: theme === t ? 'rgba(62, 199, 173, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                  border: theme === t ? '1px solid #3ec7ad' : '1px solid #1a2a39',
                  borderRadius: 6,
                  color: theme === t ? '#3ec7ad' : '#8293a7',
                  fontSize: 12,
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div>
          <h3 style={{ fontSize: 14, margin: '0 0 12px', fontWeight: 600 }}>Render Mode</h3>
          <div style={{ display: 'grid', gap: 8 }}>
            {(['realistic', 'heatmap', 'xray', 'wireframe'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setRenderMode(mode)}
                style={{
                  padding: '10px 14px',
                  background: renderMode === mode ? 'rgba(62, 199, 173, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                  border: renderMode === mode ? '1px solid #3ec7ad' : '1px solid #1a2a39',
                  borderRadius: 6,
                  color: renderMode === mode ? '#3ec7ad' : '#8293a7',
                  fontSize: 12,
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div>
          <h3 style={{ fontSize: 14, margin: '0 0 12px', fontWeight: 600 }}>Visualization</h3>
          <div style={{ display: 'grid', gap: 8 }}>
            <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}>
              <span style={{ fontSize: 12 }}>Show Annotations</span>
              <input
                type="checkbox"
                checked={showAnnotations}
                onChange={(e) => setShowAnnotations(e.target.checked)}
                style={{ width: 'auto' }}
              />
            </label>
            <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}>
              <span style={{ fontSize: 12 }}>Show Metastasis Paths</span>
              <input
                type="checkbox"
                checked={showMetastasisPaths}
                onChange={(e) => setShowMetastasisPaths(e.target.checked)}
                style={{ width: 'auto' }}
              />
            </label>
            <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}>
              <span style={{ fontSize: 12 }}>Show Biomarker Timeline</span>
              <input
                type="checkbox"
                checked={showBiomarkerTimeline}
                onChange={(e) => setShowBiomarkerTimeline(e.target.checked)}
                style={{ width: 'auto' }}
              />
            </label>
          </div>
        </div>
      </div>
    );
  }
}
