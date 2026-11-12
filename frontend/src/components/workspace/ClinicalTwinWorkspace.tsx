'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Play, Heart, X } from 'lucide-react';
import { type TumorConfig } from '../../engine/imaging/syntheticModelEngine';
import type { PatientState } from '../../types';
import { TwinMeshViewer } from './TwinMeshViewer';
import { CtVolumeViewer, type CtVolumePayload } from './CtVolumeViewer';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765/api/v1';
const ORGAN_COLORS: Record<string, string> = {
  Pancreas: '#ef565f', Liver: '#f4a524', Gallbladder: '#5b8c5a', Stomach: '#c48a7c',
  Duodenum: '#b8957a', Spleen: '#6b3fa0', 'Portal Vein': '#3b6ea5',
  'Superior Mesenteric Vein': '#4a7fb5', 'Superior Mesenteric Artery': '#b53a3a',
  'Common Bile Duct': '#8ab87a', Aorta: '#c23535', 'Inferior Vena Cava': '#4a6fa5',
  'Regional Lymph Nodes': '#c9a87c', Kidneys: '#8b5e3c', Lungs: '#c4958a', Heart: '#8b3a3a',
};

type ReferenceModel = 'organs' | 'pancreas';
const REFERENCE_MODELS: Record<ReferenceModel, { id: string; label: string; author: string; href: string }> = {
   organs: {
    id: 'f9a34dba04624c28a3131970fcac4b81',
    label: 'Adult liver, spleen & pancreas',
    author: 'Education Resource Fund',
    href: 'https://sketchfab.com/3d-models/adult-liver-spleen-and-pancreas-f9a34dba04624c28a3131970fcac4b81',
  },
  pancreas: {
    id: 'b34ba50e38fd455bae43bd07b7ed2d3b',
    label: 'Pancreas and duodenum',
    author: 'Ebers',
    href: 'https://sketchfab.com/3d-models/pancreas-and-duodenum-b34ba50e38fd455bae43bd07b7ed2d3b',
  },
};

/**
 * Screen-anchor positions (percent of the default Sketchfab view, model facing
 * front) used to overlay patient-specific tumour / risk markers onto the
 * *reference* anatomy. Exact 3-D co-registration is not possible on a generic
 * healthy model from biomarkers alone, so these are anatomically-anchored
 * organ-level markers that update live with the biomarker state.
 */
const MARKER_POS: Record<ReferenceModel, Record<string, { x: number; y: number }>> = {
  organs: {
    Pancreas:            { x: 50, y: 60 },
    Liver:               { x: 73, y: 44 },
    Spleen:              { x: 27, y: 46 },
    'Regional Lymph Nodes': { x: 50, y: 70 },
  },
  pancreas: {
    Pancreas:            { x: 50, y: 52 },
    Duodenum:            { x: 59, y: 64 },
    'Regional Lymph Nodes': { x: 50, y: 72 },
  },
};

function useBreakpoint(px: number) {
  const [match, setMatch] = useState(false);
  useEffect(() => {
    const q = window.matchMedia(`(min-width: ${px}px)`);
    setMatch(q.matches);
    const h = (e: MediaQueryListEvent) => setMatch(e.matches);
    q.addEventListener('change', h);
    return () => q.removeEventListener('change', h);
  }, [px]);
  return match;
}

export interface BiomarkerForm {
  name: string; age: number; sex: string; ca199: number; cea: number;
  bilirubin: number; albumin: number; weight_loss_pct: number; kras: boolean;
  tp53: boolean; smad4: boolean;
}

export function biomarkersFromPatient(p: PatientState): BiomarkerForm {
  return {
    name: p.name || 'PDAC-0427 Demo Patient',
    age: p.age || 63,
    sex: p.sex || 'Female',
    ca199: p.ca199 || 824,
    cea: p.cea || 18.4,
    bilirubin: p.bilirubin || 2.8,
    albumin: p.albumin || 3.1,
    weight_loss_pct: p.weightLoss || 9,
    kras: p.kras === 'Mutated',
    tp53: p.tp53 === 'Mutated',
    smad4: p.smad4 === 'Altered',
  };
}

interface ImagingSegmentation {
  tumour_size_cm: number;
  organ_masks: Record<string, boolean>;
  quality_score: number;
  segmentation_summary: string;
}

interface ClinicalTwinWorkspaceProps {
  patient?: PatientState;
  imagingSegmentation?: ImagingSegmentation;
   twinView?: 'realct' | 'ct' | 'library' | 'vtk-ct';
  ctCase?: string;
  ctVolume?: CtVolumePayload | null;
  ctVolumeLoading?: boolean;
  ctVolumeError?: string | null;
  panelOpen?: boolean;
  biomarkers: BiomarkerForm;
}
function healthColor(health: number): string {
  if (health > 80) return '#22c55e';
  if (health > 60) return '#eab308';
  if (health > 40) return '#f97316';
  return '#ef4444';
}

interface OrganHealthData {
  health: number;
  risk: number;
  burden: number;
  inflammation: number;
  fibrosis: number;
  metastasisRisk: number;
  invasion: number;
  status: string;
}

function computeOrganHealth(name: string, prob: number, biomarkers: BiomarkerForm): OrganHealthData {
  const base = 100 - prob * 40;
  const ca199 = biomarkers.ca199 || 37;
  const bili = biomarkers.bilirubin || 0.8;
  const wt = biomarkers.weight_loss_pct || 0;

  const organModifiers: Record<string, { healthOff: number; riskMult: number; desc: string }> = {
    Pancreas:  { healthOff: 0, riskMult: 1, desc: 'Primary tumour site' },
    Liver:     { healthOff: -5 - bili * 3, riskMult: 0.7, desc: 'Biliary obstruction / metastasis risk' },
    Stomach:   { healthOff: -3 - wt * 0.5, riskMult: 0.4, desc: 'Anatomical compression risk' },
    Duodenum:  { healthOff: -8 - prob * 15, riskMult: 0.5, desc: 'Anatomical invasion risk' },
    Spleen:    { healthOff: -2, riskMult: 0.3, desc: 'Splenic vein involvement' },
    Gallbladder: { healthOff: -3 - bili * 2, riskMult: 0.3, desc: 'Biliary stasis risk' },
    'Portal Vein': { healthOff: -10 - prob * 20, riskMult: 0.6, desc: 'Vascular invasion risk' },
    'Superior Mesenteric Vein': { healthOff: -10 - prob * 20, riskMult: 0.6, desc: 'SMV invasion risk' },
    'Superior Mesenteric Artery': { healthOff: -5 - prob * 15, riskMult: 0.5, desc: 'SMA involvement risk' },
    'Common Bile Duct': { healthOff: -10 - bili * 5, riskMult: 0.5, desc: 'Obstructive jaundice risk' },
    Aorta:     { healthOff: -2 - prob * 5, riskMult: 0.2, desc: 'Retroperitoneal invasion risk' },
    'Inferior Vena Cava': { healthOff: -2 - prob * 5, riskMult: 0.2, desc: 'IVC compression risk' },
    'Regional Lymph Nodes': { healthOff: -5 - prob * 25, riskMult: 0.7, desc: 'Regional lymphatic spread' },
    Kidneys:   { healthOff: -2, riskMult: 0.2, desc: 'Hydronephrosis risk' },
  };
  const mod = organModifiers[name] || { healthOff: 0, riskMult: 0.1, desc: '' };
  const health = Math.max(5, Math.min(100, base + mod.healthOff));
  const risk = Math.min(98, Math.round(prob * 100 * mod.riskMult + (ca199 / 100) * 0.1 + (wt) * 0.5));
  return {
    health: Math.round(health),
    risk, burden: Math.round((100 - health) * 1.2),
    inflammation: Math.min(100, Math.round(prob * 60 + (ca199 > 500 ? 20 : 0))),
    fibrosis: Math.min(100, Math.round(prob * 30 + (biomarkers.tp53 ? 15 : 0))),
    metastasisRisk: Math.min(100, Math.round(prob * 80 * mod.riskMult * 1.5)),
    invasion: Math.min(100, Math.round(prob * 90 * mod.riskMult * 1.2)),
    status: risk > 60 ? 'Critical' : risk > 30 ? 'At risk' : 'Stable',
  };
}

function computeTumorConfig(biomarkers: BiomarkerForm, prob: number, month: number, overrideSizeCm: number = 0): TumorConfig {
  // 1. Calculate biomarker severity modifier from all 8 clinical parameters:
  const ca199Factor = (biomarkers.ca199 || 37) / 500; // e.g. 37->0.07, 500->1.0, 2000->4.0
  const ceaFactor = (biomarkers.cea || 2.5) / 10; // e.g. 2.5->0.25, 20->2.0
  const biliFactor = Math.max(0, (biomarkers.bilirubin || 0.8) - 0.8) * 0.2; // e.g. 0.8->0, 5.0->0.84
  const albFactor = Math.max(0, (3.5 - (biomarkers.albumin || 3.5))) * 0.25; // e.g. 3.5->0, 2.0->0.37
  const wtFactor = (biomarkers.weight_loss_pct || 0) * 0.03; // e.g. 0%->0, 15%->0.45

  // Genomic Driver Mutations
  const krasBonus = biomarkers.kras ? 0.35 : 0;
  const tp53Bonus = biomarkers.tp53 ? 0.30 : 0;
  const smad4Bonus = biomarkers.smad4 ? 0.25 : 0;

  const bioSeverity = 0.85 + (ca199Factor * 0.4 + ceaFactor * 0.2 + biliFactor + albFactor + wtFactor + krasBonus + tp53Bonus + smad4Bonus);

  // Month progression factor
  const monthGrowth = 1 + month * (0.05 + prob * 0.08);

  const baselineCm = overrideSizeCm > 0 ? overrideSizeCm : 2.0;
  const sizeCm = Math.max(0.4, Math.min(8.5, baselineCm * bioSeverity * monthGrowth));

  return {
    tumorLocation: 'body',
    tumorLengthMm: sizeCm * 10,
    tumorWidthMm: sizeCm * 10 * 0.8,
    tumorHeightMm: sizeCm * 10 * 0.9,
    tumorShape: 'irregular',
    progression: prob,
    invasionLevel: Math.min(1, (prob + month * 0.02)),
    vascularInvolvement: prob * 0.8 + month * 0.015 > 0.3,
    sizeCm: Math.round(sizeCm * 10) / 10,
    position: [0.05, -0.08, 0.08],
    invasionDepth: Math.min(1, (prob + month * 0.02)),
    necrosisPct: Math.min(0.8, prob * 0.3 + month * 0.01),
    vascularInvasion: Math.min(1, prob * 0.8 + month * 0.015),
    glowIntensity: Math.min(1, prob * 0.6 + sizeCm / 10),
    spatialInfiltrationRadius: Math.round(Math.min(8, Math.max(0.5, sizeCm)) * 10) / 10,
  };
}

export function predictFromBiomarkers(b: BiomarkerForm) {
  const ageScore = Math.max(0, (b.age - 40) / 80) * 0.2 + (b.age >= 70 ? 0.05 : b.age >= 60 ? 0.02 : 0);
  let bioScore = 0;
  if (b.ca199 > 1000) bioScore += 0.2; else if (b.ca199 > 500) bioScore += 0.15; else if (b.ca199 > 100) bioScore += 0.1; else if (b.ca199 > 37) bioScore += 0.05;
  if (b.cea > 20) bioScore += 0.06; else if (b.cea > 5) bioScore += 0.03;
  if (b.bilirubin > 5) bioScore += 0.08; else if (b.bilirubin > 2) bioScore += 0.04;
  if (b.weight_loss_pct > 15) bioScore += 0.08; else if (b.weight_loss_pct > 10) bioScore += 0.05; else if (b.weight_loss_pct > 5) bioScore += 0.02;
  const prob = Math.min(ageScore + bioScore, 0.97);
  let stage = 'Non-malignant', stageCol = '#3498db', stageDetail = 'No strong malignancy evidence';
  if (prob > 0.8 && b.ca199 > 500) { stage = 'Stage IV'; stageCol = '#e74c3c'; stageDetail = 'Distant metastasis likely'; }
  else if (prob > 0.65 && b.ca199 > 200) { stage = 'Stage III'; stageCol = '#e67e22'; stageDetail = 'Locally advanced'; }
  else if (prob > 0.5 && b.ca199 > 100) { stage = 'Stage II'; stageCol = '#f39c12'; stageDetail = 'Regional spread'; }
  else if (prob > 0.35) { stage = 'Stage I'; stageCol = '#2ecc71'; stageDetail = 'Localised'; }

  const baseHazard = 1 + (stage.includes('III') || stage.includes('IV') ? 1.5 : stage.includes('II') ? 0.5 : 0) + prob * 2 + Math.max(0, (b.age - 60) / 40);
  const survival = Math.max(3, Math.round(36 - baseHazard * 5));
  const tumor = computeTumorConfig(b, prob, 0);

  const organHealth: Record<string, OrganHealthData> = {};
  for (const name of Object.keys(ORGAN_COLORS)) {
    organHealth[name] = computeOrganHealth(name, prob, b);
  }

  return { prob, stage, stageCol, stageDetail, survival, tumor, organHealth };
}

/* ============================================================
   ClinicalTwinWorkspace — Main Component
   ============================================================ */

export function ClinicalTwinWorkspace({
  patient, imagingSegmentation, twinView = 'vtk-ct', ctCase = '409',
  ctVolume = null, ctVolumeLoading = false, ctVolumeError = null,
  panelOpen = true, biomarkers,
}: ClinicalTwinWorkspaceProps) {
  const [selectedOrgan, setSelectedOrgan] = useState<string | null>(null);
  const [month, setMonth] = useState(0);
  const [simTreatment, setSimTreatment] = useState('FOLFIRINOX');
  const [isSimulating, setIsSimulating] = useState(false);
  const [notice, setNotice] = useState('');
  const [refModel, setRefModel] = useState<ReferenceModel>('organs');
  const [modelOpen, setModelOpen] = useState(true);

  // NIfTI Pair Upload state
  const [niftiModalOpen, setNiftiModalOpen] = useState(false);
  const [ctFile, setCtFile] = useState<File | null>(null);
  const [labelFile, setLabelFile] = useState<File | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadedPayload, setUploadedPayload] = useState<any>(null);

  const handleGenerate3D = async () => {
    if (!ctFile || !labelFile) {
      setUploadError('Please select both a CT Volume (.nii/.nii.gz) and a matching Segmentation Label (.nii/.nii.gz)');
      return;
    }
    setUploadLoading(true);
    setUploadError(null);
    const formData = new FormData();
    formData.append('ct_file', ctFile);
    formData.append('label_file', labelFile);

    try {
      const res = await fetch(`${API}/twin/mesh/upload-nifti`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed with status ${res.status}`);
      }
      const payload = await res.json();
      setUploadedPayload(payload);
      setNiftiModalOpen(false);
      setNotice('NIfTI pair successfully processed by VTK!');
    } catch (err: any) {
      setUploadError(err.message || 'Error processing NIfTI pair with VTK');
    } finally {
      setUploadLoading(false);
    }
  };

  const [activeBiomarkers, setActiveBiomarkers] = useState<BiomarkerForm>(biomarkers);

  useEffect(() => {
    setActiveBiomarkers(biomarkers);
  }, [biomarkers]);

  const prob = useMemo(() => predictFromBiomarkers(activeBiomarkers).prob, [activeBiomarkers]);
  const stage = useMemo(() => predictFromBiomarkers(activeBiomarkers), [activeBiomarkers]);
  const organHealth = useMemo(() => {
    const h: Record<string, OrganHealthData> = {};
    for (const name of Object.keys(ORGAN_COLORS)) {
      h[name] = computeOrganHealth(name, prob, activeBiomarkers);
    }
    return h;
  }, [activeBiomarkers, prob]);

  const simulate = useCallback(() => {
    setIsSimulating(true);
    const steps = month > 0 ? month : 6;
    let step = 0;
    const interval = setInterval(() => {
      step++;
      setMonth(step);
      if (step >= steps) {
        clearInterval(interval);
        setIsSimulating(false);
        setNotice(`Simulation complete: ${simTreatment} over ${steps} months`);
      }
    }, 400);
    return () => clearInterval(interval);
  }, [month, simTreatment]);

  const selectedHealth = selectedOrgan ? organHealth[selectedOrgan] : null;
  const segmentationOverride = imagingSegmentation?.tumour_size_cm || 0;

  const config = useMemo(() => computeTumorConfig(activeBiomarkers, prob, month, segmentationOverride), [activeBiomarkers, prob, month, segmentationOverride]);

  type Marker = { organ: string; x: number; y: number; risk: number; size: number; label: string; color: string; kind: 'tumor' | 'risk' };
  const markers = useMemo<Marker[]>(() => {
    const map = MARKER_POS[refModel];
    const out: Marker[] = [];
    for (const [organ, pos] of Object.entries(map)) {
      const h = organHealth[organ];
      if (!h) continue;
      if (organ === 'Pancreas') {
        const sizeCm = Math.max(0.16, Math.min(3.2, config.sizeCm));
        out.push({
          organ, x: pos.x, y: pos.y, risk: h.risk,
          size: Math.round(10 + sizeCm * 5),
          label: `Pancreas tumour ${sizeCm.toFixed(1)} cm`,
          color: '#ef565f', kind: 'tumor',
        });
      } else if (h.risk >= 35) {
        out.push({
          organ, x: pos.x, y: pos.y, risk: h.risk,
          size: Math.round(8 + h.risk / 14),
          label: `${organ} involvement ${h.risk}%`,
          color: h.risk > 70 ? '#ef565f' : '#f59e0b', kind: 'risk',
        });
      }
    }
    return out;
  }, [refModel, organHealth, config.sizeCm]);

  const isWide = useBreakpoint(1200);

  const treatments = ['FOLFIRINOX', 'Gemcitabine + nab-paclitaxel', 'FOLFOX', 'Best Supportive Care', 'No Treatment'];

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%', width: '100%',
      background: '#080d14', color: '#e5edf8', overflow: 'hidden',
    }}>
      {/* Main layout */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* 3D Anatomy Viewport */}
        <div style={{
          flex: 1, position: 'relative', minWidth: 0,
          borderRight: isWide ? '1px solid #152235' : 'none',
        }}>
          {twinView === 'vtk-ct' ? (
            <div style={{ width: '100%', height: '100%', minHeight: 520 }}>
              <TwinMeshViewer
                url={patient?.uploadedPayload || uploadedPayload ? undefined : `${API}/twin/mesh/vtk-ct?case=${encodeURIComponent(ctCase || '414')}`}
                directPayload={patient?.uploadedPayload || uploadedPayload || undefined}
                biomarkerTumorCm={config.sizeCm}
                prob={prob}
              />
            </div>
          ) : twinView === 'realct' ? (
            <div style={{ width: '100%', height: '100%', minHeight: 520 }}>
              <TwinMeshViewer
                url={`${API}/twin/mesh/real-ct?case=${encodeURIComponent(ctCase || '414')}`}
                biomarkerTumorCm={config.sizeCm}
                prob={prob}
              />
            </div>
          ) : twinView === 'ct' ? (
            <div style={{ width: '100%', height: '100%', minHeight: 520 }}>
              {ctVolume ? (
                <CtVolumeViewer
                  payload={ctVolume}
                  config={config}
                  prob={prob}
                  biomarkers={biomarkers}
                  month={month}
                />
              ) : (
                <div style={{
                  width: '100%', height: '100%', minHeight: 520, display: 'grid', placeItems: 'center',
                  background: 'radial-gradient(circle at top, rgba(20,38,54,0.92), rgba(7,13,20,0.98) 60%, rgba(4,8,12,1) 100%)',
                  color: '#8293a7', fontSize: 13, textAlign: 'center', padding: 24,
                }}>
                  {ctVolumeLoading
                    ? 'Loading patient CT DICOM volume…'
                    : ctVolumeError
                      ? `CT unavailable — ${ctVolumeError}`
                      : 'No patient CT loaded. Use "Load CT DICOM…" in the header to upload a DICOM series, or pick an MSD case.'}
                </div>
              )}
            </div>
          ) : (
             <div style={{ width: '100%', height: '100%', minHeight: 520, position: 'relative', borderRadius: 8, overflow: 'hidden', background: '#071018' }}>
               {modelOpen ? (
                 <>
                <div key={REFERENCE_MODELS[refModel].id} style={{ width: '100%', height: '100%' }}>
                  <iframe
                    title={`Reference Anatomy — ${REFERENCE_MODELS[refModel].label}`}
                    allow="autoplay; fullscreen; xr-spatial-tracking"
                    allowFullScreen
                    frameBorder="0"
                     src={`https://sketchfab.com/models/${REFERENCE_MODELS[refModel].id}/embed?autostart=1`}
                     style={{ width: '100%', height: '100%', border: 0, display: 'block' }}
                  />
                </div>
                {/* Patient-specific tumour / organ-risk overlay (live with biomarkers) */}
                <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 3 }}>
                  {markers.map((m) => (
                    <div
                      key={m.organ}
                      title={m.label}
                      style={{
                        position: 'absolute', left: `${m.x}%`, top: `${m.y}%`,
                        transform: 'translate(-50%, -50%)', pointerEvents: 'none',
                      }}
                    >
                      <div style={{
                        width: m.size, height: m.size, borderRadius: '50%',
                        background: m.color,
                        boxShadow: `0 0 ${m.size / 1.5}px 3px ${m.color}`,
                        border: '2px solid rgba(255,255,255,0.85)',
                        opacity: m.kind === 'tumor' ? 0.92 : 0.76,
                      }} />
                      <div style={{
                        position: 'absolute', top: '125%', left: '50%', transform: 'translateX(-50%)',
                        fontSize: 9, color: '#ffd5b3',
                        background: 'rgba(0,0,0,0.65)', padding: '2px 6px', borderRadius: 4, whiteSpace: 'nowrap',
                      }}>{m.risk}%</div>
                    </div>
                  ))}
                  <div style={{
                    position: 'absolute', top: 8, right: 8,
                    background: 'rgba(0,0,0,0.65)', borderRadius: 8, padding: '6px 8px', pointerEvents: 'none',
                  }}>
                    <div style={{ fontSize: 10, color: '#8293a7', marginBottom: 3, display: 'flex', gap: 10 }}>
                      <span style={{ color: '#ef565f' }}>● Pancreas tumour ({config.sizeCm} cm)</span>
                      <span style={{ color: '#f59e0b' }}>● Organ risk ≥35%</span>
                    </div>
                    <div style={{ fontSize: 9, color: '#607389' }}>
                      {markers.filter((m) => m.organ !== 'Pancreas').length > 0
                        ? `At-risk regions: ${markers.filter((m) => m.organ !== 'Pancreas').map((m) => `${m.organ} (${m.risk}%)`).join(', ')}`
                        : 'No metastatic regions predicted'}
                    </div>
                  </div>
                </div>
               <div style={{
                 position: 'absolute', top: 8, left: 8, right: 8,
                 display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 4, alignItems: 'center',
               }}>
                <div style={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                  <span style={{
                    fontSize: 10, color: '#607389', background: 'rgba(0,0,0,0.6)',
                    padding: '3px 8px', borderRadius: 5, whiteSpace: 'nowrap',
                  }}>
                    Anatomy Library
                  </span>
                  {(['organs', 'pancreas'] as ReferenceModel[]).map((m) => (
                    <button
                      key={m}
                      onClick={() => setRefModel(m)}
                      style={{
                        fontSize: 10, padding: '4px 8px', borderRadius: 5,
                        background: refModel === m ? 'rgba(62,199,173,0.2)' : 'rgba(0,0,0,0.55)',
                        border: refModel === m ? '1px solid #3ec7ad' : '1px solid #1a2a39',
                        color: refModel === m ? '#3ec7ad' : '#8293a7', cursor: 'pointer',
                        display: 'inline-flex', alignItems: 'center', gap: 4,
                      }}
                    >
                      {REFERENCE_MODELS[m].label}
                    </button>
                  ))}
                  <button
                    onClick={() => setModelOpen(false)}
                    title="Close reference model view"
                    style={{
                      fontSize: 10, padding: '4px 8px', borderRadius: 5,
                      background: 'rgba(239,86,89,0.12)', border: '1px solid #ef565f',
                      color: '#ef565f', cursor: 'pointer',
                      display: 'inline-flex', alignItems: 'center', gap: 4,
                    }}
                  >
                    <X size={13} /> Close view
                  </button>
                 </div>
                </div>
                <div style={{
                  position: 'absolute', bottom: 8, left: 8, right: 8,
                  display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 4, alignItems: 'center',
                }}>
                  <span style={{
                    fontSize: 10, color: '#607389', background: 'rgba(0,0,0,0.6)',
                    padding: '2px 8px', borderRadius: 4,
                  }}>
                    <a href={REFERENCE_MODELS[refModel].href} target="_blank" rel="noreferrer"
                      style={{ color: '#62dec0', whiteSpace: 'nowrap' }}>
                      {REFERENCE_MODELS[refModel].label}
                    </a> · by {REFERENCE_MODELS[refModel].author} (CC BY Sketchfab)
                  </span>
                  <span style={{
                    fontSize: 10, color: '#ff9d5c', background: 'rgba(0,0,0,0.6)',
                    padding: '3px 8px', borderRadius: 4,
                  }}>
                    Stage {stage.stage} · risk {Math.round(prob * 100)}% · CA19-9 {biomarkers.ca199} · {biomarkers.kras ? 'KRASmut' : 'KRASwt'}
                  </span>
                </div>
                 </>
               ) : (
                  <div style={{
                    width: '100%', height: '100%', minHeight: 520,
                    display: 'grid', placeItems: 'center',
                    background: 'radial-gradient(circle at top, rgba(20,38,54,0.92), rgba(7,13,20,0.98) 60%, rgba(4,8,12,1) 100%)',
                    color: '#8293a7',
                  }}>
                   <div style={{ textAlign: 'center' }}>
                     <span style={{ fontSize: 11, color: '#607389' }}>Reference model closed</span>
                     <button
                       onClick={() => setModelOpen(true)}
                       style={{
                         marginTop: 12, padding: '8px 16px', fontSize: 11,
                         background: 'rgba(62,199,173,0.2)', border: '1px solid #3ec7ad',
                         borderRadius: 6, color: '#3ec7ad', cursor: 'pointer',
                       }}
                     >
                       Reopen reference model
                     </button>
                   </div>
                 </div>
               )}
             </div>
           )}

          {/* Overlay labels & NIfTI Upload Button */}
          <div style={{
            position: 'absolute', bottom: 12, left: '50%', transform: 'translateX(-50%)',
            display: 'flex', gap: 8, alignItems: 'center', zIndex: 10, flexWrap: 'wrap',
            justifyContent: 'center', maxWidth: '92%',
          }}>
              <span style={{ fontSize: 10, color: '#607389', background: 'rgba(0,0,0,0.6)', padding: '3px 8px', borderRadius: 4 }}>
                {twinView === 'vtk-ct'
                  ? `VTK CT Twin — MSD case ${ctCase || '409'} (VTK pipeline, patient segmentation)`
                  : twinView === 'realct'
                    ? `Real CT Twin — MSD case ${ctCase || '409'} (real patient CT meshes)`
                    : twinView === 'ct'
                      ? 'Patient CT — uploaded DICOM volume, live biomarker tumor impact'
                      : 'Anatomy Library — Sketchfab organ reference'}
              </span>

              <button
                onClick={() => setNiftiModalOpen(true)}
                style={{
                  fontSize: 10, padding: '3px 10px', borderRadius: 4,
                  background: 'rgba(62, 199, 173, 0.25)', border: '1px solid #3ec7ad',
                  color: '#3ec7ad', cursor: 'pointer', fontWeight: 600,
                  display: 'inline-flex', alignItems: 'center', gap: 4,
                }}
              >
                Upload NIfTI Pair (.nii.gz)
              </button>
            {twinView !== 'realct' && twinView !== 'vtk-ct' && (
              <span style={{
                fontSize: 10, color: '#ef565f', background: 'rgba(0,0,0,0.6)',
                padding: '3px 8px', borderRadius: 4, fontWeight: 600,
              }}>
                Tumor {config.sizeCm} cm
              </span>
            )}
            {month > 0 && (
              <span style={{
                fontSize: 10, color: '#f59e0b', background: 'rgba(0,0,0,0.6)',
                padding: '3px 8px', borderRadius: 4, fontWeight: 600,
              }}>
                Simulation: Month {month}
              </span>
            )}
          </div>

          {notice && (
            <div style={{
              position: 'absolute', top: 8, left: '50%', transform: 'translateX(-50%)',
              padding: '6px 16px', background: 'rgba(0,0,0,0.8)', borderRadius: 8,
              border: '1px solid rgba(62,199,173,0.3)', fontSize: 11, color: '#b7f6e4',
              zIndex: 10,
            }}>
              {notice}
            </div>
          )}
        </div>

        {/* Right Panel */}
        {panelOpen && (
          <div style={{
            width: isWide ? 340 : 300, flexShrink: 0, display: 'flex', flexDirection: 'column',
            background: 'rgba(10, 18, 28, 0.95)', overflow: 'hidden',
          }}>
            <div style={{ flex: 1, overflow: 'auto', padding: 12 }}>

              {/* Selected Organ Detail */}
              {selectedOrgan && selectedHealth && (
                <div style={{
                  padding: 12, background: 'rgba(62,199,173,0.05)', borderRadius: 8,
                  border: '1px solid rgba(62,199,173,0.15)', marginBottom: 12,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <Heart size={14} color="#3ec7ad" />
                    <span style={{ fontSize: 13, fontWeight: 600 }}>{selectedOrgan}</span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11 }}>
                    <span style={{ color: '#607389' }}>Health</span>
                    <span style={{ color: healthColor(selectedHealth.health), fontWeight: 600 }}>{selectedHealth.health}%</span>
                    <span style={{ color: '#607389' }}>Risk</span>
                    <span style={{ color: healthColor(100 - selectedHealth.risk), fontWeight: 600 }}>{selectedHealth.risk}%</span>
                    <span style={{ color: '#607389' }}>Status</span>
                    <span style={{ color: selectedHealth.status === 'Critical' ? '#ef565f' : '#eab308' }}>{selectedHealth.status}</span>
                    <span style={{ color: '#607389' }}>Inflammation</span>
                    <span>{selectedHealth.inflammation}%</span>
                    <span style={{ color: '#607389' }}>Metastasis Risk</span>
                    <span style={{ color: selectedHealth.metastasisRisk > 50 ? '#ef565f' : '#eab308' }}>{selectedHealth.metastasisRisk}%</span>
                    <span style={{ color: '#607389' }}>Invasion</span>
                    <span style={{ color: selectedHealth.invasion > 50 ? '#ef565f' : '#eab308' }}>{selectedHealth.invasion}%</span>
                  </div>
                </div>
              )}

              {/* Organ Health List */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>
                  Organ Status
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {Object.entries(organHealth).map(([name, h]) => (
                    <button
                      key={name}
                      onClick={() => setSelectedOrgan(selectedOrgan === name ? null : name)}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 6, padding: '5px 8px',
                        background: selectedOrgan === name ? 'rgba(62,199,173,0.1)' : 'transparent',
                        border: selectedOrgan === name ? '1px solid rgba(62,199,173,0.3)' : '1px solid transparent',
                        borderRadius: 5, cursor: 'pointer', width: '100%', textAlign: 'left',
                      }}
                    >
                      <span style={{
                        width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                        background: ORGAN_COLORS[name] || '#5db7a2',
                      }} />
                      <span style={{ flex: 1, fontSize: 11, color: '#b0c8d8' }}>{name}</span>
                      <span style={{
                        fontSize: 10, fontWeight: 600, color: healthColor(h.health),
                      }}>{h.health}%</span>
                      <span style={{
                        fontSize: 9, padding: '1px 5px', borderRadius: 3,
                        background: h.risk > 60 ? 'rgba(239,68,68,0.15)' : h.risk > 30 ? 'rgba(234,179,8,0.15)' : 'rgba(34,197,94,0.15)',
                        color: healthColor(h.health),
                      }}>{h.risk}%</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Live Biomarkers & Genomic Drivers Controls */}
              <div style={{
                marginBottom: 16, padding: 12, background: 'rgba(15, 23, 36, 0.8)',
                borderRadius: 8, border: '1px solid #1a2a39',
              }}>
                <div style={{ fontSize: 10, color: '#3ec7ad', textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 700, marginBottom: 10 }}>
                  🧬 Live Biomarkers & Genomic Drivers
                </div>

                {/* CA 19-9 Slider */}
                <div style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                    <span style={{ color: '#8293a7' }}>CA 19-9</span>
                    <span style={{ color: '#ef565f', fontWeight: 600 }}>{activeBiomarkers.ca199} U/mL</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="3000"
                    step="10"
                    value={activeBiomarkers.ca199}
                    onChange={(e) => setActiveBiomarkers((b) => ({ ...b, ca199: Number(e.target.value) }))}
                    style={{ width: '100%', accentColor: '#ef565f', cursor: 'pointer' }}
                  />
                </div>

                {/* CEA Slider */}
                <div style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                    <span style={{ color: '#8293a7' }}>CEA</span>
                    <span style={{ color: '#f59e0b', fontWeight: 600 }}>{activeBiomarkers.cea} ng/mL</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="50"
                    step="0.5"
                    value={activeBiomarkers.cea}
                    onChange={(e) => setActiveBiomarkers((b) => ({ ...b, cea: Number(e.target.value) }))}
                    style={{ width: '100%', accentColor: '#f59e0b', cursor: 'pointer' }}
                  />
                </div>

                {/* Bilirubin Slider */}
                <div style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                    <span style={{ color: '#8293a7' }}>Bilirubin</span>
                    <span style={{ color: '#eab308', fontWeight: 600 }}>{activeBiomarkers.bilirubin} mg/dL</span>
                  </div>
                  <input
                    type="range"
                    min="0.2"
                    max="12.0"
                    step="0.1"
                    value={activeBiomarkers.bilirubin}
                    onChange={(e) => setActiveBiomarkers((b) => ({ ...b, bilirubin: Number(e.target.value) }))}
                    style={{ width: '100%', accentColor: '#eab308', cursor: 'pointer' }}
                  />
                </div>

                {/* Albumin Slider */}
                <div style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                    <span style={{ color: '#8293a7' }}>Albumin</span>
                    <span style={{ color: '#62dec0', fontWeight: 600 }}>{activeBiomarkers.albumin} g/dL</span>
                  </div>
                  <input
                    type="range"
                    min="1.5"
                    max="5.0"
                    step="0.1"
                    value={activeBiomarkers.albumin}
                    onChange={(e) => setActiveBiomarkers((b) => ({ ...b, albumin: Number(e.target.value) }))}
                    style={{ width: '100%', accentColor: '#62dec0', cursor: 'pointer' }}
                  />
                </div>

                {/* Weight Loss % Slider */}
                <div style={{ marginBottom: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                    <span style={{ color: '#8293a7' }}>Weight Loss</span>
                    <span style={{ color: '#ef565f', fontWeight: 600 }}>{activeBiomarkers.weight_loss_pct}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="30"
                    step="1"
                    value={activeBiomarkers.weight_loss_pct}
                    onChange={(e) => setActiveBiomarkers((b) => ({ ...b, weight_loss_pct: Number(e.target.value) }))}
                    style={{ width: '100%', accentColor: '#ef565f', cursor: 'pointer' }}
                  />
                </div>

                {/* Genomic Mutation Switches (KRAS, TP53, SMAD4) */}
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 6 }}>
                  Genomic Driver Status
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
                  <button
                    onClick={() => setActiveBiomarkers((b) => ({ ...b, kras: !b.kras }))}
                    style={{
                      padding: '4px 0', fontSize: 10, fontWeight: 600, borderRadius: 4,
                      background: activeBiomarkers.kras ? 'rgba(239, 68, 68, 0.25)' : 'rgba(255, 255, 255, 0.04)',
                      border: activeBiomarkers.kras ? '1px solid #ef4444' : '1px solid #1a2a39',
                      color: activeBiomarkers.kras ? '#fca5a5' : '#607389', cursor: 'pointer',
                    }}
                  >
                    KRAS {activeBiomarkers.kras ? 'Mut' : 'WT'}
                  </button>
                  <button
                    onClick={() => setActiveBiomarkers((b) => ({ ...b, tp53: !b.tp53 }))}
                    style={{
                      padding: '4px 0', fontSize: 10, fontWeight: 600, borderRadius: 4,
                      background: activeBiomarkers.tp53 ? 'rgba(245, 158, 11, 0.25)' : 'rgba(255, 255, 255, 0.04)',
                      border: activeBiomarkers.tp53 ? '1px solid #f59e0b' : '1px solid #1a2a39',
                      color: activeBiomarkers.tp53 ? '#fcd34d' : '#607389', cursor: 'pointer',
                    }}
                  >
                    TP53 {activeBiomarkers.tp53 ? 'Mut' : 'WT'}
                  </button>
                  <button
                    onClick={() => setActiveBiomarkers((b) => ({ ...b, smad4: !b.smad4 }))}
                    style={{
                      padding: '4px 0', fontSize: 10, fontWeight: 600, borderRadius: 4,
                      background: activeBiomarkers.smad4 ? 'rgba(168, 85, 247, 0.25)' : 'rgba(255, 255, 255, 0.04)',
                      border: activeBiomarkers.smad4 ? '1px solid #a855f7' : '1px solid #1a2a39',
                      color: activeBiomarkers.smad4 ? '#e9d5ff' : '#607389', cursor: 'pointer',
                    }}
                  >
                    SMAD4 {activeBiomarkers.smad4 ? 'Alt' : 'WT'}
                  </button>
                </div>
              </div>

              {/* Treatment */}
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 10, color: '#607389', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>
                  Treatment Simulation
                </div>
                <select
                  value={simTreatment}
                  onChange={(e) => setSimTreatment(e.target.value)}
                  style={{
                    width: '100%', padding: '6px 8px', fontSize: 11,
                    background: 'rgba(0,0,0,0.3)', border: '1px solid #152235',
                    borderRadius: 5, color: '#e5edf8', marginBottom: 6,
                  }}
                >
                  {treatments.map((t) => <option key={t}>{t}</option>)}
                </select>
                <div style={{ display: 'flex', gap: 4, marginBottom: 6 }}>
                  {[0, 3, 6, 12, 24].map((m) => (
                    <button
                      key={m}
                      onClick={() => setMonth(m)}
                      style={{
                        flex: 1, padding: '4px 0', fontSize: 10,
                        background: month === m ? 'rgba(62,199,173,0.2)' : 'rgba(255,255,255,0.04)',
                        border: month === m ? '1px solid #3ec7ad' : '1px solid transparent',
                        borderRadius: 4, color: month === m ? '#3ec7ad' : '#607389',
                        cursor: 'pointer',
                      }}
                    >
                      {m === 0 ? 'Now' : `${m}m`}
                    </button>
                  ))}
                </div>
                <button
                  onClick={simulate}
                  disabled={isSimulating}
                  style={{
                    width: '100%', padding: '6px 0', fontSize: 11, fontWeight: 600,
                    background: isSimulating ? 'rgba(255,255,255,0.05)' : 'linear-gradient(135deg,#3ec7ad,#62dec0)',
                    border: '1px solid #54dfc4', borderRadius: 5,
                    color: isSimulating ? '#607389' : '#08221f',
                    cursor: isSimulating ? 'not-allowed' : 'pointer',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
                  }}
                >
                  <Play size={11} /> {isSimulating ? 'Simulating...' : `Run ${simTreatment}`}
                </button>
              </div>

              {/* Timeline Data */}
              {month > 0 && (
                <div style={{
                  padding: 10, background: 'rgba(245,158,11,0.08)', borderRadius: 8,
                  border: '1px solid rgba(245,158,11,0.2)',
                }}>
                  <div style={{ fontSize: 10, color: '#f59e0b', fontWeight: 600, marginBottom: 6 }}>
                    📊 Simulation at Month {month}
                  </div>
                  <div style={{ fontSize: 11, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3 }}>
                    <span style={{ color: '#607389' }}>Tumor size</span>
                    <span style={{ color: '#ef565f' }}>{config.sizeCm} cm ({(1.5 + (biomarkers.ca199 / 1000) * 2.5).toFixed(1)} → {config.sizeCm})</span>
                    <span style={{ color: '#607389' }}>Vascular invasion</span>
                    <span style={{ color: config.vascularInvasion > 0.4 ? '#ef565f' : '#eab308' }}>{Math.round(config.vascularInvasion * 100)}%</span>
                    <span style={{ color: '#607389' }}>Necrosis</span>
                    <span>{Math.round(config.necrosisPct * 100)}%</span>
                    <span style={{ color: '#607389' }}>Pancreas health</span>
                    <span style={{ color: healthColor(organHealth.Pancreas?.health || 100) }}>{organHealth.Pancreas?.health}%</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      {/* Modal for NIfTI Pair Upload */}
      {niftiModalOpen && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 999,
          background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16,
        }}>
          <div style={{
            width: '100%', maxWidth: 480, background: '#0a121c',
            border: '1px solid #1f354d', borderRadius: 12, padding: 24,
            boxShadow: '0 20px 40px rgba(0,0,0,0.6)', color: '#e5edf8',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h3 style={{ margin: 0, fontSize: 16, color: '#3ec7ad', fontWeight: 600 }}>
                Upload NIfTI CT + Label Pair (VTK Pipeline)
              </h3>
              <button
                onClick={() => setNiftiModalOpen(false)}
                style={{ background: 'transparent', border: 0, color: '#8293a7', cursor: 'pointer', fontSize: 18 }}
              >
                ✕
              </button>
            </div>

            <p style={{ fontSize: 12, color: '#8293a7', marginBottom: 16, lineHeight: 1.4 }}>
              Upload matching Medical Segmentation Decathlon Task07 NIfTI files (CT image + label mask).
              The Python VTK pipeline strictly validates matching dimensions, spacing, and origin.
            </p>

            {uploadError && (
              <div style={{
                padding: '10px 12px', background: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid #ef4444', borderRadius: 6, color: '#fca5a5',
                fontSize: 12, marginBottom: 16,
              }}>
                ⚠️ {uploadError}
              </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginBottom: 20 }}>
              <div>
                <label style={{ display: 'block', fontSize: 12, color: '#b0c8d8', marginBottom: 6, fontWeight: 500 }}>
                  CT Volume File (.nii / .nii.gz)
                </label>
                <input
                  type="file"
                  accept=".nii,.gz"
                  onChange={(e) => setCtFile(e.target.files?.[0] || null)}
                  style={{
                    width: '100%', padding: '8px 10px', fontSize: 12,
                    background: 'rgba(255,255,255,0.04)', border: '1px solid #1f354d',
                    borderRadius: 6, color: '#e5edf8',
                  }}
                />
                {ctFile && <span style={{ fontSize: 11, color: '#3ec7ad', marginTop: 4, display: 'block' }}>✓ {ctFile.name} ({(ctFile.size / 1024 / 1024).toFixed(1)} MB)</span>}
              </div>

              <div>
                <label style={{ display: 'block', fontSize: 12, color: '#b0c8d8', marginBottom: 6, fontWeight: 500 }}>
                  Segmentation Label File (.nii / .nii.gz)
                </label>
                <input
                  type="file"
                  accept=".nii,.gz"
                  onChange={(e) => setLabelFile(e.target.files?.[0] || null)}
                  style={{
                    width: '100%', padding: '8px 10px', fontSize: 12,
                    background: 'rgba(255,255,255,0.04)', border: '1px solid #1f354d',
                    borderRadius: 6, color: '#e5edf8',
                  }}
                />
                {labelFile && <span style={{ fontSize: 11, color: '#3ec7ad', marginTop: 4, display: 'block' }}>✓ {labelFile.name} ({(labelFile.size / 1024 / 1024).toFixed(1)} MB)</span>}
              </div>
            </div>

            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
              <button
                onClick={() => setNiftiModalOpen(false)}
                style={{
                  padding: '8px 16px', background: 'rgba(255,255,255,0.05)',
                  border: '1px solid #1f354d', borderRadius: 6, color: '#8293a7',
                  fontSize: 12, cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleGenerate3D}
                disabled={uploadLoading || !ctFile || !labelFile}
                style={{
                  padding: '8px 20px',
                  background: uploadLoading || !ctFile || !labelFile ? 'rgba(62,199,173,0.2)' : 'linear-gradient(135deg, #3ec7ad, #62dec0)',
                  border: '1px solid #54dfc4', borderRadius: 6,
                  color: uploadLoading || !ctFile || !labelFile ? '#607389' : '#08221f',
                  fontSize: 12, fontWeight: 600,
                  cursor: uploadLoading || !ctFile || !labelFile ? 'not-allowed' : 'pointer',
                }}
              >
                {uploadLoading ? 'VTK Processing...' : 'Generate 3D'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
