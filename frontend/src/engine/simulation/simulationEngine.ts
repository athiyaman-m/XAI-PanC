import type {
  PatientState,
  OrganHealth,
  OrganName,
  SimulationResult,
  PredictionResult,
  TreatmentOption,
} from '../../types';
import { ORGAN_COLORS } from '../twin/twinStore';

export interface SimulationParameters {
  days: number;
  treatment: string;
  treatmentResponse: number;
  krasMutation: boolean;
  tp53Mutation: boolean;
  smad4Alteration: boolean;
}

export interface CompartmentalState {
  tumor: number;
  ca199: number;
  cea: number;
  bilirubin: number;
  drug: number;
  immune: number;
}

export interface CompartmentalParams {
  r: number;
  K: number;
  alpha_ca: number;
  beta_ca: number;
  alpha_ce: number;
  beta_ce: number;
  delta_d: number;
  gamma_d: number;
  delta_i: number;
  sigma: number;
  k_sigma: number;
  alpha_b: number;
  beta_b: number;
  alpha_i: number;
  beta_i: number;
  k_i: number;
}

export class PancreaticTwinModel {
  state: CompartmentalState;
  params: CompartmentalParams;

  constructor(state: CompartmentalState, params: CompartmentalParams) {
    this.state = state;
    this.params = params;
  }

  update(dt: number = 1): void {
    const s = this.state;
    const p = this.params;

    const obstructionThreshold = p.K * 0.3;
    const obstruction = Math.max(0, s.tumor - obstructionThreshold);

    const dT = p.r * s.tumor * (1 - s.tumor / p.K) - p.delta_d * s.drug * s.tumor - p.delta_i * s.immune * s.tumor;
    const dCA = p.alpha_ca * Math.max(0, s.tumor) - p.beta_ca * s.ca199;
    const dCE = p.alpha_ce * Math.max(0, s.tumor) - p.beta_ce * s.cea;
    const dB = p.alpha_b * obstruction - p.beta_b * s.bilirubin;
    const dD = -p.gamma_d * s.drug;
    const dI = p.alpha_i * s.tumor / (s.tumor + p.k_i) - p.beta_i * s.immune;

    s.tumor = Math.max(0, s.tumor + dT * dt);
    s.ca199 = Math.max(0, s.ca199 + dCA * dt);
    s.cea = Math.max(0, s.cea + dCE * dt);
    s.bilirubin = Math.max(0, s.bilirubin + dB * dt);
    s.drug = Math.max(0, s.drug + dD * dt);
    s.immune = Math.max(0, Math.min(1, s.immune + dI * dt));
  }

  addIntervention(compartment: keyof CompartmentalState, amount: number): void {
    (this.state[compartment] as number) = Math.max(0, (this.state[compartment] as number) + amount);
  }

  runForecast(days: number, interventions: { day: number; compartment: keyof CompartmentalState; amount: number }[] = []): CompartmentalState[] {
    const trajectory: CompartmentalState[] = [];
    const initialState = { ...this.state };

    for (let day = 0; day <= days; day++) {
      for (const intervention of interventions) {
        if (intervention.day === day) {
          this.addIntervention(intervention.compartment, intervention.amount);
        }
      }
      this.update(1);
      trajectory.push({ ...this.state });
    }

    this.state = initialState;
    return trajectory;
  }
}

export function fitTwinParameters(
  prob: number,
  stage: string,
  ca199: number,
  cea: number,
  bilirubin: number,
  treatmentResponse: number,
  krasExpr: number,
  tp53Expr: number,
  smad4Expr: number
): CompartmentalParams {
  const stageNum = stage.includes('IV') ? 4 : stage.includes('III') ? 3 : stage.includes('II') ? 2 : stage.includes('I') ? 1 : 0;
  const K = [0, 8, 20, 40, 80][stageNum] * (0.8 + 0.4 * prob);

  let r = 0.008 + prob * 0.025;
  r += Math.max(0, (krasExpr - 2) / 10) * 0.015;
  r += Math.max(0, (5 - tp53Expr) / 5) * 0.008;

  const delta_d = 0.05 + treatmentResponse * 0.30;
  const delta_d_final = delta_d * (1 - Math.max(0, (5 - smad4Expr) / 5) * 0.5);

  const alpha_ca = ca199 / (K * 0.6);
  const alpha_ce = cea / (K * 0.4);

  return {
    r,
    K,
    alpha_ca: Math.max(0.01, alpha_ca),
    beta_ca: 0.03,
    alpha_ce: Math.max(0.01, alpha_ce),
    beta_ce: 0.02,
    delta_d: delta_d_final,
    gamma_d: 0.25,
    delta_i: 0.05,
    sigma: 0.01,
    k_sigma: 10.0,
    alpha_b: 0.04,
    beta_b: 0.10,
    alpha_i: 0.03,
    beta_i: 0.02,
    k_i: 5.0,
  };
}

export function computeTwinMetrics(state: CompartmentalState, params: CompartmentalParams): {
  tumorRadiusCm: number;
  risk: number;
  health: number;
  stage: string;
} {
  const tumorRadiusCm = state.tumor > 0 ? Math.pow((3 * state.tumor) / (4 * Math.PI), 1 / 3) : 0;

  const sizeRisk = Math.min(40, tumorRadiusCm * 5);
  const ca199Risk = Math.min(25, (state.ca199 / 500) * 25);
  const ceaRisk = Math.min(15, (state.cea / 20) * 15);
  const biliRisk = Math.min(20, (state.bilirubin / 5) * 20);
  const immuneRisk = (1 - state.immune) * 10;

  const risk = Math.min(100, sizeRisk + ca199Risk + ceaRisk + biliRisk + immuneRisk);
  const health = Math.max(0, 100 - risk * 1.1);

  let stage: string;
  if (risk >= 70) stage = 'Stage IV';
  else if (risk >= 50) stage = 'Stage III';
  else if (risk >= 35) stage = 'Stage II';
  else if (risk >= 20) stage = 'Stage I';
  else stage = 'Normal';

  return { tumorRadiusCm, risk, health, stage };
}

export class DiseaseProgressionSimulator {
  private patient: PatientState;
  private prediction: PredictionResult | null;

  constructor(patient: PatientState, prediction: PredictionResult | null) {
    this.patient = patient;
    this.prediction = prediction;
  }

  simulate(months: number, treatment: string, treatmentResponse: number): SimulationResult {
    const prob = this.prediction?.probability ?? 0.5;
    const ca199 = this.patient.ca199;
    const cea = this.patient.cea;
    const bilirubin = this.patient.bilirubin;
    const stage = this.prediction?.stage ?? 'Stage II';

    const krasExpr = this.patient.kras === 'Mutated' ? 8 : 5;
    const tp53Expr = this.patient.tp53 === 'Mutated' ? 8 : 5;
    const smad4Expr = this.patient.smad4 === 'Altered' ? 2 : 5;

    const params = fitTwinParameters(prob, stage, ca199, cea, bilirubin, treatmentResponse, krasExpr, tp53Expr, smad4Expr);

    const initialState: CompartmentalState = {
      tumor: params.K * 0.3,
      ca199: ca199,
      cea: cea,
      bilirubin: bilirubin,
      drug: treatmentResponse > 0.5 ? 0.8 : 0,
      immune: 0.3,
    };

    const twin = new PancreaticTwinModel(initialState, params);

    const interventions: { day: number; compartment: keyof CompartmentalState; amount: number }[] = [];
    if (treatmentResponse > 0.5) {
      interventions.push({ day: 0, compartment: 'drug', amount: 0.8 });
      interventions.push({ day: 30, compartment: 'drug', amount: 0.6 });
      interventions.push({ day: 60, compartment: 'drug', amount: 0.6 });
    }

    const trajectory = twin.runForecast(months * 30, interventions);
    const finalState = trajectory[trajectory.length - 1];
    const metrics = computeTwinMetrics(finalState, params);

    const organHealth: Record<OrganName, Partial<OrganHealth>> = {
      Pancreas: {
        tumorBurden: metrics.tumorRadiusCm * 20,
        healthScore: Math.max(0, 100 - metrics.risk),
        riskScore: metrics.risk,
      },
      Liver: {
        metastasisRisk: finalState.bilirubin > 2 ? 60 : 30,
        healthScore: Math.max(0, 100 - finalState.bilirubin * 10),
      },
      'Bile Duct': {
        obstruction: finalState.bilirubin > 1.5 ? 70 : 30,
        healthScore: Math.max(0, 100 - finalState.bilirubin * 15),
      },
      Lungs: {
        metastasisRisk: prob > 0.6 ? 40 : 15,
        healthScore: Math.max(0, 100 - (prob > 0.6 ? 40 : 15)),
      },
      'Lymph Nodes': {
        tumorBurden: metrics.tumorRadiusCm > 3 ? 50 : 20,
        healthScore: Math.max(0, 100 - (metrics.tumorRadiusCm > 3 ? 50 : 20)),
      },
      Duodenum: {
        obstruction: finalState.bilirubin > 1.5 ? 40 : 10,
        healthScore: Math.max(0, 100 - finalState.bilirubin * 8),
      },
      Stomach: {
        healthScore: Math.max(0, 100 - metrics.risk * 0.3),
      },
      Spleen: {
        healthScore: Math.max(0, 100 - finalState.bilirubin * 5),
      },
      Heart: {
        healthScore: Math.max(0, 100 - metrics.risk * 0.1),
      },
      Kidneys: {
        healthScore: Math.max(0, 100 - finalState.bilirubin * 3),
      },
    };

    const biomarkers: Record<string, number> = {
      'CA 19-9': finalState.ca199,
      CEA: finalState.cea,
      Bilirubin: finalState.bilirubin,
      Tumor: finalState.tumor,
      Immune: finalState.immune,
    };

    const survivalProb = Math.max(0.05, 1 - metrics.risk / 100);

    return {
      month: months,
      treatment,
      tumorVolume: metrics.tumorRadiusCm,
      tumorVolumeChange: (metrics.tumorRadiusCm - (params.K * 0.3 > 0 ? Math.pow((3 * params.K * 0.3) / (4 * Math.PI), 1 / 3) : 0)),
      organHealth,
      biomarkers,
      survivalProbability: survivalProb,
      responseRate: treatmentResponse * 100,
      notes: `Simulated ${months} months with ${treatment}. Final tumor radius: ${metrics.tumorRadiusCm.toFixed(2)} cm.`,
    };
  }
}

export function getTreatmentOptions(prediction: PredictionResult, patient: PatientState): TreatmentOption[] {
  const prob = prediction.probability;
  const age = patient.age;
  const stage = prediction.stage;

  const options: TreatmentOption[] = [
    {
      id: 'folfi',
      name: 'FOLFIRINOX',
      description: 'Combination chemotherapy: 5-FU, leucovorin, irinotecan, oxaliplatin',
      responseProbability: Math.min(0.92, 0.5 + (0.15 * (age < 65 ? 1 : 0)) + (0.08 * (prob < 0.6 ? 1 : 0))),
      sideEffects: ['Neutropenia', 'Peripheral neuropathy', 'Diarrhea', 'Fatigue'],
      complications: ['Myelosuppression', 'Hand-foot syndrome', 'Hypokalemia'],
      survivalBenefit: 4.2,
      tumorShrinkage: 35,
      organRecovery: 60,
      contraindications: ['Severe cardiovascular disease', 'Poor performance status'],
      bestFor: ['Younger patients', 'Fit patients', 'High tumor burden'],
    },
    {
      id: 'gem',
      name: 'Gemcitabine + nab-paclitaxel',
      description: 'Gemcitabine with albumin-bound paclitaxel',
      responseProbability: Math.min(0.88, 0.45 + Math.max(0, (1 - prob) * 0.2)),
      sideEffects: ['Myelosuppression', 'Peripheral neuropathy', 'Nausea', 'Fatigue'],
      complications: ['Thrombocytopenia', 'Neutropenia', 'Pneumonitis'],
      survivalBenefit: 3.8,
      tumorShrinkage: 28,
      organRecovery: 65,
      contraindications: ['Severe liver dysfunction', 'Hypersensitivity to paclitaxel'],
      bestFor: ['Older patients', 'Moderate tumor burden', 'Liver metastases'],
    },
    {
      id: 'folfox',
      name: 'FOLFOX',
      description: '5-FU with leucovorin and oxaliplatin',
      responseProbability: Math.min(0.80, (0.5 + (0.15 * (age < 65 ? 1 : 0)) + (0.08 * (prob < 0.6 ? 1 : 0))) * 0.85),
      sideEffects: ['Peripheral neuropathy', 'Diarrhea', 'Fatigue', 'Nausea'],
      complications: ['Myelosuppression', 'Hand-foot syndrome', 'Allergic reactions'],
      survivalBenefit: 3.5,
      tumorShrinkage: 30,
      organRecovery: 55,
      contraindications: ['Severe neuropathy', 'Cardiac disease'],
      bestFor: ['Colorectal metastases', 'Liver-dominant disease'],
    },
    {
      id: 'bsc',
      name: 'Best Supportive Care',
      description: 'Palliative care focused on symptom management',
      responseProbability: Math.min(0.55, 0.30 + prob * 0.15),
      sideEffects: [],
      complications: [],
      survivalBenefit: 1.5,
      tumorShrinkage: 0,
      organRecovery: 40,
      contraindications: [],
      bestFor: ['Poor performance status', 'Advanced age', 'Multiple comorbidities'],
    },
  ];

  return options.sort((a, b) => b.responseProbability - a.responseProbability);
}

export { ORGAN_COLORS };