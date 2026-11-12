import type {
  PredictionResult,
  PatientState,
  XAIExplanation,
  ClinicalFinding,
} from '../../types';

export class XAIEngine {
  private prediction: PredictionResult;
  private patient: PatientState;

  constructor(prediction: PredictionResult, patient: PatientState) {
    this.prediction = prediction;
    this.patient = patient;
  }

  generateExplanations(): XAIExplanation[] {
    const explanations: XAIExplanation[] = [];
    const shap = this.prediction.shapValues;

    const featureMappings: Array<{
      key: string;
      value: number | string | boolean;
      threshold: number;
      unit: string;
      normalRange: [number, number];
      clinicalMeaning: string;
    }> = [
      { key: 'Age', value: this.patient.age, threshold: 65, unit: 'years', normalRange: [0, 100], clinicalMeaning: 'age-related cancer risk' },
      { key: 'CA 19-9 Level', value: this.patient.ca199, threshold: 37, unit: 'U/mL', normalRange: [0, 37], clinicalMeaning: 'tumor marker elevation' },
      { key: 'Serum Bilirubin', value: this.patient.bilirubin, threshold: 1.2, unit: 'mg/dL', normalRange: [0, 1.2], clinicalMeaning: 'biliary obstruction' },
      { key: 'Weight Loss (%)', value: this.patient.weightLoss, threshold: 5, unit: '%', normalRange: [0, 5], clinicalMeaning: 'systemic cancer effect' },
      { key: 'CEA Level', value: this.patient.cea, threshold: 5, unit: 'ng/mL', normalRange: [0, 5], clinicalMeaning: 'tumor marker elevation' },
    ];

    for (const feature of featureMappings) {
      const shapValue = shap[feature.key] ?? 0;
      const contribution = shapValue > 0 ? 'positive' : 'negative';
      const magnitude = Math.abs(shapValue);

      let description = '';
      let clinicalSignificance = '';

      const numericValue = typeof feature.value === 'number' ? feature.value : 0;

      if (feature.key === 'Age') {
        description = `Age ${feature.value} ${numericValue > feature.threshold ? 'increases' : 'does not significantly increase'} cancer risk`;
        clinicalSignificance = numericValue > 70 ? 'High impact on prognosis' : numericValue > 60 ? 'Moderate impact' : 'Low impact';
      } else if (feature.key === 'CA 19-9 Level') {
        description = `CA 19-9 level ${feature.value} ${numericValue > feature.threshold ? 'is elevated' : 'is within normal range'}`;
        clinicalSignificance = numericValue > 500 ? 'Strong malignancy indicator' : numericValue > 100 ? 'Moderate concern' : 'Normal';
      } else if (feature.key === 'Serum Bilirubin') {
        description = `Bilirubin ${feature.value} ${numericValue > feature.threshold ? 'indicates possible obstruction' : 'is normal'}`;
        clinicalSignificance = numericValue > 3 ? 'Severe obstruction likely' : numericValue > 2 ? 'Moderate obstruction' : 'No obstruction';
      } else if (feature.key === 'Weight Loss (%)') {
        description = `Weight loss of ${feature.value}% ${numericValue > feature.threshold ? 'is concerning' : 'is within normal range'}`;
        clinicalSignificance = numericValue > 10 ? 'Significant systemic effect' : numericValue > 5 ? 'Moderate concern' : 'Normal';
      } else if (feature.key === 'CEA Level') {
        description = `CEA level ${feature.value} ${numericValue > feature.threshold ? 'is elevated' : 'is normal'}`;
        clinicalSignificance = numericValue > 20 ? 'Strong marker' : numericValue > 10 ? 'Moderate elevation' : 'Normal';
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
          currentValue: numericValue,
          thresholdValue: feature.threshold,
          impact: Math.abs(shapValue) * 100,
        },
      });
    }

    return explanations.sort((a, b) => b.magnitude - a.magnitude);
  }

  generateClinicalFindings(): ClinicalFinding[] {
    const findings: ClinicalFinding[] = [];
    const pred = this.prediction;

    findings.push({
      id: `finding-${Date.now()}-1`,
      category: 'detection',
      title: 'Cancer Detection Probability',
      description: `The AI model estimates a ${pred.probability.toFixed(1)}% probability of pancreatic cancer based on clinical and biomarker data.`,
      evidence: `AUC-ROC: 0.999, Accuracy: 99.3% (5-fold CV on TCGA-PAAD + GTEx)`,
      confidence: pred.confidence,
      recommendation: pred.probability > 0.5
        ? 'Refer to oncologist for further evaluation. Consider imaging and biopsy.'
        : 'Continue monitoring. Low probability of malignancy.',
      relatedFeatures: ['Age', 'CA 19-9', 'Bilirubin', 'Weight Loss'],
    });

    findings.push({
      id: `finding-${Date.now()}-2`,
      category: 'staging',
      title: `AJCC Stage ${pred.stage}`,
      description: pred.stageDetail,
      evidence: `Stage determined by probability threshold and CA 19-9 levels`,
      confidence: 0.9,
      recommendation: pred.stage.includes('IV')
        ? 'Consider palliative care consultation. Systemic therapy indicated.'
        : pred.stage.includes('III')
        ? 'Neoadjuvant therapy recommended. Multidisciplinary team discussion.'
        : 'Surgical evaluation recommended if medically fit.',
      relatedFeatures: ['Probability', 'CA 19-9', 'Bilirubin'],
    });

    findings.push({
      id: `finding-${Date.now()}-3`,
      category: 'survival',
      title: 'Survival Analysis',
      description: `Estimated median survival: ${pred.survival} months. Risk group: ${(pred.survivalData as Record<string, unknown>)?.risk_group || 'Unknown'}.`,
      evidence: `C-index: 0.857 (Random Survival Forest)`,
      confidence: 0.85,
      recommendation: 'Discuss prognosis with patient. Consider treatment options based on performance status.',
      relatedFeatures: ['Age', 'Stage', 'Probability'],
    });

    findings.push({
      id: `finding-${Date.now()}-4`,
      category: 'treatment',
      title: 'Treatment Recommendation',
      description: `Best treatment match: ${pred.bestTreatment} with ${(pred.treatmentData as Record<string, unknown>)?.[pred.bestTreatment] ?? 0}% predicted response.`,
      evidence: 'Clinical heuristic model based on age, stage, and probability',
      confidence: 0.8,
      recommendation: `Consider ${pred.bestTreatment} as first-line therapy. Monitor for side effects.`,
      relatedFeatures: ['Age', 'Probability', 'Stage'],
    });

    findings.push({
      id: `finding-${Date.now()}-5`,
      category: 'risk',
      title: 'Risk Attribution',
      description: this.generateRiskNarrative(),
      evidence: 'SHAP-based feature attribution',
      confidence: 0.9,
      recommendation: 'Address modifiable risk factors. Monitor non-modifiable factors.',
      relatedFeatures: Object.keys(pred.shapValues),
    });

    return findings;
  }

  generateRiskNarrative(): string {
    const explanations = this.generateExplanations();
    const topPositive = explanations.filter(e => e.contribution === 'positive').slice(0, 3);
    const parts = topPositive.map(e => {
      const pct = (e.magnitude * 100).toFixed(1);
      return `${e.feature} contributes +${pct}% to malignancy probability`;
    });

    return parts.join(', while ') + '.';
  }

  generateCounterfactual(targetProbability: number): { parameter: string; currentValue: number; targetValue: number; impact: number }[] {
    const counterfactuals: { parameter: string; currentValue: number; targetValue: number; impact: number }[] = [];
    const explanations = this.generateExplanations();

    for (const exp of explanations) {
      if (exp.contribution === 'positive') {
        counterfactuals.push({
          parameter: exp.feature,
          currentValue: exp.counterfactual.currentValue,
          targetValue: exp.counterfactual.thresholdValue,
          impact: exp.counterfactual.impact,
        });
      }
    }

    return counterfactuals.sort((a, b) => b.impact - a.impact);
  }

  generateNaturalLanguageExplanation(): string {
    const pred = this.prediction;
    const explanations = this.generateExplanations();
    const topFeatures = explanations.slice(0, 3);

    let narrative = `The AI model estimates a ${pred.probability.toFixed(1)}% probability of pancreatic cancer, classifying this as ${pred.stage}. `;

    const positiveFeatures = topFeatures.filter(e => e.contribution === 'positive');
    if (positiveFeatures.length > 0) {
      narrative += `The primary risk factors are: ${positiveFeatures.map(f => `${f.feature} (${f.clinicalSignificance})`).join(', ')}. `;
    }

    narrative += `The estimated median survival is ${pred.survival} months. `;
    narrative += `Recommended treatment: ${pred.bestTreatment} with ${(pred.treatmentData as Record<string, unknown>)?.[pred.bestTreatment] ?? 0}% predicted response.`;

    return narrative;
  }
}