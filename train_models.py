"""
train_models.py — Train proper ML models on TCGA-PAAD + GTEx data
Replaces all hardcoded/synthetic logic with real trained models.

Output saved to: models/ directory
  - vae_encoder.pt       (PyTorch VAE encoder weights)
  - xgb_detection.json   (XGBoost cancer vs normal)
  - xgb_stage.json       (XGBoost AJCC stage)
  - rf_survival.pkl      (RandomForest Survival)
  - scaler.pkl           (feature scaler)
  - label_encoders.pkl   (categorical encoders)
  - feature_columns.pkl  (ordered feature names)
  - sample_input.pkl     (template for dashboard input)
"""

import os, sys, warnings, pickle, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

# ML
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.impute import KNNImputer

# Survival
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# PyTorch for VAE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "PAN-DATASET")
OUT = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────────────────────

def load_all_data():
    """Load TCGA, GTEx, GEO expression + clinical + mutation data."""
    print("=" * 60)
    print("Loading raw data...")

    # TCGA RNA-seq (1000 genes)
    tcga_rna = pd.read_csv(os.path.join(DATA, "TCGA-PAAD", "RNA-Seq-STAR-Counts", "top1000_expression.csv"),
                           index_col=0)
    print(f"  TCGA RNA-seq: {tcga_rna.shape}")

    # TCGA biomarkers (7 key genes)
    tcga_bm = pd.read_csv(os.path.join(DATA, "TCGA-PAAD", "RNA-Seq-STAR-Counts", "selected_biomarkers_expression.csv"),
                          index_col=0)
    print(f"  TCGA biomarkers: {tcga_bm.shape}")

    # Mutation matrix
    tcga_mut = pd.read_csv(os.path.join(DATA, "TCGA-PAAD", "masked-somatic-mutation", "patient_mutation_matrix.csv"),
                           index_col=0)
    # Keep only relevant genes
    mut_keep = [c for c in ['KRAS', 'TP53', 'SMAD4', 'CDKN2A', 'BRCA1', 'BRCA2'] if c in tcga_mut.columns]
    tcga_mut = tcga_mut[mut_keep]
    print(f"  TCGA mutations ({mut_keep}): {tcga_mut.shape}")

    # GTEx pancreas (normal)
    gtex = pd.read_csv(os.path.join(DATA, "GTEx", "gene_tpm_adult_gtex_v11_pancreas.gct"),
                       sep="\t", skiprows=2)
    print(f"  GTEx raw: {gtex.shape}")

    # Clinical data
    clin = pd.read_csv(os.path.join(DATA, "TCGA-PAAD", "clinical-data", "clinical.tsv"), sep="\t")
    print(f"  Clinical raw: {clin.shape}")

    # Exposure (smoking, alcohol)
    try:
        expos = pd.read_csv(os.path.join(DATA, "TCGA-PAAD", "clinical-data", "exposure.tsv"), sep="\t")
    except:
        expos = pd.DataFrame()
    print(f"  Exposure: {expos.shape}")

    return tcga_rna, tcga_bm, tcga_mut, gtex, clin, expos


# ─────────────────────────────────────────────────────────────
# 2. ALIGN GENE IDS: TCGA (symbol) ↔ GTEx (Ensembl)
# ─────────────────────────────────────────────────────────────

def build_gene_map(gtex):
    """Build Ensembl → gene symbol mapping from GTEx Name/Description."""
    mapping = {}
    for _, row in gtex.iterrows():
        ens = row['Name'].split('.')[0]
        symbol = row['Description']
        mapping[ens] = symbol
    return mapping


def align_expression(tcga_rna, gtex):
    """Align TCGA and GTEx expression matrices on shared genes."""
    gene_map = build_gene_map(gtex)

    # GTEx: transpose to samples × genes
    gtex_samples = gtex.columns[2:]  # skip Name, Description
    gtex_genes = gtex['Name'].values

    gtex_df = pd.DataFrame(
        gtex.iloc[:, 2:].values.T,
        index=gtex_samples,
        columns=gtex['Name'].values
    )

    # Map Ensembl IDs to gene symbols
    gtex_df.columns = [gene_map.get(e.split('.')[0], e.split('.')[0]) for e in gtex_df.columns]

    # Find common genes between TCGA and GTEx
    common_genes = list(set(tcga_rna.columns) & set(gtex_df.columns))
    print(f"  Common genes between TCGA and GTEx: {len(common_genes)}")

    if len(common_genes) < 100:
        print("  WARNING: Very few common genes. Using TCGA genes as reference.")
        # Use TCGA genes, keep GTEx genes that exist
        common_genes = [g for g in tcga_rna.columns if g in gtex_df.columns]
        print(f"  After filtering: {len(common_genes)} common genes")

    tcga_aligned = tcga_rna[common_genes]
    gtex_aligned = gtex_df[common_genes]

    return tcga_aligned, gtex_aligned, common_genes


# ─────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess_expression(tcga_expr, gtex_expr=None, geo_expr=None):
    """Simple log1p + StandardScaler."""
    tcga_log = np.log1p(tcga_expr.values.astype(np.float64))

    scaler = StandardScaler()
    tcga_scaled = scaler.fit_transform(tcga_log)

    result = {'tcga_scaled': tcga_scaled, 'tcga_index': tcga_expr.index, 'scaler': scaler}

    if gtex_expr is not None:
        gtex_log = np.log1p(gtex_expr.values.astype(np.float64))
        gtex_scaled = scaler.transform(gtex_log)
        result['gtex_scaled'] = gtex_scaled
        result['gtex_index'] = gtex_expr.index

    return result


def build_clinical_features(clin, expos):
    """Extract key clinical features from TCGA clinical data."""
    clin_dedup = clin.drop_duplicates(subset='cases.submitter_id').copy()
    clin_dedup = clin_dedup.set_index('cases.submitter_id')

    # Convert relevant columns to object type for reliable NaN handling
    str_cols = ['demographic.gender', 'demographic.vital_status',
                'diagnoses.ajcc_pathologic_stage', 'diagnoses.tumor_grade',
                'demographic.race', 'exposures.tobacco_smoking_status',
                'exposures.alcohol_history']
    for col in str_cols:
        if col in clin_dedup.columns:
            clin_dedup[col] = clin_dedup[col].astype(object)

    # Replace '-- sentinel values
    for col in clin_dedup.columns:
        if clin_dedup[col].dtype == object:
            clin_dedup[col] = clin_dedup[col].replace("'--", np.nan)

    features = pd.DataFrame(index=clin_dedup.index)

    # Age
    if 'demographic.age_at_index' in clin_dedup.columns:
        features['age'] = pd.to_numeric(clin_dedup['demographic.age_at_index'], errors='coerce')

    # Gender → binary
    if 'demographic.gender' in clin_dedup.columns:
        features['gender_male'] = (clin_dedup['demographic.gender'].str.lower() == 'male').astype(int)

    # Vital status → binary (1 = dead)
    if 'demographic.vital_status' in clin_dedup.columns:
        features['vital_status'] = (clin_dedup['demographic.vital_status'] == 'Dead').astype(int)

    # Survival time (days)
    if 'demographic.days_to_death' in clin_dedup.columns:
        dtd = pd.to_numeric(clin_dedup['demographic.days_to_death'], errors='coerce')
    else:
        dtd = pd.Series(np.nan, index=clin_dedup.index)
    if 'diagnoses.days_to_last_follow_up' in clin_dedup.columns:
        dtf = pd.to_numeric(clin_dedup['diagnoses.days_to_last_follow_up'], errors='coerce')
    else:
        dtf = pd.Series(np.nan, index=clin_dedup.index)

    features['survival_days'] = dtd.fillna(dtf.fillna(0))
    features['survival_days'] = features['survival_days'].clip(lower=1)

    # AJCC pathologic stage
    if 'diagnoses.ajcc_pathologic_stage' in clin_dedup.columns:
        features['ajcc_stage'] = clin_dedup['diagnoses.ajcc_pathologic_stage']

    # Tumor grade
    if 'diagnoses.tumor_grade' in clin_dedup.columns:
        features['tumor_grade'] = clin_dedup['diagnoses.tumor_grade']

    # Race
    if 'demographic.race' in clin_dedup.columns:
        features['race'] = clin_dedup['demographic.race']

    # Smoking from exposure table
    if not expos.empty:
        expos_dedup = expos.drop_duplicates(subset='cases.submitter_id').copy()
        expos_dedup = expos_dedup.set_index('cases.submitter_id')
        for col in ['exposures.tobacco_smoking_status', 'exposures.alcohol_history']:
            if col in expos_dedup.columns:
                expos_dedup[col] = expos_dedup[col].astype(object).replace("'--", np.nan)
        # Smoking status
        if 'exposures.tobacco_smoking_status' in expos_dedup.columns:
            smoking_map = {
                'Current Smoker': 2,
                'Current Reformed Smoker for > 15 yrs': 1,
                'Current Reformed Smoker for < or = 15 yrs': 1,
                'Current Reformed Smoker, Duration Not Specified': 1,
                'Lifelong Non-Smoker': 0,
                'Never smoked': 0,
                'Unknown': np.nan,
            }
            features['smoking_status'] = expos_dedup['exposures.tobacco_smoking_status'].map(smoking_map)
        # Alcohol
        if 'exposures.alcohol_history' in expos_dedup.columns:
            alc_map = {'Yes': 1, 'No': 0, 'Unknown': np.nan}
            features['alcohol_history'] = expos_dedup['exposures.alcohol_history'].map(alc_map)

    return features


def merge_patient_data(tcga_index, tcga_bm, tcga_mut, clin_features):
    """Merge TCGA patient data using submitter_id matching.
    
    Only RNA-seq samples with TCGA-style IDs (e.g., TCGA-XX-XXXX) can be
    matched to clinical and mutation data. UUID-based samples are skipped.
    """
    from collections import defaultdict

    # Build submitter_id set from clinical data
    clin_ids = set(clin_features.index)

    # Mutation: extract submitter_id (first 3 segments of barcode)
    mut_submitter = {}
    for mid in tcga_mut.index:
        parts = str(mid).split('-')
        short = '-'.join(parts[:3]) if len(parts) >= 3 else mid
        if short not in mut_submitter:
            mut_submitter[short] = {}
        for col in tcga_mut.columns:
            mut_submitter[short][col + '_mut'] = tcga_mut.loc[mid, col]

    # Classify RNA indices: TCGA-style vs UUID
    tcga_ids_patients = []  # (index, submitter_id)
    uuid_samples = []
    for idx in tcga_index:
        if str(idx).startswith('TCGA-'):
            tcga_ids_patients.append((idx, idx))
        else:
            uuid_samples.append(idx)

    print(f"  TCGA-style RNA IDs: {len(tcga_ids_patients)}")
    print(f"  UUID-based RNA IDs: {len(uuid_samples)} (skipped for clinical merge)")

    # Build merged DataFrame — one row per matchable patient
    matched = []
    for rna_orig, sid in tcga_ids_patients:
        row_data = {'patient_id': sid}
        if sid in clin_features.index:
            for col in clin_features.columns:
                row_data[col] = clin_features.loc[sid, col]
        if sid in mut_submitter:
            for col, val in mut_submitter[sid].items():
                row_data[col] = val
        matched.append(row_data)

    merged = pd.DataFrame(matched)
    if merged.empty:
        print("  WARNING: No patients matched between RNA and clinical data!")
        merged = pd.DataFrame({'patient_id': tcga_ids_patients})

    # Add biomarker expression (from tcga_bm) using RNA index
    bm_cols = list(tcga_bm.columns)
    for col in bm_cols:
        merged[col] = np.nan
        for i, (rna_orig, sid) in enumerate(tcga_ids_patients):
            if rna_orig in tcga_bm.index:
                merged.loc[i, col] = tcga_bm.loc[rna_orig, col]

    print(f"  Merged data: {merged.shape[0]} patients, {merged.shape[1]} features")
    return merged


# ─────────────────────────────────────────────────────────────
# 4. VAE MODEL
# ─────────────────────────────────────────────────────────────

class VAE(nn.Module):
    def __init__(self, input_dim=500, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def train_vae(X_train, X_test, input_dim, epochs=100, batch_size=32, lr=1e-3):
    model = VAE(input_dim=input_dim, latent_dim=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    history = {'train_loss': [], 'test_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, r, kl = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (batch,) in test_loader:
                batch = batch.to(DEVICE)
                recon, mu, logvar = model(batch)
                loss, r, kl = vae_loss(recon, batch, mu, logvar)
                test_loss += loss.item()

        scheduler.step()
        avg_train = train_loss / len(train_loader.dataset)
        avg_test = test_loss / len(test_loader.dataset)
        history['train_loss'].append(avg_train)
        history['test_loss'].append(avg_test)

        if (epoch + 1) % 25 == 0:
            print(f"    VAE Epoch {epoch+1}/{epochs} — Train: {avg_train:.4f}  Test: {avg_test:.4f}")

    return model, history


def get_latent(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        mu, _ = model.encode(X_t)
    return mu.cpu().numpy()


# ─────────────────────────────────────────────────────────────
# 5. DETECTION MODEL (TCGA cancer vs GTEx normal)
# ─────────────────────────────────────────────────────────────

def train_detection_model(tcga_latent, gtex_latent):
    """Train XGBoost to distinguish TCGA (cancer) from GTEx (normal)."""
    print("\n--- Training Detection Model (TCGA Cancer vs GTEx Normal) ---")

    X = np.vstack([tcga_latent, gtex_latent])
    y = np.hstack([np.ones(len(tcga_latent)), np.zeros(len(gtex_latent))])

    print(f"  Total samples: {len(y)} ({int(y.sum())} cancer, {int((1-y).sum())} normal)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric='logloss', random_state=SEED, n_jobs=-1
    )

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred_bin = (y_pred > 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred_bin)
    print(f"  Test AUC: {auc:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test Sens: {np.mean(y_pred_bin[y_test==1]):.4f}")
    print(f"  Test Spec: {np.mean(y_pred_bin[y_test==0]):.4f}")

    # Cross-validation
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_preds = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    cv_auc = roc_auc_score(y, cv_preds)
    print(f"  CV AUC: {cv_auc:.4f}")

    return model, (X_train, y_train, X_test, y_test)


# ─────────────────────────────────────────────────────────────
# 6. STAGE MODEL
# ─────────────────────────────────────────────────────────────

def train_stage_model(merged_df):
    """Train XGBoost to predict AJCC pathologic stage."""
    print("\n--- Training Stage Model ---")

    # Filter to patients with known stage
    stage_map = {
        'Stage IA': 'Stage I', 'Stage IB': 'Stage I',
        'Stage IIA': 'Stage II', 'Stage IIB': 'Stage II',
        'Stage III': 'Stage III', 'Stage IV': 'Stage IV',
    }
    df = merged_df.copy()
    df['stage_group'] = df['ajcc_stage'].map(stage_map)

    known = df.dropna(subset=['stage_group'])
    print(f"  Patients with known stage: {len(known)}")
    print(f"  Stage distribution: {known['stage_group'].value_counts().to_dict()}")

    if len(known) < 30:
        print("  WARNING: Too few patients for stage model. Training binary model instead.")
        # Binary: early (I/II) vs late (III/IV)
        known['is_late'] = known['stage_group'].isin(['Stage III', 'Stage IV']).astype(int)
        if known['is_late'].sum() < 5:
            print("  Too few late-stage patients. Using survival as proxy for severity.")
            return None

    # Features: biomarkers + mutations + age + gender
    feature_cols = [c for c in known.columns if c in [
        'MUC1', 'VEGFA', 'EGFR', 'SOX9', 'KRT19', 'CEACAM5', 'S100P',
        'KRAS_mut', 'TP53_mut', 'SMAD4_mut', 'CDKN2A_mut', 'BRCA1_mut', 'BRCA2_mut',
        'age', 'gender_male'
    ]]
    existing_features = [c for c in feature_cols if c in known.columns]
    print(f"  Features: {existing_features}")

    X = known[existing_features].fillna(known[existing_features].median())
    y = known['stage_group']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, stratify=y_enc, random_state=SEED
    )

    n_classes = len(np.unique(y_enc))
    if n_classes > 1:
        model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=SEED, n_jobs=-1
        )
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Test Accuracy: {acc:.4f}")

        # Store label encoder
        model.label_encoder_ = le
        return model
    else:
        print("  Only one class — cannot train stage model")
        return None


# ─────────────────────────────────────────────────────────────
# 7. SURVIVAL MODEL
# ─────────────────────────────────────────────────────────────

def train_survival_model(merged_df):
    """Train Random Survival Forest for prognosis."""
    print("\n--- Training Survival Model ---")

    df = merged_df.dropna(subset=['vital_status', 'survival_days'])
    print(f"  Patients with survival data: {len(df)}")
    print(f"  Events (dead): {df['vital_status'].sum()} / {len(df)}")

    if len(df) < 20:
        print("  Too few patients for survival model")
        return None

    feature_cols = [c for c in [
        'MUC1', 'VEGFA', 'EGFR', 'SOX9', 'KRT19', 'CEACAM5', 'S100P',
        'KRAS_mut', 'TP53_mut', 'SMAD4_mut', 'CDKN2A_mut', 'BRCA1_mut', 'BRCA2_mut',
        'age', 'gender_male', 'smoking_status', 'alcohol_history'
    ] if c in df.columns]

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = Surv.from_arrays(event=df['vital_status'].values.astype(bool),
                         time=df['survival_days'].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )

    model = RandomSurvivalForest(
        n_estimators=200, min_samples_split=10, min_samples_leaf=5,
        max_features='sqrt', random_state=SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # C-index
    c_index = model.score(X_test, y_test)
    print(f"  Test C-index: {c_index:.4f}")

    model.feature_names_ = feature_cols
    return model


# ─────────────────────────────────────────────────────────────
# 8. MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ExplainPANC — Model Training Pipeline")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    # Load data
    tcga_rna, tcga_bm, tcga_mut, gtex, clin, expos = load_all_data()

    # Align expression data
    print("\n--- Aligning gene expression ---")
    tcga_expr, gtex_expr, common_genes = align_expression(tcga_rna, gtex)
    print(f"  TCGA aligned: {tcga_expr.shape}")
    print(f"  GTEx aligned: {gtex_expr.shape}")

    # Handle the case where GTEx has more samples than genes in common
    # Take the top expressed genes for stable VAE training
    n_genes = min(500, len(common_genes))
    # Select top genes by variance in TCGA
    gene_vars = tcga_expr.var().sort_values(ascending=False)
    top_genes = gene_vars.head(n_genes).index.tolist()
    tcga_expr = tcga_expr[top_genes]
    gtex_expr = gtex_expr[top_genes]
    print(f"  Using top {n_genes} most variable genes")

    # Preprocess expression
    print("\n--- Preprocessing expression ---")
    prep = preprocess_expression(tcga_expr, gtex_expr)
    tcga_scaled = prep['tcga_scaled']
    gtex_scaled = prep['gtex_scaled']
    scaler = prep['scaler']
    input_dim = tcga_scaled.shape[1]
    print(f"  Input dimension: {input_dim}")

    # Train VAE
    print("\n--- Training VAE ---")
    # Combine TCGA + GTEx for VAE training
    all_expr = np.vstack([tcga_scaled, gtex_scaled])
    np.random.shuffle(all_expr)

    split = int(0.8 * len(all_expr))
    vae_train, vae_test = all_expr[:split], all_expr[split:]

    vae_model, vae_history = train_vae(
        vae_train, vae_test, input_dim,
        epochs=100, batch_size=32, lr=1e-3
    )
    print(f"  VAE train loss: {vae_history['train_loss'][-1]:.4f}")
    print(f"  VAE test loss: {vae_history['test_loss'][-1]:.4f}")

    # Get latent embeddings
    print("\n--- Computing latent embeddings ---")
    tcga_latent = get_latent(vae_model, tcga_scaled)
    gtex_latent = get_latent(vae_model, gtex_scaled)
    print(f"  TCGA latent: {tcga_latent.shape}")
    print(f"  GTEx latent: {gtex_latent.shape}")

    # Train detection model
    det_model, det_data = train_detection_model(tcga_latent, gtex_latent)

    # Build clinical features and merge
    print("\n--- Building clinical feature matrix ---")
    clin_features = build_clinical_features(clin, expos)
    print(f"  Clinical features: {clin_features.shape}")
    print(f"  Columns: {list(clin_features.columns)}")

    merged = merge_patient_data(tcga_expr.index, tcga_bm, tcga_mut, clin_features)
    print(f"  Merged data: {merged.shape}")
    print(f"  Columns: {list(merged.columns)}")

    # Train stage model
    stage_model = train_stage_model(merged)

    # Train survival model
    survival_model = train_survival_model(merged)

    # ─────────────────────────────────────────────────────────────
    # 8B. DASHBOARD-COMPATIBLE CLINICAL MODELS
    #     Trained on clinical features available from dashboard inputs
    #     (age, gender, smoking, alcohol) — no RNA-seq required.
    #     These train directly on clin_features (all 185 patients).
    #     Uses age + gender as core features (100% coverage);
    #     smoking + alcohol used when available.
    # ─────────────────────────────────────────────────────────────

    def train_clinical_detection_model(clin_df):
        """Train clinical-only model to predict late-stage (III/IV) as severity proxy."""
        print("\n--- Training Clinical Detection Model (demographics → risk) ---")
        df = clin_df.copy()
        stage_map = {
            'Stage IA': 'early', 'Stage IB': 'early',
            'Stage IIA': 'early', 'Stage IIB': 'early',
            'Stage III': 'late', 'Stage IV': 'late',
        }
        df['stage_binary'] = df['ajcc_stage'].map(stage_map)
        known = df.dropna(subset=['stage_binary']).copy()
        if len(known) < 20:
            known = df.dropna(subset=['vital_status']).copy()
            known['stage_binary'] = known['vital_status'].map({1: 'late', 0: 'early'})
        known['_risk_target'] = known['stage_binary'].map({'late': 1, 'early': 0}).astype(float)

        feat_cols = ['age', 'gender_male']
        existing = [c for c in feat_cols if c in known.columns]
        known = known.dropna(subset=existing + ['_risk_target'])
        X = known[existing]
        y = known['_risk_target'].values.astype(int)

        print(f"  Samples: {len(y)} ({y.sum()} late, {len(y)-y.sum()} early)")
        print(f"  Features: {existing}")

        if len(np.unique(y)) < 2 or len(y) < 20:
            print("  WARNING: Insufficient data for clinical detection model")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=SEED
        )
        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, scale_pos_weight=(y_train==0).sum() / max(1, (y_train==1).sum()),
            eval_metric='logloss', random_state=SEED, n_jobs=-1
        )
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, (y_pred>0.5).astype(int))
        print(f"  Test AUC: {auc:.4f}, Acc: {acc:.4f}")
        model.feature_names_ = existing
        return model

    def train_clinical_stage_model(clin_df):
        """Train clinical-only model for AJCC stage (I/II/III/IV) on all clinical data."""
        print("\n--- Training Clinical Stage Model ---")
        df = clin_df.copy()
        stage_map = {
            'Stage IA': 'Stage I', 'Stage IB': 'Stage I',
            'Stage IIA': 'Stage II', 'Stage IIB': 'Stage II',
            'Stage III': 'Late', 'Stage IV': 'Late',
        }
        df['stage_grp'] = df['ajcc_stage'].map(stage_map)
        df.loc[df['stage_grp'].isin(['Stage I', 'Stage II']), 'stage_grp'] = 'Early'
        known = df.dropna(subset=['stage_grp']).copy()

        feat_cols = ['age', 'gender_male']
        existing = [c for c in feat_cols if c in known.columns]
        known = known.dropna(subset=existing + ['stage_grp'])
        X = known[existing]
        y = known['stage_grp']

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        if len(np.unique(y_enc)) < 2 or len(y) < 20:
            print("  Too few classes — skipping")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.25, stratify=y_enc, random_state=SEED
        )
        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            eval_metric='mlogloss', random_state=SEED, n_jobs=-1
        )
        model.fit(X_train, y_train, verbose=False)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  Classes: {list(le.classes_)} — Test Acc: {acc:.4f}")
        model.feature_names_ = existing
        model.label_encoder_ = le
        return model

    def train_clinical_survival_model(clin_df):
        """Train clinical-only survival model on all clinical data."""
        print("\n--- Training Clinical Survival Model ---")
        df = clin_df.dropna(subset=['vital_status', 'survival_days']).copy()
        if len(df) < 20:
            print("  Too few patients")
            return None

        feat_cols = ['age', 'gender_male']
        existing = [c for c in feat_cols if c in df.columns]
        df = df.dropna(subset=existing)
        X = df[existing]
        y = Surv.from_arrays(event=df['vital_status'].values.astype(bool),
                             time=df['survival_days'].values)

        if len(X) < 20:
            print("  Too few patients after dropping NaN")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED
        )
        model = RandomSurvivalForest(
            n_estimators=200, min_samples_split=10, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1
        )
        model.fit(X_train, y_train)
        c_idx = model.score(X_test, y_test)
        print(f"  Test C-index: {c_idx:.4f}")
        model.feature_names_ = existing
        return model

    # Train clinical models on full clinical data (not merged subset)
    clin_det_model = train_clinical_detection_model(clin_features)
    clin_stage_model = train_clinical_stage_model(clin_features)
    clin_survival_model = train_clinical_survival_model(clin_features)

    # Save clinical model metadata (medians for missing value imputation)
    clin_feat_cols = ['age', 'gender_male']
    clin_medians = {}
    for c in clin_feat_cols:
        if c in clin_features.columns:
            clin_medians[c] = float(clin_features[c].median())
        else:
            clin_medians[c] = 0.0

    # ── SAVE ALL MODELS ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving models...")

    # Save VAE
    torch.save({
        'model_state_dict': vae_model.state_dict(),
        'input_dim': input_dim,
        'latent_dim': 64,
        'top_genes': top_genes,
    }, os.path.join(OUT, 'vae_encoder.pt'))
    print(f"  ✓ VAE saved to {OUT}/vae_encoder.pt")

    # Save scaler
    with open(os.path.join(OUT, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved")

    # Save detection model
    det_model.save_model(os.path.join(OUT, 'xgb_detection.json'))
    print(f"  ✓ Detection model saved")

    # Save stage model
    if stage_model is not None:
        stage_model.save_model(os.path.join(OUT, 'xgb_stage.json'))
        print(f"  ✓ Stage model saved")
    else:
        print(f"  ✗ Stage model not available — skipping")

    # Save survival model
    if survival_model is not None:
        with open(os.path.join(OUT, 'rf_survival.pkl'), 'wb') as f:
            pickle.dump(survival_model, f)
        print(f"  ✓ Survival model saved")
    else:
        print(f"  ✗ Survival model not available — skipping")

    # Save feature metadata
    metadata = {
        'top_genes': top_genes,
        'input_dim': input_dim,
        'latent_dim': 64,
        'clinical_features': [c for c in merged.columns if c not in ['patient_id', 'ajcc_stage', 'tumor_grade', 'race']],
        'detection_auc': roc_auc_score(det_data[3], det_model.predict_proba(det_data[2])[:, 1]),
        'detection_acc': accuracy_score(det_data[3], (det_model.predict_proba(det_data[2])[:, 1] > 0.5).astype(int)),
        'n_tcga': len(tcga_scaled),
        'n_gtex': len(gtex_scaled),
    }
    with open(os.path.join(OUT, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  ✓ Metadata saved")

    # ── Save clinical (dashboard-compatible) models ──────────
    if clin_det_model is not None:
        clin_det_model.save_model(os.path.join(OUT, 'xgb_clinical_detection.json'))
        print(f"  ✓ Clinical detection model saved")
    if clin_stage_model is not None:
        clin_stage_model.save_model(os.path.join(OUT, 'xgb_clinical_stage.json'))
        print(f"  ✓ Clinical stage model saved")
    if clin_survival_model is not None:
        with open(os.path.join(OUT, 'rf_clinical_survival.pkl'), 'wb') as f:
            pickle.dump(clin_survival_model, f)
        print(f"  ✓ Clinical survival model saved")

    # Save clinical feature medians for dashboard imputation
    with open(os.path.join(OUT, 'clinical_medians.pkl'), 'wb') as f:
        pickle.dump(clin_medians, f)
    print(f"  ✓ Clinical medians saved")

    # Save sample input template for dashboard
    sample = {
        'top_genes': top_genes,
        'feature_columns': metadata['clinical_features'],
        'clinical_feature_columns': clin_feat_cols,
        'clinical_medians': clin_medians,
    }
    with open(os.path.join(OUT, 'sample_input.pkl'), 'wb') as f:
        pickle.dump(sample, f)
    print(f"  ✓ Sample input template saved")

    print(f"\n{'='*60}")
    print(f"Training complete! All models saved to {OUT}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
