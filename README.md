# ExAI-PanC

ExAI-PanC is a research prototype for an explainable pancreatic-cancer clinical digital twin. The current application combines a Next.js/React frontend, a FastAPI backend, clinical prediction workflows, CT/DICOM/NIfTI processing, interactive 3D anatomy, simulations, and report-oriented API routes.

> **Research use only.** This software is not a medical device and must not be used for diagnosis, treatment selection, or other clinical decisions. Outputs require qualified clinician and researcher review.

## Current application

- **Frontend:** Next.js 14, React 18, TypeScript, Three.js / React Three Fiber.
- **Backend:** FastAPI with SQLAlchemy async database access, local JWT authentication, imaging workflows, prediction routes, digital-twin routes, simulation, treatment, reporting, and workflow routes.
- **Default local URLs:** frontend at `http://localhost:3000`; API at `http://127.0.0.1:8765`.
- **API documentation:** `http://127.0.0.1:8765/docs` and `http://127.0.0.1:8765/redoc`.
- **Health check:** `http://127.0.0.1:8765/api/v1/health`.
- **3D modes:** realistic anatomy, real CT mesh, VTK CT mesh, uploaded DICOM volume, uploaded NIfTI pair, and uploaded patient-folder processing.
- **Prediction behavior:** the main `/api/v1/predictions/run` route uses the trained model artifacts when the optional ML packages are available and falls back to a deterministic heuristic implementation if they are not. The fallback is for demonstration only.

## Repository layout

```text
ExAI-PanC/
├── backend/
│   ├── main.py                    FastAPI application entry point
│   ├── requirements.txt            Core backend dependencies
│   ├── prediction_engine.py        Legacy/model-backed prediction engine
│   ├── app/
│   │   ├── config.py               Environment-backed settings and paths
│   │   ├── database.py             Async SQLAlchemy engine and DB initialization
│   │   ├── core/                   Auth, security, dependencies, validation
│   │   ├── domain/                 Workflow and domain enums
│   │   ├── models/                 Database models
│   │   ├── routers/                API route groups
│   │   └── services/               Imaging, mesh, CT, biomarker, and export services
│   └── data/
│       ├── exai_panc.db            Generated local SQLite database; do not commit
│       └── uploads/                Generated uploaded studies; do not commit
├── digital_twin/
│   ├── anatomy/                    CT, segmentation, meshes, organ loading/rendering
│   ├── config/                     Model and anatomy configuration
│   ├── core/                       Twin state, simulation, and recommendations
│   ├── explainability/             SHAP and biomarker explainability helpers
│   ├── pipeline/                   Patient, preprocessing, and feature pipelines
│   ├── prediction/                 Prediction abstractions and ensembles
│   ├── simulation/                 Disease progression and intervention simulation
│   ├── visualization/              Plotly/visualization helpers
│   └── vtk_engine.py               Physical-coordinate CT/NIfTI mesh pipeline
├── frontend/
│   ├── app/                        Next.js app entry points
│   ├── src/components/             Clinical UI and 3D viewers
│   ├── src/engine/                 Imaging, twin, simulation, timeline, and XAI state
│   ├── package.json                Frontend scripts and dependencies
│   └── package-lock.json           Locked frontend dependency versions
├── models/
│   ├── *.pt, *.pkl, *.json         Trained prediction artifacts and metadata
│   └── meshes/                     Optional anatomical mesh assets/cache
├── DATASETS/MSD_Pancreas/          Downloaded external CT/label data; not committed
├── notebooks/                      Publication/evaluation notebook and outputs
├── .env.example                    Local configuration template
├── .gitignore                      Runtime, secret, build, and data exclusions
└── README.md
```

## Requirements

Install these before starting:

- Git
- Python 3.10 or newer
- Node.js 18 or newer with npm
- Optional for the full model/imaging stack: a compatible PyTorch installation, XGBoost, scikit-learn, nibabel, scikit-image, and VTK

The core dependency file intentionally keeps some heavy ML and medical-imaging packages optional. This allows the API and frontend to start with the safe fallback prediction path.

## Clone the project

Replace the placeholder URL with the GitHub repository URL:

```powershell
git clone https://github.com/<your-account>/<your-repository>.git
cd <your-repository>
```

For an existing clone:

```powershell
git pull origin main
```

## Backend setup

Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
Copy-Item .env.example .env
New-Item -ItemType Directory -Force backend\data\uploads | Out-Null
```

If PowerShell blocks environment activation, either activate the environment from a shell permitted by your machine policy or run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Optional backend packages

Install only the packages needed for the feature being used:

```powershell
# Trained model-backed predictions and the legacy prediction endpoint
pip install torch xgboost scikit-learn

# NIfTI upload and marching-cubes/VTK CT workflows
pip install nibabel scikit-image vtk

# MONAI or nnU-Net DICOM segmentation engines, only when a compatible model is available
pip install monai SimpleITK nnunetv2
```

PyTorch installation can vary by CPU, CUDA version, operating system, and Python version. For GPU use, install the matching command from the official PyTorch instructions before installing the other optional packages.

### Environment configuration

`.env.example` is safe to copy and edit locally. Never commit `.env`.

Important settings include:

| Variable | Local default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./backend/data/exai_panc.db` | Local async SQLite database |
| `JWT_SECRET_KEY` | Development placeholder | Replace before any shared or deployed use |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Access-token lifetime |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | `7` | Refresh-token lifetime |
| `MODELS_DIR` | `./models` | Trained model artifact directory |
| `UPLOAD_DIR` | `./backend/data/uploads` | Runtime study and mesh directory |
| `CORS_ORIGINS` | Localhost frontend origins | Intended CORS configuration |
| `AUTH_PROVIDER` | `local` | Development authentication provider |

The current `backend/main.py` also contains a hard-coded local CORS allow-list. If the frontend is hosted on another origin, update both `NEXT_PUBLIC_API_URL` and the backend CORS middleware before deployment.

The database tables are created automatically during API startup. The SQLite file is generated under `backend/data/` and is ignored by Git.

## Frontend setup

Open a second terminal from the repository root:

```powershell
cd frontend
npm ci
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8765/api/v1"
npm run dev
```

If the API is running at the default URL, the environment variable can be omitted because the frontend already defaults to `http://localhost:8765/api/v1`.

Open <http://localhost:3000> after Next.js starts.

Available frontend scripts:

```powershell
npm run dev        # Development server
npm run typecheck  # TypeScript validation
npm run build      # Production build
npm start          # Serve the production build
```

On Windows systems where PowerShell script execution blocks `npm`, use `npm.cmd` in the commands above.

## Start the application end to end

1. Start the backend from the project root:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
   ```

   Add `--reload` during development if automatic backend reloads are needed.

2. Verify the API:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8765/api/v1/health
   ```

3. Start the frontend in a second terminal:

   ```powershell
   cd frontend
   npm run dev
   ```

4. Open <http://localhost:3000> and use the API docs at <http://127.0.0.1:8765/docs> to inspect or exercise endpoints.

## Development login

Local authentication is currently an in-memory development store in `backend/app/routers/auth.py`; users and registered accounts reset whenever the backend process restarts. The built-in demo account is:

```text
username: demo
password: Demo123!
```

Additional development accounts are defined in that router. These credentials must not be used in a deployed environment. Replace the in-memory store, rotate the JWT secret, and configure persistent authentication before sharing the application with real users or patient data.

## Data sources and runtime paths

### MSD Task07 Pancreas data

The 3D CT examples use the Medical Segmentation Decathlon Task07 Pancreas data. The application expects this local layout:

```text
DATASETS/MSD_Pancreas/
├── imagesTr/
│   ├── pancreas_001.nii.gz
│   ├── pancreas_010.nii.gz
│   └── ...
├── labelsTr/
│   ├── pancreas_001.nii.gz
│   ├── pancreas_010_label.nii.gz
│   └── ...
└── meshes_<case>.npz             Optional pre-extracted mesh cache
```

Both label naming conventions are supported: `pancreas_<case>.nii.gz` and `pancreas_<case>_label.nii.gz`. The VTK pipeline uses label `1` for pancreas and label `2` for tumour, as defined in `digital_twin/vtk_engine.py`.

The current working copy contains a small subset of cases (`001`, `010`, `219`, `409`, and `414`) and pre-extracted mesh caches. A fresh clone does not receive these raw files; download Task07 Pancreas separately from the [Medical Segmentation Decathlon download page](https://medicaldecathlon.com/dataaws/) and follow its attribution and license terms. The dataset registry identifies the dataset license as CC BY-SA 4.0; verify the upstream terms before redistribution.

### Models and mesh assets

- `models/` contains the prediction artifacts used by the application, including `vae_encoder.pt`, XGBoost JSON files, survival-model pickle files, `scaler.pkl`, `clinical_medians.pkl`, and `metadata.json`.
- `models/meshes/` contains large optional anatomical assets and generated processed caches. It is ignored by default in `.gitignore`; provide these assets separately when the realistic mesh view needs them. The code has procedural fallbacks for missing optional meshes.
- Do not place secrets, patient records, DICOM files, or local databases in `models/` or the repository.

### Runtime-generated files

```text
backend/data/exai_panc.db              SQLite database created at startup
backend/data/uploads/<patient>/<study>/ Uploaded DICOM files and study artifacts
backend/data/uploads/.../study.json    Imaging workflow metadata
backend/data/uploads/.../segmentation.npy
backend/data/uploads/.../tumour_mesh.json
```

These files can contain sensitive medical information and are ignored by Git. Delete or secure them according to your institutional data-retention policy.

### Notebooks and evaluation data

`notebooks/publication_results.ipynb` reads from `DATASETS/` and writes publication/evaluation outputs under `notebooks/publication_outputs/`. Some output templates explicitly report that metrics are not estimable when `DATASETS/publication_evaluation.csv` is absent. Do not describe those templates as completed independent evaluation results.

## Main API route groups

All routes below are under `/api/v1`:

| Route group | Purpose |
|---|---|
| `/health` | Service health and version |
| `/auth` | Login, registration, refresh, current user |
| `/patients` | Patient records and patient-linked data |
| `/patients/{patient_id}/clinical` | Clinical information |
| `/patients/{patient_id}/labs` | Laboratory values |
| `/patients/{patient_id}/genomics` | Genomic/biomarker data |
| `/patients/{patient_id}/imaging` | DICOM study upload, processing, and study status |
| `/predictions` | Prediction and prediction history |
| `/twin` | Patient twin generation and simulation |
| `/twin/mesh` | Realistic, real-CT, VTK, NIfTI, DICOM-volume, and folder mesh workflows |
| `/simulation` | Disease and intervention simulation |
| `/treatment` | Treatment options, recommendations, comparisons, and records |
| `/patients/{patient_id}/reports` | Report generation and export |
| `/workflow` | End-to-end workflow state |

The OpenAPI page at `/docs` is the authoritative request/response reference for the current implementation.

## Upload and 3D workflow notes

- **Existing MSD case:** use `/api/v1/twin/mesh/real-ct?case=409` for a pre-extracted mesh cache, or `/api/v1/twin/mesh/vtk-ct?case=409` when VTK and the matching NIfTI files are available.
- **DICOM CT series:** upload at least two slices to the DICOM volume endpoint. The backend validates and reconstructs the series while preserving physical spacing.
- **NIfTI pair:** upload a matching CT `.nii`/`.nii.gz` file and label `.nii`/`.nii.gz` file to the NIfTI endpoint. Dimensions, spacing, and origin must match.
- **Patient folder:** the folder processor can discover `imagesTr/*.nii.gz`, `labelsTr/*.nii.gz`, and/or a pre-extracted `.npz` mesh.
- **AI segmentation:** MONAI and nnU-Net routes require an actual compatible model checkpoint. The repository does not claim to include a production segmentation checkpoint.

## Testing and checks before a pull request

From the repository root:

```powershell
# Backend import check
.\.venv\Scripts\python.exe -c "import backend.main; print('backend import OK')"

# Frontend checks
cd frontend
npm run typecheck
npm run build
cd ..

# Repository status and ignored-file review
git status --short
git status --ignored --short
```

If optional packages are installed, also manually verify `/docs`, `/api/v1/health`, login, prediction, and the 3D route needed by the change. Do not upload real patient data to a public issue, pull request, CI artifact, or public demo.

## GitHub publishing workflow

Before the first push:

1. Confirm that `.env`, `.venv`, `frontend/node_modules`, `frontend/.next`, `backend/data`, and `DATASETS` are ignored.
2. Confirm that no DICOM, NIfTI, database, upload, credential, or secret file is staged.
3. Confirm that trained model artifacts and optional mesh assets have redistribution permission. If an asset is not redistributable, remove it from Git and document how to obtain it.
4. Review the staged file list and diff:

   ```powershell
   git add README.md .gitignore backend digital_twin frontend models notebooks
   git status --short
   git diff --cached --stat
   git diff --cached -- .env .env.example README.md .gitignore
   ```

5. Commit and push:

   ```powershell
   git commit -m "Document current ExAI-PanC setup"
   git branch -M main
   git remote add origin https://github.com/<your-account>/<your-repository>.git
   git push -u origin main
   ```

If `origin` already exists, use `git remote -v` and push to the intended repository instead of adding it again.

## Should the dataset be pushed to GitHub?

**Recommended answer: keep the raw dataset outside the Git repository and make users download it separately.** The current local MSD subset is approximately 462 MB, and at least one raw NIfTI file is larger than GitHub's enforced 100 MiB single-object limit. A normal `git push` will therefore fail or create an unhealthy repository.

Use this approach:

1. Keep `DATASETS/` ignored.
2. Add the official download page, dataset citation, license, expected folder layout, and a small version/checksum manifest to the README or a separate data document.
3. Let each user download the dataset from the upstream source and extract it into `DATASETS/MSD_Pancreas/`.
4. If a fixed research snapshot must be distributed, publish it through an appropriate research/data repository or institutional object storage with access controls, versioning, checksums, and the required license/attribution.
5. Use Git LFS only after checking the dataset terms and storage/bandwidth costs. Git LFS solves GitHub's large-file transfer mechanism; it does not remove licensing, privacy, or medical-data governance responsibilities.

The same rule applies to private clinical DICOM studies: never put them in a public GitHub repository. De-identification, consent, access control, and institutional approval are required before any sharing.

## Troubleshooting

### Frontend cannot reach the API

Check that the backend is running on port `8765`, then set:

```powershell
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8765/api/v1"
```

For a non-local frontend origin, update the backend CORS allow-list in `backend/main.py` as well.

### Prediction uses fallback output

Install the optional ML dependencies and restart the backend:

```powershell
pip install torch xgboost scikit-learn
```

Fallback output is intentionally available for local UI development, but it is not equivalent to a validated model run.

### Real CT or NIfTI processing fails

Check that the requested case exists under `DATASETS/MSD_Pancreas/`, the CT and label volumes match, and the required optional packages (`nibabel`, `scikit-image`, or `vtk`) are installed for the selected endpoint.

### Database or upload errors occur

Make sure the backend was started from the repository root and that `backend/data/uploads/` exists and is writable. Never solve an upload problem by committing the generated files.

## License and attribution

Add the project's software license before publishing. Separately retain the license and citation requirements for MSD Task07 Pancreas and any third-party anatomical mesh assets. A software license for this repository does not automatically grant redistribution rights for datasets, model weights, or mesh files.
