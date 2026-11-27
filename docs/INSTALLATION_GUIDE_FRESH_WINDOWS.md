# FraudLens — Fresh Windows Installation & Setup Guide

Welcome! This guide walks you step by step through cloning **FraudLens — Intelligent Transaction Risk Analysis** onto a brand-new Windows 10/11 laptop and launching it with the pre-trained Graph Neural Network (**`ExpA2_RankGauss182_BiDirGCN`**).

**You do NOT need to write any code or train any model.** Everything needed for inference and explainability is already inside this GitHub repository.

---

## 1. Required Software (Install First)

Install these three free tools before cloning:

1. **Git for Windows:** [https://git-scm.com/download/win](https://git-scm.com/download/win)  
   - Keep default options during installation.
2. **Python 3.10 – 3.12 (Recommended: Python 3.12):** [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)  
   - **CRITICAL:** On the first screen of the Python installer, check **"Add python.exe to PATH"** before clicking Install Now!
3. **Node.js 18+ LTS (Verified on v24.15.0, includes `npm`):** [https://nodejs.org/](https://nodejs.org/)  
   - Choose the **LTS** installer with default options.
4. **Modern Web Browser:** Google Chrome or Microsoft Edge.

### Quick Check in PowerShell
Open **PowerShell** (Start Menu -> type `PowerShell`) and verify all tools work:
```powershell
git --version
python --version
node --version
npm --version
```
You should see versions printed cleanly for each command (`Python 3.12.x`, `v18+` or `v24.x` for Node).

---

## 2. Clone the Repository from GitHub

Choose a clean folder (for example `Documents` or `Downloads`), open PowerShell there, and clone:

```powershell
git clone https://github.com/Mansi-Upadhyay-12/Capstone_Fraud_Detection.git
cd Capstone_Fraud_Detection
```

*(Note: All required model checkpoints and preprocessing pipelines are stored directly in Git—Git LFS is NOT required.)*

---

## 3. Create Python Virtual Environment & Install Backend Dependencies

Run these exact commands from inside `Capstone_Fraud_Detection`:

```powershell
# Step 1: Create an isolated virtual environment named 'venv'
python -m venv venv

# Step 2: Activate the virtual environment
.\venv\Scripts\activate
```

> **Note if PowerShell blocks `activate`:**  
> If you see `running scripts is disabled on this system`, either run this once:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> .\venv\Scripts\activate
> ```
> OR simply skip activation and use `.\venv\Scripts\python.exe` directly in the next command!

Now install the Python backend and deep learning libraries (`FastAPI`, `PyTorch`, `torch-geometric`, `scikit-learn`, `pandas`, etc.):

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r app\backend\requirements.txt
```

Wait 2–5 minutes while `pip` downloads and installs PyTorch and dependencies cleanly inside `venv\`.

---

## 4. Install Frontend Dependencies

From the exact same `Capstone_Fraud_Detection` project root folder, run:

```powershell
npm --prefix app/frontend install
```

Wait 30–90 seconds while `npm` installs `React`, `Vite`, `Recharts`, `Cytoscape.js`, `Axios`, and `Lucide React` into `app\frontend\node_modules\`.

*(Optional Configuration: The frontend automatically connects to `http://127.0.0.1:8010`. Only copy `app/frontend/.env.example` to `app/frontend/.env.local` if you ever change the backend host/port.)*

---

## 5. Verify Model & Preprocessing Artifacts Are Present

Unlike the raw `663 MB` Elliptic++ training CSV (`dataset/`), **all frozen inference artifacts are already included in this repository**:

- Frozen Model Checkpoint: `models/stage4h/ExpA2_RankGauss182_BiDirGCN_seed42.pt` (`0.62 MB`)
- Stage 4H RankGauss Normalizer: `models/stage4h/rankgauss_transformer_train_1_34.joblib` (`1.40 MB`)
- Stage 2 Median Imputer + Scaler: `app/backend/artifacts/stage2_imputer_scaler_train_1_34.joblib` (`6 KB`)
- Canonical 182 Feature Schema: `processed/metadata.json` (`7 KB`)
- Ready-to-Upload Sample CSVs: `app/sample_data/transactions_sample.csv`, `edges_sample.csv`, `classes_optional_sample.csv`

And **database creation is automatic**: on first backend startup, FraudLens automatically creates `app/data/fraud_app.db` (SQLite) and `app/data/runs_cache/`.

---

## 6. Launch FraudLens

### Option A — One-Click Launcher (Recommended)
In Windows File Explorer, double-click **`start_fraudlens.bat`** inside `Capstone_Fraud_Detection\` — OR run in PowerShell:

```powershell
.\start_fraudlens.bat
```

Two terminal windows will open automatically:
1. **FraudLens Backend (`127.0.0.1:8010`)** — FastAPI + PyTorch frozen inference server.
2. **FraudLens Frontend (`127.0.0.1:5173`)** — Vite React dashboard.

### Option B — Manual Two-Terminal Launch
- **Terminal 1 (Backend):** `.\start_backend.bat`
- **Terminal 2 (Frontend):** `.\start_frontend.bat`

Wait ~5–10 seconds until the Backend terminal prints `Uvicorn running on http://127.0.0.1:8010` and the Frontend terminal prints `Local: http://127.0.0.1:5173/`.

---

## 7. Open URLs in Your Browser

- **FraudLens Web Application:** [http://127.0.0.1:5173](http://127.0.0.1:5173)
- **Interactive API Docs (Swagger UI):** [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs)
- **Backend Health Check JSON:** [http://127.0.0.1:8010/api/health](http://127.0.0.1:8010/api/health)

On opening **`http://127.0.0.1:5173`**, verify that the top-right status pill turns green/teal and reads:
**`API Online • Threshold 0.57`**.

---

## 8. Run Your First Sample Fraud Analysis

1. In the left sidebar, click **Upload CSV Bundle** (or **New CSV Analysis** on the Dashboard).
2. *(Optional)* Enter a friendly Run Label such as `first_sample_run`.
3. Under **1. transactions.csv**, click Browse (or drag-and-drop) and select:
   `app\sample_data\transactions_sample.csv`
4. Under **2. edges.csv**, select:
   `app\sample_data\edges_sample.csv`
5. *(Optional)* Expand **Optional File 3: Separate Ground-Truth classes.csv** and select:
   `app\sample_data\classes_optional_sample.csv`
6. Click **`1. Validate CSV Bundle`** -> confirm the green **Validation Passed** box appears (`13` transactions, `12` unique directed edges, Timestep `42`).
7. Click **`2. Run Fraud Analysis`** -> wait ~1 second to be taken to **Run Details**.
8. Click **`Inspect Suspicious Clusters`** -> click **`Open Cluster`** on **`C_T42_001`** to view the interactive directed **Cytoscape.js** fraud-ring candidate graph centered around hub transaction **`73282397`**.
9. Click node **`73282397`**, click **`Inspect & Explain Transaction (IG)`**, and then click **`Explain Prediction (IG)`** to view live `M=200` Integrated Gradients feature attributions!

---

## 9. Optional Self-Tests on Your Fresh Laptop

If you want to verify backend and frontend health via command line:

```powershell
# Standalone Backend API, validation, clustering & leakage isolation tests (7 tests):
.\venv\Scripts\python.exe -m pytest -v app/backend/tests/test_stage6b_backend.py -k "not (end_to_end_run_queries or numerical_parity)"

# Frontend React & XAI/Cytoscape Vitest suite (8 tests):
npm --prefix app/frontend test
```

*(Note: The 3 additional parity tests in `test_stage6b_backend.py` require the optional `663 MB` raw Elliptic++ dataset under `dataset/` and `processed/temporal_snapshots.pt`, which are excluded from GitHub due to GitHub's 100 MB file limit. The FraudLens app itself does NOT need those files.)*

---

## 10. Common Installation Errors & Quick Solutions

| Problem | Why It Happens | Solution |
| :--- | :--- | :--- |
| `'python' is not recognized` | Python was installed without checking "Add to PATH", or Windows Store alias intercepted it. | Re-run Python 3.12 installer -> Modify -> check **Add python.exe to PATH**, or disable App Execution Aliases for `python.exe` in Windows Settings. |
| `running scripts is disabled on this system` when running `activate` | Default Windows PowerShell execution policy restricts `.ps1` scripts. | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or run `.\venv\Scripts\python.exe ...` directly without activating. |
| Top-right pill shows **`Backend Offline`** | Backend server window is closed, crashed, or blocked on port `8010`. | Confirm `start_backend.bat` window is open and says `Uvicorn running on http://127.0.0.1:8010`, then click the `↺` refresh button beside the pill. |
| Port `8010` or `5173` already in use (`Address already in use`) | An old terminal from a previous session is still running. | Close old Command Prompt/PowerShell windows running Uvicorn/Vite, or inspect with `netstat -ano \| findstr :8010`. |
| `pip install torch` takes long or fails on disk space | PyTorch packages require ~1.5–3 GB free space while unpacking. | Ensure at least `5 GB` free disk space on `C:\` and stable internet, then re-run `.\venv\Scripts\python.exe -m pip install -r app\backend\requirements.txt`. |
| CSV upload rejected with **`HTTP 422`** | Custom CSV is missing one of the 182 feature columns, has text in a numeric cell, or has dangling/cross-timestep edges. | Compare your CSV headers against [`app/sample_data/transactions_sample.csv`](../app/sample_data/transactions_sample.csv) and [`CSV_FORMAT_AND_USER_GUIDE.md`](CSV_FORMAT_AND_USER_GUIDE.md). |
