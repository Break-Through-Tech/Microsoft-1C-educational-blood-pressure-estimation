# Microsoft 1C – Cuffless Blood Pressure Estimation Model

Estimating blood pressure from photoplethysmography (PPG) signals using machine learning — a Break Through Tech AI Studio project hosted by Microsoft.

---

## 👥 Team Members

| Name           | GitHub Handle                                          | Contribution                                                            |
|----------------|--------------------------------------------------------|-------------------------------------------------------------------------|
| Taylor Nguyen  | [@taylornguyen](https://github.com/taylornguyen)       | Data exploration, visualization, overall project coordination           |
| Claire Zhu     | [@czhu1231](https://github.com/czhu1231)               | Data collection, exploratory data analysis (EDA), dataset documentation |
| Katherine Shih | [@katherineShih113](https://github.com/katherineShih113) | Data preprocessing, feature engineering, data validation              |
| Sri Lakshmi Anbarasan| [@srilakshmi-anbarasan](https://github.com/srilakshmi-anbarasan) | Data preprocessing, feature engineering, data validation|
| Charis Xiong   | [@karrixxa](https://github.com/karrixxa)               | Data preprocessing, exploration, visualization, model evaluation        |
| Chris Park     | [@chrispark](https://github.com/chrispark)             | Model evaluation, performance analysis, results interpretation          |
| Emily Ta      | [@emilyta788](https://github.com/emilyta788)             | Model evaluation, performance analysis, results interpretation          |
| Sophia Lu     | [@ssophiaa-lu](https://github.com/ssophiaa-lu)          | Data visualization, model evaluation, results interpretation            |

---

## 🎯 Project Highlights

- Explored PPG, arterial blood pressure (ABP), and ECG recordings from the UCI cuffless blood-pressure dataset
- Divided recordings into five-second windows (625 samples at 125 Hz) for analysis
- Examined signal quality, pulse characteristics, and blood-pressure labels using NeuroKit2

---

## 👩🏽‍💻 Setup and Installation

> 🚧 *To be completed.*

```bash
# 1. Clone the repository
git clone https://github.com/<org-or-user>/<repo-name>.git
cd <repo-name>

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Dataset:** Download the [UCI Cuff-Less Blood Pressure Estimation dataset](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation) and place the `.mat` files in a `data/` folder.

**Run:** Open `<notebook-name>.ipynb` in Jupyter or VS Code and run all cells.

---

## 🌐 Deploying the App on Render

The web app (FastAPI backend + React frontend) runs in one Docker container (`Dockerfile`), described for
Render in `render.yaml`. Locally, the same app starts with one command, which builds the frontend if
`frontend/dist` is missing:

```bash
python scripts/start.py            # http://localhost:8080; add --rebuild to force a fresh frontend build
```

Without trained results the app runs in **demo mode** (synthetic signals, placeholder models, a "Demo data"
badge). With them it runs in **real mode**. Missing or broken result files never stop it from starting; it
falls back to demo mode and `/api/health` says why.

### 1. Package the trained results (once per new set of results)

Render builds only from git, and the six result files are deliberately not in git. The app downloads them
at startup from a private URL instead. Pack them into one zip:

```bash
.venv/Scripts/python -m src.results_bundle pack       # writes results-bundle.zip, prints its SHA-256
```

| File in the zip | Size | What it is |
|---|---|---|
| `artifacts/resnet/resnet_bp_v3.pt` | 1.2 MB | ResNet checkpoint |
| `artifacts/baseline/mean_baseline.json` | 1 KB | Mean baseline |
| `results/best_model.json` | 2 KB | Which model is primary, and why |
| `results/metrics.csv` | 1 KB | MAE / RMSE per model and split |
| `results/residuals.csv` | 18 MB | Validation/test errors, used to size the 90% ranges |
| `data/test_samples.npz` | 1 MB | The held-out sample recordings shown on Predict and Explain |

### 2. Put the zip somewhere private

Recommended: a release in a **private** GitHub repo (for example `<you>/cufflessai-results`).

1. Create the private repo on GitHub, then **Releases → Draft a new release**, tag `v1`, attach
   `results-bundle.zip`, publish. (Or: `gh release create v1 results-bundle.zip --repo <you>/cufflessai-results`.)
2. Get the asset's API URL:
   `gh api repos/<you>/cufflessai-results/releases/tags/v1 --jq ".assets[0].url"`
   It looks like `https://api.github.com/repos/<you>/cufflessai-results/releases/assets/123456789`.
3. Create a **fine-grained personal access token** (GitHub → Settings → Developer settings) with access to
   only that repo and **Contents: Read-only**.

Any other private HTTPS link to the zip works too; the token is sent as `Authorization: Bearer <token>`.

### 3. Create the service on Render

1. On [render.com](https://render.com), choose **New → Blueprint** and connect this GitHub repository.
   Render reads `render.yaml`: a free Docker web service built from the `app` branch, with a health check.
   (Connecting a repo owned by a GitHub organization may need an org admin to approve Render's GitHub app.)
2. When asked for the environment variables, enter:

   | Key | Value |
   |---|---|
   | `CUFFLESS_RESULTS_URL` | the asset API URL from step 2 |
   | `CUFFLESS_RESULTS_TOKEN` | the fine-grained token |
   | `CUFFLESS_RESULTS_SHA256` | the SHA-256 printed in step 1 |

3. Deploy. The first build takes several minutes. Then open `https://<service>.onrender.com/api/health`:
   it should say `"mode": "real"`. If it says `"demo"`, the service log shows a `[results]` line explaining
   why (wrong URL or token, or a checksum mismatch).

The checksum matters: `.pt` checkpoints are loaded with `weights_only=False`, so the app refuses a bundle
whose SHA-256 doesn't match instead of loading a swapped file.

### 4. Updating

- **Code:** push to `app`. Render rebuilds and redeploys automatically.
- **New results** (for example, after a teammate's model is added and `scripts/evaluate.py` is rerun): pack a
  new zip, attach it to a new release, then update `CUFFLESS_RESULTS_URL` and `CUFFLESS_RESULTS_SHA256` in
  the Render dashboard. Saving the variables redeploys the service.

Notes:
- Free Render services sleep after a period without visitors; the first visit afterwards takes up to a
  minute while the container starts and loads PyTorch. The results are downloaded again on each start.
- The app uses about 330 MB of memory (free instances have 512 MB): one worker, one inference thread.
- Every page shows the "not a medical device" disclaimer. Uploaded signals and camera data are processed in
  memory and never stored.

---

## 🏗️ Project Overview

This project explores whether blood pressure can be estimated from **photoplethysmography (PPG)**, an optical pulse signal that can be captured by wearable devices. PPG serves as the model input, while the corresponding arterial blood-pressure (ABP) waveform provides systolic and diastolic pressure targets. Before building models, we focused on understanding the recordings and assessing their quality.

### Connection to Break Through Tech AI
This work is part of the **Break Through Tech AI Studio** program, in which student teams tackle real-world machine learning problems with an industry host company.

### Host Company: Microsoft
Microsoft builds AI-powered platforms and tools to meet evolving customer needs and is committed to expanding access to AI responsibly.

### Why It Matters
High blood pressure (hypertension) raises the risk of heart attack, heart disease, and stroke [CDC]. It affects roughly 1.3 billion people worldwide, and many are unaware they have it because it often causes few or no symptoms [1]. Standard cuff-based measurement is reliable but uncomfortable and impractical for continuous monitoring. Cuffless approaches based on signals like PPG could make frequent, convenient blood-pressure tracking possible — supporting earlier detection and treatment.

---

## 📊 Data Exploration

We used the **UCI Cuff-Less Blood Pressure Estimation dataset**, stored in four MATLAB (`.mat`) files containing PPG, ABP, and ECG signals. Our exploration so far focuses on **Part 1 (3,000 recordings)**, with plans to expand to the remaining parts.

- **Sampling rate:** 125 Hz → each 5-second window = 625 samples
- **Data type:** continuous physiological waveforms
- **Checks performed:** recording lengths, signal distributions, missing values, zero values, example waveforms
- **NeuroKit2 analysis:** filtering, pulse detection, signal-quality metrics
- **Features computed:** pulse rate, amplitude, approximate pulse width, and other waveform statistics, compared against draft blood-pressure targets

### Visualizations

**PPG, ABP, and ECG — first 5 seconds** 

(<img width="528" height="370" alt="image" src="https://github.com/user-attachments/assets/48e5c581-d107-4a6f-a123-ca1d25bad7c6" />
)

**Distribution of recording lengths**
<img width="1161" height="469" alt="image (1)" src="https://github.com/user-attachments/assets/aad7cc96-34f4-4809-b1e3-ff21b229da9f" />

**NeuroKit2 analysis of a PPG signal**
<img width="1416" height="843" alt="Screenshot 2026-09-22 at 10 50 04 PM" src="https://github.com/user-attachments/assets/7d032e17-cf3e-4155-86e8-f32b895b5a1b" />



---

## 🧠 Model Development

> 🚧 *In progress.* This section will cover:
> - Model(s) used
> - Feature selection and hyperparameter tuning
> - Training setup (train/validation/test split, evaluation metrics, baseline)

---

## 📈 Results & Key Findings

> 🚧 *In progress.* This section will cover performance metrics (e.g., MAE/RMSE for systolic and diastolic BP), model comparisons, and fairness/explainability insights.

---

## 🚀 Next Steps

- [ ] Feature engineering
- [ ] Model development and evaluation
- [ ] Select final model
- [ ] Extend analysis to dataset Parts 2–4
- [ ] Document limitations and future directions

---

## 📝 License

> 🚧 *License pending Challenge Advisor approval.* (e.g., This project is licensed under the [MIT License](LICENSE).)

---

## 📄 References

1. *A benchmark for machine-learning based non-invasive blood pressure estimation using photoplethysmogram.* Scientific Data (Nature).
2. *Estimating Blood Pressure from the Photoplethysmogram Signal and Demographic Features Using Machine Learning Techniques.*
3. *Exploring supervised machine learning models to estimate blood pressure using non-fiducial features of the photoplethysmogram (PPG) and its derivatives.*
4. *A continuous cuffless blood pressure measurement from optimal PPG characteristic features using machine learning algorithms.*
5. Kaggle – *Blood Pressure Analysis.*
6. Centers for Disease Control and Prevention (CDC) – High Blood Pressure.
7. Kachuee, M., et al. *Cuff-Less Blood Pressure Estimation* dataset. UCI Machine Learning Repository.

---

## 🙏 Acknowledgements

Thank you to Fatima Rafiqui, Wee Hyong Tok, Anshul Rehpade, the Microsoft team, Break Through Tech staff, and everyone else who supported our project.
