# Microsoft 1C – Cuffless Blood Pressure Estimation Model

Estimating blood pressure from photoplethysmography (PPG) signals using machine learning — a Break Through Tech AI Studio project hosted by Microsoft.

---

## 👥 Team Members

| Name           | GitHub Handle                                          | Contribution                                                            |
|----------------|--------------------------------------------------------|-------------------------------------------------------------------------|
| Taylor Nguyen  | [@taylornguyen](https://github.com/taylornguyen)       | Data exploration, visualization, overall project coordination           |
| Claire Zhu     | [@czhu1231](https://github.com/czhu1231)               | Data collection, exploratory data analysis (EDA), dataset documentation |
| Katherine Shih | [@katherineShih113](https://github.com/katherineShih113) | Data preprocessing, feature engineering, data validation              |
| Charis Xiong   | [@karrixxa](https://github.com/karrixxa)               | Data preprocessing, exploration, visualization, model evaluation        |
| Chris Park     | [@chrispark](https://github.com/chrispark)             | Model evaluation, performance analysis, results interpretation          |

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
![Recording length distribution](images/recording_length_distribution.png)

**NeuroKit2 analysis of a PPG signal**
![NeuroKit2 PPG analysis](images/neurokit2_ppg.png)

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
