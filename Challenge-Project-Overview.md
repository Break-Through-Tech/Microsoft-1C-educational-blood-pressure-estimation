---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

### 🔍 SME Feedback from the Break Through Tech Evaluation Team

*Challenge Advisor: Please address the following feedback by editing this page. Your AI Studio Coach can help make project adjustments as needed, too. In addition to the grey section above, this section should be removed before sharing the repo with your student team.*

| Check | Status | Notes |
|-------|--------|-------|
| Python Compatibility | 🟢 | The tech stack is centered on Python, aligning with the fellows' experience and academic requirements. |
| Data Readiness | 🟡 | While the data is publicly available, preprocessing may be needed because it originates in different formats (CSV/TSV and MATLAB), which could require additional effort. |
| Resource Check | 🟢 | No specialized hardware required; free-tier tools (Google Colab) can meet the project's needs at no additional cost. |

**Student Fit Score:** 7/10  
**Technical Depth Score:** 8/10  
**Overall Recommendation:** REVISE

**Advisor Feedback Draft:**
The project leverages relevant and compelling datasets for public health applications, providing a rich educational opportunity. However, we should consider simplifying aspects of the machine learning techniques to better align with the fellows' existing expertise in basic modeling or provide supplementary resources. Ensuring students are adequately prepared for the unique characteristics of working with medical data is crucial. I recommend working closely with mentors to address these challenges.


---

# CufflessAI: An Educational Blood Pressure Estimation

**Company / Org:** Microsoft  
**Challenge Advisor:** Fatima Rafiqui, fatima.rafiqui@gmail.com  
**AI Studio Coach:** Anshul Rehpade, anshul.rehpade@breakthroughtech.org   
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About Microsoft

Microsoft is a global technology company that specializes in software, services, devices, and solutions. Our mission is to empower every person and every organization on the planet to achieve more, focusing on innovation and technology impact across various industries.

---

## 🎯 The Challenge

### Project Summary
In this project, you will use public physiological signal data, including photoplethysmography (PPG), electrocardiogram (ECG), and arterial blood pressure waveforms, and supervised machine learning regression techniques to build an educational model that estimates systolic and diastolic blood pressure from pulse-signal features. This will help our organization address the need for responsible, hands-on AI education by giving students experience with real-world sensor data, signal preprocessing, model evaluation, and the limitations of health-related machine learning prototypes.

### Success Criteria

Success should be measured by whether the team produces a complete, responsible, end-to-end ML prototype rather than by whether the model achieves clinical accuracy.

A successful outcome by December would include:   
- A reproducible Python notebook that loads the dataset, preprocesses signals, extracts features, trains models, and evaluates predictions.   
- At least two regression models compared against a naïve baseline.   
- Evaluation metrics reported for both SBP and DBP:   
- Evaluation using Mean Absolute Error, or MAE Root Mean Squared Error, or RMSE   
- A clear model comparison table.   
- Basic feature importance or interpretability analysis.   

### Stretch Goals

_Advanced modeling_
Students who progress quickly can try a simple 1D convolutional neural network that learns directly from raw PPG waveforms, instead of relying only on hand-crafted features.

_Blood pressure category classification_
In addition to predicting SBP and DBP, students can classify readings into broad categories such as low, normal, elevated, or high blood pressure. This would add a classification component.

_Personal calibration simulation_
Students can simulate a simple calibration step where a known cuff reading is used to adjust model predictions for an individual user.
Improved app experience

Students can expand the Streamlit app with waveform plots, model confidence/error bands, feature explanations, and side-by-side model comparisons.

_Camera/video-inspired prototype_
As an advanced stretch, students can explore how phone-camera or webcam-based pulse extraction works. This should remain optional because it introduces computer vision and signal quality challenges beyond the core ML project.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | [Title] | Data understanding, preprocessing, and baseline |
| October | [Title] | Model training and evaluation |
| November | [Title] | Final prototype, report, and demo app |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** Cuff-Less Blood Pressure Estimation   
**Format:** CSV, TSV, Matlab v7.3 mat file  
**Size:** 1gb to 5gb  
**Location:** https://archive.ics.uci.edu/dataset/340/cuff%2Bless%2Bblood%2Bpressure%2Bestimation

### Key Details
- [Brief description of what's in the data]
- [Any known limitations or preprocessing needed]
- [Link to data dictionary or documentation, if available]


---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification,Regression,Time Series Analysis,Large Language Models (LLMs)/ Generative AI

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- [e.g., Accuracy, Precision/Recall, RMSE, BLEU score]

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
