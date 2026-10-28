# Random Forest / ensemble resources — Charis

Reviewed October 14, 2026, alongside the earlier reading notes in `LITERATURE_REVIEW.md`. This guide connects the team's links to the actual all-Part EDA and RF implementation. It is not a reproduction of published scores.

## Start with the ensemble guide

[Scikit-learn: ensembles](https://scikit-learn.org/stable/modules/ensemble.html) is the main implementation reference. We need **regression**, because SBP and DBP are numbers, not class labels.

Random Forest fits trees on bootstrap samples and considers a random subset of features at each split. Its regression prediction averages the trees. Extra Trees adds randomness to the candidate split thresholds. Boosting builds trees sequentially to correct earlier errors. Our first comparison uses RF and Extra Trees; boosting is a later option.

The guide also cautions that training impurity importance can be misleading. We save validation permutation importance as an additional descriptive check. Neither importance measure establishes causation, and correlated features can share importance.

**Our choices:** 200 trees, `max_features=0.8`, and a small search over maximum depth (10/unlimited) and minimum leaf size (5/10). Three training folds keep entire detected overlap groups together. OOB scores and random-window cross-validation are not substitutes for this grouping. Imputation and target scaling are fitted inside each training fold.

The online docs currently describe a newer scikit-learn release than our installed 1.6.1. The APIs used here were checked by executing them in this environment.

## Background papers

| Resource | What it contributes | How we use it / limitation |
|---|---|---|
| [González et al., BP benchmark](https://www.nature.com/articles/s41597-023-02020-6) and [official code](https://github.com/inventec-ai-center/bp-benchmark) | Shared preprocessing and comparisons of feature-to-label and waveform models, with evaluation designed to address leakage | Keep five-second windows, beat-derived targets, overlap checks, grouped evaluation and a training-mean baseline. UCI record IDs do not recover patient identity. Our pipeline does not reproduce the whole benchmark. |
| [Chowdhury et al., PPG and demographics](https://pmc.ncbi.nlm.nih.gov/articles/PMC7309072/) | Combines signal features and demographic information; compares feature selection and regression, including GPR | Supports a feature-model baseline. Our MAT files have no age/sex/body-size metadata, so we do not add demographic columns or expect the paper's scores. |
| [Argüello-Prada and Castaño Mosquera, non-fiducial features](https://pmc.ncbi.nlm.nih.gov/articles/PMC12511243/) | Statistical features of PPG and its derivatives reduce reliance on manually identifiable pulse landmarks | Use raw/clean PPG statistics, first/second derivatives and simple frequency descriptors. Their 30-second overlapping windows differ from our five-second nonoverlapping windows; this is an adaptation. |
| [Nishan et al., optimal PPG features](https://pmc.ncbi.nlm.nih.gov/articles/PMC10963242/) | Compares SVR, nearest-neighbor, decision-tree and Random Forest regression using PPG characteristics, feature selection and demographics | Supports comparing model families instead of assuming RF wins. We keep feature selection/tuning inside training folds and cannot reuse unavailable demographics. |

The four papers were previously reviewed in `LITERATURE_REVIEW.md`. On this pass, the Nature page, official benchmark code and Nishan full-text page were retrievable. The Chowdhury and non-fiducial PMC pages showed browser checks; their NCBI BioC requests were rate-limited. Their notes above use the existing methods review and are not a claim of a fresh full-text read. We do not use reported paper errors as accuracy targets for our different dataset/splits.

## Technical tutorials and examples

| Resource | What to use it for | Project decision |
|---|---|---|
| [SciPy `loadmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) | Understand MATLAB file loading | SciPy documents support through MAT 7.2 and requires an HDF5 library for 7.3. Our files are HDF5-backed; use `h5py`, as in the existing notebook. |
| [NeuroKit examples](https://neuropsychology.github.io/NeuroKit/examples/) and [PPG function reference](https://neuropsychology.github.io/NeuroKit/functions/ppg.html) | Cleaning, peaks, rates, quality and waveform review | Use the locally installed 0.2.10. Mean `PPG_Quality` is a template-similarity measure, not a probability of a correct waveform. Pair quality with beat counts, detector agreement and visual review. The function reference was accessible; the examples index failed to fetch. |
| [pyPPG](https://pypi.org/project/pyPPG/) | A possible later route to carefully defined pulse landmarks and biomarkers | Reviewed the package description; not installed or executed. Current RF uses simpler statistical/derivative features. Review its methods, compatibility and license before incorporating it. |
| [Kaggle BloodPressure analysis](https://www.kaggle.com/code/stephenmugisha/bloodpressure-analysis) | Possible notebook structure/plot ideas | The browser retrieved the title but not notebook content. We did not copy, execute or verify its preprocessing, split or results. |
| [ICASSP 2020 video](https://www.youtube.com/watch?v=tZLotOFiyZ4) | Team background discussion | The video could not be retrieved in this session. No transcript or performance claim was used; this remains a team viewing resource. |

## What the implementation actually uses

`expanded_data_analysis.py` samples 200 recording IDs from each Part and up to 12 windows per recording. It checks full-record PPG processing, same-window PPG/ABP counts and rates, detector sensitivity, mean quality, minimum beats, flat extrema, and record-filter boundaries. The initial `PPG_Quality >= 0.8` rule is provisional; thresholds 0.6/0.7/0.8/0.9 are compared for retention, not optimized on test errors.

`random_forest_pipeline.py` uses a fixed whitelist of 37 PPG-only features: raw and cleaned waveform statistics, derivative statistics, pulse timing, approximate peak-to-peak shape summaries and frequency descriptors. Quality, ABP measurements, IDs, overlap groups and count-comparison flags are not predictor columns. The pulse summaries are descriptors, not validated physiological landmarks or HRV.

The comparison is training mean, Ridge, grouped-CV-tuned Random Forest, a predefined Extra Trees model, and an RBF support vector regressor (SVR) selected on grouped training folds. In the current validation run, SVR is close to RF but does not improve its MAE (17.03/9.03 mmHg versus RF 16.86/8.83). Extra Trees is slightly lower on DBP at 8.69 mmHg. These nearby scores do not establish a winning model. Validation reporting includes SBP/DBP MAE/RMSE/bias, record- and overlap-group-average errors, per-Part errors, paired group-bootstrap intervals and permutation importance. A wider grouped training-CV search over 27 RF configurations selected a shallower tree, but did not meaningfully improve validation: SBP/DBP MAE moved from 16.86/8.83 to 16.91/8.88 mmHg. Paired group bootstrap intervals for the change included zero, so the original RF remains the simpler reference. Histogram gradient boosting was considered, but its local grouped search was stopped due to runtime; there is no result to report. The test split remains reserved.

This expanded all-Part experiment has different data and gates from the earlier small ResNet pilot. Its errors cannot establish that RF outperforms that ResNet; a fair comparison needs both models evaluated on the same reviewed rows, labels and splits.
