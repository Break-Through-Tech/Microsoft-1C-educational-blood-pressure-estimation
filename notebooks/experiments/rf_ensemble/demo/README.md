# Random Forest and AdaBoost

[Open the notebook](blood_pressure_demo.ipynb) to see the inputs, model setup, an example prediction, validation plots and a Random Forest vs AdaBoost comparison. Saved outputs are included for viewing on GitHub.

## Run it

Use Python 3.10–3.12. Open a terminal in this `demo` folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter notebook blood_pressure_demo.ipynb
```

On Windows, activate with `.venv\Scripts\activate` instead. In VS Code, open this notebook and select the environment you installed the requirements into. Restart the kernel and run all cells. Keep the notebook, helper scripts and `assets` folder together.

Dependencies need to be installed once. After that, the demo runs offline and does not retrain or download the full dataset. Change `EXAMPLE` from 1 to 8 to try another window. The model uses the PPG and ECG recordings; the ABP trace and reference labels are only for checking the answer.

## Results

PPG + ECG RF validation MAE: **15.03 mmHg systolic / 7.63 mmHg diastolic**. **77.8%** of diastolic estimates were within 10 mmHg. These are development validation results, not a new test or proof of clinical accuracy. The eight examples were selected randomly without consulting errors; overall metrics cover all 857 validation windows.

[Full methods and results](../RF_ECG_RESULTS.md) · [Training notebook](../rf_ecg.ipynb) · [Data attribution](DATA_SOURCE.md)

The bundled joblib model belongs to this experiment. Load serialized models only from sources you trust. `verification.json` records checks that the eight raw-signal predictions match the original validation predictions.

[Speaker notes](ONE_MINUTE_NOTES.md) follow the notebook sections for a one-minute presentation.
