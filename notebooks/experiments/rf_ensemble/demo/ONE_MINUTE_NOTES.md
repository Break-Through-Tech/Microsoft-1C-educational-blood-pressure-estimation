# Speaker notes

## Model
“I compared Random Forest and AdaBoost for predicting blood pressure from PPG and ECG. Random Forest averages predictions from many trees, while AdaBoost builds trees in sequence, focusing on examples that earlier trees got wrong. Here’s what a prediction looks like.”

## Example prediction
“The model takes five seconds of signals and predicts systolic and diastolic pressure. This table compares its prediction with the measured values. To see how it performs beyond one example, we can look at the full validation set.”

## Random Forest results
“Each dot is one prediction. Closer to the red line means less error. We can see that the model struggles more at low and high pressures.”

## Random Forest vs. AdaBoost
“Comparing both models on the same data, Random Forest had lower error than AdaBoost for both pressures—about 15 for systolic and 7.6 for diastolic, in millimeters of mercury. So Random Forest performed better here, though there’s still room to improve.”
