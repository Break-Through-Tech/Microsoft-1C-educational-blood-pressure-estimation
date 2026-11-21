/** Team roster (from the repository README) and the models each person built. Edit here. */
export interface Member {
  name: string;
  role: string;
  models?: { name: string; detail: string; inApp: boolean }[];
}

export const TEAM: Member[] = [
  { name: "Taylor Nguyen", role: "Data exploration, visualization, project coordination" },
  {
    name: "Claire Zhu",
    role: "Data collection, exploratory analysis, dataset documentation",
    models: [{ name: "1D ResNet (v3)", detail: "Deep network on the raw PPG waveform", inApp: true }],
  },
  { name: "Katherine Shih", role: "Data preprocessing, feature engineering, data validation" },
  { name: "Sri Lakshmi Anbarasan", role: "Data preprocessing, feature engineering, data validation" },
  {
    name: "Charis Xiong",
    role: "Data preprocessing, exploration, visualization, model evaluation",
    models: [
      { name: "Random Forest", detail: "68 PPG and ECG features", inApp: false },
      { name: "AdaBoost", detail: "68 PPG and ECG features", inApp: false },
    ],
  },
  { name: "Chris Park", role: "Model evaluation, performance analysis, results interpretation" },
  { name: "Emily Ta", role: "Model evaluation, performance analysis, results interpretation" },
  {
    name: "Sophia Lu",
    role: "Data visualization, model evaluation, results interpretation",
    models: [{ name: "Support vector regression", detail: "44 PPG features, RBF kernel", inApp: false }],
  },
];
