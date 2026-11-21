/** Blood pressure is shown as whole mmHg; errors keep one decimal. */
export const mmHg = (v: number) => Math.round(v).toString();
export const err1 = (v: number) => v.toFixed(1);
export const pct = (v: number) => `${Math.round(v * 100)}%`;
export const secs = (v: number) => `${v.toFixed(v < 10 ? 1 : 0)} s`;

/** "lowest validation MAE_SBP + MAE_DBP, ..." is for engineers; this is the sentence for people. */
export const SELECTION_PLAIN = "Selected by lowest validation error";
