/** Blood pressure is shown as whole mmHg; errors keep one decimal. */
export const mmHg = (v: number) => Math.round(v).toString();
export const err1 = (v: number) => v.toFixed(1);
export const pct = (v: number) => `${Math.round(v * 100)}%`;
export const secs = (v: number) => `${v.toFixed(v < 10 ? 1 : 0)} s`;

/** "lowest validation MAE_SBP + MAE_DBP, ..." is for engineers; this is the sentence for people. */
export const SELECTION_PLAIN = "Selected by lowest validation error";

/** Signed whole mmHg with a real minus sign: "−7", "+3", "0". */
export const signed = (v: number) => {
  const r = Math.round(v);
  return r > 0 ? `+${r}` : r < 0 ? `−${-r}` : "0";
};

/**
 * `recording_id` is the UCI recording id ((part-1)*3000 + record index), 0-based. UCI has no patient
 * ids, so several recordings may come from one person: call it a recording, never a subject or
 * patient. People count from 1, so "Recording 0" / "Sample 0" never appear.
 */
export const recordingNo = (recordingId: number) => recordingId + 1;
export const recordingLabel = (recordingId: number) => `Recording ${recordingNo(recordingId)}`;
