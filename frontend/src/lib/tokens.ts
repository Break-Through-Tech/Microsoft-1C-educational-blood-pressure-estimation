import { useTheme } from "@/lib/theme";

/**
 * The one chart mapping, used by every chart and legend:
 *   truth (true value / measured signal) = neutral dark
 *   primary model                         = accent (LED green)
 *   other models                          = muted tones, in order
 *   baseline                              = gray, dashed
 * Literals mirror index.css: the theme class is applied in an effect, after the render that would
 * read computed styles.
 */
const LIGHT = {
  ink: "#1c1f1e",
  muted: "#5c6460",
  rule: "#d8dbd7",
  surface: "#ffffff",
  truth: "#1c1f1e",
  primary: "#0b7a53",
  others: ["#7f9a8e", "#a39a7e", "#7d8a9c"],
  baseline: "#8e9591",
};
const DARK: typeof LIGHT = {
  ink: "#e9ece9",
  muted: "#a0a8a4",
  rule: "#2f3532",
  surface: "#1b1f1d",
  truth: "#e9ece9",
  primary: "#4fc291",
  others: ["#8fae9f", "#b8ad8e", "#8e9cb0"],
  baseline: "#7f8783",
};

/** dash pattern for the baseline, everywhere */
export const BASELINE_DASH = "5 4";

export type ChartColors = typeof LIGHT;

export function useChartColors(): ChartColors {
  return useTheme().dark ? DARK : LIGHT;
}

/** Color for a model by role. `otherIndex` counts only non-primary, non-baseline models. */
export function seriesColor(c: ChartColors, role: { is_primary: boolean; is_baseline: boolean }, otherIndex = 0) {
  if (role.is_primary) return c.primary;
  if (role.is_baseline) return c.baseline;
  return c.others[otherIndex % c.others.length];
}
/** chart text uses the page font; tabular figures come from index.css */
export const MONO = "Public Sans Variable, Arial, sans-serif";
