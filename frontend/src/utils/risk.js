/**
 * Shared risk-level utilities.
 *
 * Centralizes the color/label/background logic that was duplicated
 * across Dashboard, History, ScoreBar, RiskGauge, and VerdictCard.
 */

/** Thresholds (percentage 0-100). */
const HIGH_THRESHOLD = 70;
const MEDIUM_THRESHOLD = 40;

/**
 * Normalize a score that may be 0-1 or 0-100 into 0-100 percentage.
 *
 * Scores <= 1 are treated as fractional (0.0–1.0) and scaled to 0–100.
 * Scores > 1 are already in percentage form and clamped to 100.
 */
export function normalizeScore(score) {
  if (score == null) return 0;
  const n = Number(score);
  if (Number.isNaN(n)) return 0;
  if (n < 0) return 0;
  if (n <= 1) return n * 100;
  return Math.min(n, 100);
}

export function getRiskColor(pct) {
  if (pct > HIGH_THRESHOLD) return 'var(--risk-critical)';
  if (pct > MEDIUM_THRESHOLD) return 'var(--risk-caution)';
  return 'var(--risk-clear)';
}

export function getRiskColorRaw(pct) {
  if (pct > HIGH_THRESHOLD) return '#FB7185';
  if (pct > MEDIUM_THRESHOLD) return '#FACC15';
  return '#22C55E';
}

export function getRiskBg(pct) {
  if (pct > HIGH_THRESHOLD) return 'rgba(251,113,133,0.10)';
  if (pct > MEDIUM_THRESHOLD) return 'rgba(250,204,21,0.10)';
  return 'rgba(34,197,94,0.10)';
}

export function getRiskLabel(pct) {
  if (pct > HIGH_THRESHOLD) return 'Deepfake Detected';
  if (pct > MEDIUM_THRESHOLD) return 'Suspicious';
  return 'Authentic';
}

export function getRiskLevel(pct) {
  if (pct > HIGH_THRESHOLD) return 'High Risk';
  if (pct > MEDIUM_THRESHOLD) return 'Medium Risk';
  return 'Low Risk';
}

export function getRiskGlow(pct) {
  if (pct > HIGH_THRESHOLD) return 'rgba(251,113,133,0.4)';
  if (pct > MEDIUM_THRESHOLD) return 'rgba(250,204,21,0.4)';
  return 'rgba(34,197,94,0.4)';
}

/**
 * Risk tier ('critical' | 'caution' | 'clear') for a verdict.
 *
 * Primary signal is the numeric risk score when available; keyword
 * matching on the verdict text is a fallback only. Order matters: critical
 * is checked first, then caution, then fallback to clear.
 */
const CRITICAL_KEYWORDS = ['critical', 'fake', 'manipulat', 'deepfake', 'synthetic', 'forged', 'altered'];
const CAUTION_KEYWORDS = ['medium', 'suspicious', 'caution', 'uncertain', 'inconclusive', 'partial'];
const CLEAR_KEYWORDS = ['authentic', 'genuine', 'real', 'clear', 'low'];

export function classifyVerdict(verdict, riskScore) {
  if (riskScore != null && typeof riskScore === 'number') {
    if (riskScore > HIGH_THRESHOLD) return 'critical';
    if (riskScore > MEDIUM_THRESHOLD) return 'caution';
    return 'clear';
  }

  const lower = (verdict || '').toLowerCase();
  if (CRITICAL_KEYWORDS.some((kw) => lower.includes(kw))) return 'critical';
  if (CAUTION_KEYWORDS.some((kw) => lower.includes(kw))) return 'caution';
  if (CLEAR_KEYWORDS.some((kw) => lower.includes(kw))) return 'clear';

  // Default to caution when unable to classify
  return 'caution';
}
