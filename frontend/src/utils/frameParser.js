/**
 * Parser for the `frame_details` raw monospace text blob returned by video
 * analysis (columns: Frame, Time, Risk, Pred, Face, ViT, Freq, Forns,
 * FaceM, DINO, Eff — space-separated, first 2 lines are a header).
 *
 * Shared by FrameTable (tabular view) and VideoRiskTimeline (chart view) so
 * there's one parser for this format, not two.
 *
 * Returns an array of string arrays, index-aligned with the column list
 * above (row[0] = frame number, row[1] = time, row[2] = risk, ...).
 */
export function parseFrameDetails(framesRawStr) {
  if (!framesRawStr) return [];

  const lines = framesRawStr.split('\n');
  if (lines.length < 3) return [];

  const rows = [];
  for (let i = 2; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const cols = line.split(/\s+/);
    if (cols.length >= 8) {
      rows.push(cols);
    }
  }
  return rows;
}
