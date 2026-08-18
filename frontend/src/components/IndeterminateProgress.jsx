import React from 'react';

/**
 * Honest "still working" indicator for long-running analyses.
 *
 * Deliberately indeterminate — the backend has no progress signal for
 * analysis requests, so this never claims a specific percentage, just
 * that something is actively happening.
 */
export default function IndeterminateProgress({ label = 'Processing…' }) {
  return (
    <div className="w-full" role="status" aria-live="polite">
      <div className="progress-indeterminate-track" />
      {label && <p className="text-xs mt-2 text-center text-text-3">{label}</p>}
    </div>
  );
}
