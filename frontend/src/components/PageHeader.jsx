import React from 'react';

/**
 * Consistent page header used across all pages.
 *
 * @param {React.ElementType} [icon] - Lucide icon component
 * @param {string} title - Page title
 * @param {string} [subtitle] - Optional subtitle text
 * @param {boolean} [gradient] - Use gradient text for title
 * @param {React.ReactNode} [actions] - Right-side slot for buttons/filters
 */
export default function PageHeader({ icon: Icon, title, subtitle, gradient = false, actions }) {
  return (
    <header className="flex items-start justify-between flex-wrap gap-5 mb-1">
      <div>
        <div className="flex items-center gap-3 mb-1.5">
          {Icon && (
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center bg-accent-dim border border-accent/20"
            >
              <Icon size={14} className="text-accent" />
            </div>
          )}
          <h1
            className={`font-display text-2xl font-bold tracking-tight ${gradient ? 'gradient-text' : 'text-text-1'}`}
          >
            {title}
          </h1>
        </div>
        {subtitle && (
          <p className="text-sm text-text-2 leading-relaxed">{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  );
}
