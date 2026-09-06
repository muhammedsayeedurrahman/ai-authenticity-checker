import React from 'react';

/**
 * Consistent page header with sleek pill badge and modern typography.
 */
export default function PageHeader({ icon: Icon, title, subtitle, actions }) {
  return (
    <header className="flex items-start justify-between flex-wrap gap-4 mb-6">
      <div className="space-y-1">
        <div className="flex items-center gap-3">
          {Icon && (
            <div className="w-10 h-10 rounded-2xl flex items-center justify-center bg-white border border-purple-200 shadow-sm text-purple-700 shadow-purple-500/10">
              <Icon size={20} />
            </div>
          )}
          <h1 className="font-display text-2xl sm:text-3xl font-black tracking-tight text-[#1E1238]">
            {title}
          </h1>
        </div>
        {subtitle && (
          <p className="text-xs sm:text-sm text-[#5B4E75] font-medium leading-relaxed max-w-2xl pl-0 sm:pl-1">
            {subtitle}
          </p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  );
}
