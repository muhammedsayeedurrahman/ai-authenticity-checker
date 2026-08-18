import React from 'react';

/**
 * Shared layout for Login, Signup, and ForgotPassword pages.
 * Glass card with ambient gradient background and brand header.
 */
export default function AuthLayout({ title, children }) {
  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative bg-bg-void">
      {/* Ambient radial gradient background */}
      <div
        className="fixed inset-0 pointer-events-none"
        aria-hidden="true"
        style={{
          background: `
            radial-gradient(ellipse 60% 60% at 50% 40%, rgba(59,130,246,0.10) 0%, transparent 70%),
            radial-gradient(ellipse 40% 50% at 70% 60%, rgba(56,189,248,0.06) 0%, transparent 60%)
          `,
        }}
      />

      <div className="w-full max-w-sm relative z-10 glass-card p-9 rounded-xl border-t border-t-white/10 shadow-modal">
        {/* Brand */}
        <div className="text-center mb-8">
          <p className="text-[11px] font-bold tracking-widest gradient-text">PROOFYX</p>
          <h1 className="font-display text-xl font-bold mt-1 text-text-1">{title}</h1>
        </div>

        {children}
      </div>
    </div>
  );
}
