import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock, ArrowRight, AlertCircle, CheckCircle2 } from 'lucide-react';
import useAuthStore from '../store/useAuthStore';

export default function Signup() {
  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const [confirm,  setConfirm]  = useState('');
  const [error,    setError]    = useState('');
  const [success,  setSuccess]  = useState('');
  const [loading,  setLoading]  = useState(false);
  const { signUp } = useAuthStore();
  const navigate   = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    if (password !== confirm) { setError('Passwords do not match'); return; }
    setLoading(true);
    const { data, error: authError } = await signUp(email, password);
    setLoading(false);
    if (authError) setError(authError.message);
    else if (data?.session) navigate('/');
    else setSuccess('Check your email to confirm your account.');
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center p-4 relative"
      style={{ background: 'var(--bg-void)' }}
    >
      {/* Ambient radial gradient background */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background: `
            radial-gradient(ellipse 60% 60% at 50% 40%, rgba(59,130,246,0.10) 0%, transparent 70%),
            radial-gradient(ellipse 40% 50% at 30% 60%, rgba(6,182,212,0.06) 0%, transparent 60%)
          `,
        }}
      />

      <div
        className="w-full max-w-sm relative z-10"
        style={{
          background: 'rgba(12,15,22,0.92)',
          backdropFilter: 'blur(16px) saturate(1.4)',
          border: '1px solid rgba(59,130,246,0.15)',
          borderTop: '1px solid rgba(255,255,255,0.10)',
          borderRadius: '12px',
          padding: '36px',
          boxShadow: '0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(59,130,246,0.08)',
        }}
      >
        {/* Brand */}
        <div className="text-center mb-8">
          <p className="text-[11px] font-bold tracking-widest gradient-text">PROOFYX</p>
          <h1 className="font-display text-xl font-bold mt-1" style={{ color: 'var(--text-1)' }}>Create account</h1>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div
              role="alert"
              className="flex items-start gap-2.5 p-3 rounded-lg text-[12px]"
              style={{ background: 'rgba(251,113,133,0.08)', border: '1px solid rgba(251,113,133,0.20)', color: 'var(--risk-critical)' }}
            >
              <AlertCircle size={14} className="mt-0.5 flex-shrink-0" />
              {error}
            </div>
          )}
          {success && (
            <div
              role="alert"
              className="flex items-start gap-2.5 p-3 rounded-lg text-[12px]"
              style={{ background: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.22)', color: 'var(--risk-clear)' }}
            >
              <CheckCircle2 size={14} className="mt-0.5 flex-shrink-0" />
              {success}
            </div>
          )}

          {/* Email */}
          <div>
            <label htmlFor="email" className="block text-[11px] font-semibold mb-1.5 uppercase tracking-wider" style={{ color: 'var(--text-2)' }}>
              Email
            </label>
            <div className="relative">
              <Mail size={13} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: 'var(--text-3)' }} />
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="field-input pl-9"
                placeholder="you@example.com"
              />
            </div>
          </div>

          {/* Password */}
          <div>
            <label htmlFor="password" className="block text-[11px] font-semibold mb-1.5 uppercase tracking-wider" style={{ color: 'var(--text-2)' }}>
              Password
            </label>
            <div className="relative">
              <Lock size={13} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: 'var(--text-3)' }} />
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
                className="field-input pl-9"
                placeholder="Min 6 characters"
              />
            </div>
          </div>

          {/* Confirm Password */}
          <div>
            <label htmlFor="confirm" className="block text-[11px] font-semibold mb-1.5 uppercase tracking-wider" style={{ color: 'var(--text-2)' }}>
              Confirm Password
            </label>
            <div className="relative">
              <Lock size={13} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: 'var(--text-3)' }} />
              <input
                id="confirm"
                type="password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                required
                minLength={6}
                className="field-input pl-9"
                placeholder="Repeat your password"
              />
            </div>
          </div>

          <button type="submit" disabled={loading} className="btn-primary w-full py-2.5 mt-2">
            {loading ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Creating...
              </>
            ) : (
              <>
                Create Account
                <ArrowRight size={14} />
              </>
            )}
          </button>
        </form>

        <p className="text-center text-[12px] mt-6" style={{ color: 'var(--text-2)' }}>
          Already have an account?{' '}
          <Link to="/login" className="font-semibold" style={{ color: 'var(--accent)' }}>Sign in</Link>
        </p>
      </div>
    </div>
  );
}
