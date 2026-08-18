import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock, ArrowRight, AlertCircle, CheckCircle2, Eye, EyeOff } from 'lucide-react';
import useAuthStore from '../store/useAuthStore';
import AuthLayout from '../components/AuthLayout';

export default function Signup() {
  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const [confirm,  setConfirm]  = useState('');
  const [showPw,   setShowPw]   = useState(false);
  const [showConf, setShowConf] = useState(false);
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
    <AuthLayout title="Create account">
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div
            role="alert"
            className="flex items-start gap-2.5 p-3 rounded-lg text-xs bg-risk-criticalDim text-risk-critical border border-[rgba(251,113,133,0.20)]"
          >
            <AlertCircle size={14} className="mt-0.5 flex-shrink-0" />
            {error}
          </div>
        )}
        {success && (
          <div
            role="alert"
            className="flex items-start gap-2.5 p-3 rounded-lg text-xs bg-risk-clearDim text-risk-clear border border-[rgba(34,197,94,0.22)]"
          >
            <CheckCircle2 size={14} className="mt-0.5 flex-shrink-0" />
            {success}
          </div>
        )}

        {/* Email */}
        <div>
          <label htmlFor="email" className="block text-xs font-semibold mb-1.5 uppercase tracking-wider text-text-2">
            Email
          </label>
          <div className="relative">
            <Mail size={13} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none text-text-3" />
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
          <label htmlFor="password" className="block text-xs font-semibold mb-1.5 uppercase tracking-wider text-text-2">
            Password
          </label>
          <div className="relative">
            <Lock size={13} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none text-text-3" />
            <input
              id="password"
              type={showPw ? 'text' : 'password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
              className="field-input pl-9 pr-10"
              placeholder="Min 6 characters"
            />
            <button
              type="button"
              onClick={() => setShowPw((v) => !v)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-text-3 hover:text-text-1 transition-colors"
              aria-label={showPw ? 'Hide password' : 'Show password'}
            >
              {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          </div>
        </div>

        {/* Confirm Password */}
        <div>
          <label htmlFor="confirm" className="block text-xs font-semibold mb-1.5 uppercase tracking-wider text-text-2">
            Confirm Password
          </label>
          <div className="relative">
            <Lock size={13} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none text-text-3" />
            <input
              id="confirm"
              type={showConf ? 'text' : 'password'}
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              required
              minLength={6}
              className="field-input pl-9 pr-10"
              placeholder="Repeat your password"
            />
            <button
              type="button"
              onClick={() => setShowConf((v) => !v)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-text-3 hover:text-text-1 transition-colors"
              aria-label={showConf ? 'Hide password' : 'Show password'}
            >
              {showConf ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
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

      <p className="text-center text-xs mt-6 text-text-2">
        Already have an account?{' '}
        <Link to="/login" className="font-semibold text-accent">Sign in</Link>
      </p>
    </AuthLayout>
  );
}
