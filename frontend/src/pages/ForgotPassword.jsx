import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Mail, ArrowLeft, AlertCircle, CheckCircle2 } from 'lucide-react';
import useAuthStore from '../store/useAuthStore';
import AuthLayout from '../components/AuthLayout';

export default function ForgotPassword() {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const { resetPassword } = useAuthStore();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);
    const { error: resetError } = await resetPassword(email);
    setLoading(false);
    if (resetError) {
      setError(resetError.message);
    } else {
      setSuccess('Password reset email sent. Check your inbox.');
    }
  };

  return (
    <AuthLayout title="Reset password">
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

        <p className="text-sm text-text-2">
          Enter your email address and we'll send you a link to reset your password.
        </p>

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

        <button
          type="submit"
          disabled={loading}
          className="btn-primary w-full py-2.5 mt-2"
        >
          {loading ? (
            <>
              <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Sending...
            </>
          ) : (
            'Send Reset Link'
          )}
        </button>
      </form>

      <p className="text-center text-xs mt-6 text-text-2">
        <Link to="/login" className="font-semibold text-accent inline-flex items-center gap-1">
          <ArrowLeft size={12} />
          Back to sign in
        </Link>
      </p>
    </AuthLayout>
  );
}
