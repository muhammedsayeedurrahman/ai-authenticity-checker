import React, { useState, useCallback } from 'react';
import { forensicApi } from '../services/api';
import useToastStore from '../store/useToastStore';

/**
 * Complainant-details form that generates a downloadable cyber crime
 * complaint document from an analysis result already held in memory.
 * Never submits anything anywhere — only produces a file for the user
 * to review and file themselves. Shared between ComplaintModal (popup,
 * triggered from an analysis results panel) and the standalone
 * CyberComplaint page (pick any past AI-GENERATED analysis first).
 *
 * @param {object} analysis - analysis result dict (verdict, risk_percent, model_scores, ...)
 * @param {string} fileName - original media file name, for the record
 * @param {() => void} [onDone] - called after a successful generate+download
 * @param {() => void} [onCancel] - called when the user cancels; omit to hide the Cancel button
 */
const ID_PROOF_MAX_BYTES = 5 * 1024 * 1024;
const ID_PROOF_TYPES = new Set(['image/jpeg', 'image/png']);

export default function ComplaintForm({ analysis, fileName, onDone, onCancel }) {
  const [form, setForm] = useState({
    name: '', phone: '', email: '', address: '', incidentDescription: '',
  });
  const [idProof, setIdProof] = useState(null);
  const [idProofError, setIdProofError] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  const handleChange = useCallback((field) => (e) => {
    setForm((prev) => ({ ...prev, [field]: e.target.value }));
  }, []);

  const handleIdProofChange = useCallback((e) => {
    const file = e.target.files?.[0] || null;
    if (!file) {
      setIdProof(null);
      setIdProofError('');
      return;
    }
    if (!ID_PROOF_TYPES.has(file.type)) {
      setIdProof(null);
      setIdProofError('Must be a JPG or PNG image.');
      return;
    }
    if (file.size > ID_PROOF_MAX_BYTES) {
      setIdProof(null);
      setIdProofError('Must be under 5MB.');
      return;
    }
    setIdProofError('');
    setIdProof(file);
  }, []);

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!form.name.trim() || !analysis) return;
    setIsGenerating(true);
    try {
      const { blob, filename } = await forensicApi.generateComplaint(analysis, fileName, form, idProof);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      useToastStore.getState().addToast('Complaint document generated', 'success');
      onDone?.();
    } catch (err) {
      useToastStore.getState().addToast(
        err.response?.data?.detail || 'Could not generate complaint document', 'error',
      );
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <form onSubmit={handleGenerate} className="space-y-3">
      <div>
        <label className="text-xs mb-1 block text-text-2">Full Name *</label>
        <input
          type="text"
          required
          value={form.name}
          onChange={handleChange('name')}
          className="field-input text-sm"
          placeholder="Your full name"
        />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs mb-1 block text-text-2">Phone</label>
          <input
            type="tel"
            value={form.phone}
            onChange={handleChange('phone')}
            className="field-input text-sm"
            placeholder="Phone number"
          />
        </div>
        <div>
          <label className="text-xs mb-1 block text-text-2">Email</label>
          <input
            type="email"
            value={form.email}
            onChange={handleChange('email')}
            className="field-input text-sm"
            placeholder="Email address"
          />
        </div>
      </div>

      <div>
        <label className="text-xs mb-1 block text-text-2">Address</label>
        <textarea
          value={form.address}
          onChange={handleChange('address')}
          rows={2}
          className="field-input text-sm resize-none"
          placeholder="Your address"
        />
      </div>

      <div>
        <label className="text-xs mb-1 block text-text-2">Government ID Proof (Aadhaar / PAN / Passport)</label>
        <input
          type="file"
          accept="image/jpeg,image/png"
          onChange={handleIdProofChange}
          className="field-input text-sm file:mr-3 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-bg-inset file:text-text-2"
        />
        {idProofError ? (
          <p className="text-xs mt-1 text-risk-critical">{idProofError}</p>
        ) : (
          <p className="text-xs mt-1 text-text-3">
            JPG or PNG, under 5MB — the portal requires this in Complainant Details when filing.
            {idProof && <span className="text-risk-clear"> {idProof.name} attached.</span>}
          </p>
        )}
      </div>

      <div>
        <label className="text-xs mb-1 block text-text-2">Incident Description (optional)</label>
        <textarea
          value={form.incidentDescription}
          onChange={handleChange('incidentDescription')}
          rows={3}
          className="field-input text-sm resize-none"
          placeholder="Where/how did you encounter this content, and what harm or risk does it pose?"
        />
      </div>

      <div className="flex justify-end gap-2 pt-1">
        {onCancel && (
          <button type="button" onClick={onCancel} className="btn-ghost text-xs">
            Cancel
          </button>
        )}
        <button
          type="submit"
          disabled={!form.name.trim() || !analysis || isGenerating}
          className="btn-danger text-xs"
        >
          {isGenerating ? 'Generating…' : 'Generate & Download'}
        </button>
      </div>
    </form>
  );
}
