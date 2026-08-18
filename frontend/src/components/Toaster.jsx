import React from 'react';
import { AnimatePresence } from 'framer-motion';
import { CheckCircle, AlertTriangle, Info, X } from 'lucide-react';
import useToastStore from '../store/useToastStore';

const ICONS = {
  success: CheckCircle,
  error: AlertTriangle,
  info: Info,
};

const STYLES = {
  success: 'bg-risk-clearDim text-risk-clear border-[rgba(34,197,94,0.20)]',
  error: 'bg-risk-criticalDim text-risk-critical border-[rgba(251,113,133,0.20)]',
  info: 'bg-accent-dim text-accent border-[rgba(59,130,246,0.20)]',
};

export default function Toaster() {
  const { toasts, removeToast } = useToastStore();

  return (
    <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 max-w-sm w-full pointer-events-none">
      <AnimatePresence>
        {toasts.map((toast) => {
          const Icon = ICONS[toast.type] || Info;
          const style = STYLES[toast.type] || STYLES.info;

          return (
            <div
              key={toast.id}
              className={`pointer-events-auto flex items-start gap-2.5 px-4 py-3 rounded-lg border text-sm shadow-lg ${style}`}
            >
              <Icon size={16} className="flex-shrink-0 mt-0.5" />
              <span className="flex-1 leading-snug">{toast.message}</span>
              <button
                onClick={() => removeToast(toast.id)}
                className="flex-shrink-0 opacity-60 hover:opacity-100 transition-opacity"
                aria-label="Dismiss"
              >
                <X size={14} />
              </button>
            </div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
