import React, { useEffect, useState } from 'react';
import { Outlet, useLocation, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu } from 'lucide-react';
import ErrorBoundary from './ErrorBoundary';
import Toaster from './Toaster';
import Sidebar from './Sidebar';
import useForensicStore from '../store/useForensicStore';
import logo from '../assets/logo.jpeg';

export default function Layout() {
  const { fetchStatus, systemStatus } = useForensicStore();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  const loadedCount = systemStatus?.loaded_models?.length || 7;

  return (
    <div className="min-h-screen flex flex-col relative overflow-x-hidden">
      <Sidebar open={mobileMenuOpen} onClose={() => setMobileMenuOpen(false)} />

      {/* Mobile-only top bar: brand + menu toggle (sidebar is off-canvas below md) */}
      <header className="md:hidden sticky top-0 z-40 flex items-center justify-between px-4 py-3 bg-white/90 backdrop-blur-xl border-b border-purple-200/60">
        <Link to="/" className="flex items-center gap-2" title="ProofyX Home">
          <img src={logo} alt="ProofyX" className="w-7 h-7 rounded-lg object-cover ring-2 ring-purple-300/60" />
          <span className="font-display font-black text-base tracking-tight text-[#1E1238]">
            PROOFY<span className="text-purple-600">X</span>
          </span>
        </Link>
        <button
          onClick={() => setMobileMenuOpen((prev) => !prev)}
          className="p-2 rounded-full hover:bg-purple-100 text-purple-900 transition-colors"
          aria-label="Toggle navigation"
        >
          <Menu size={20} />
        </button>
      </header>

      {/* Main Page Area — left padding clears the desktop sidebar rail */}
      <main className="flex-1 w-full md:pl-[72px] px-4 sm:px-8 py-6 relative z-10">
        <div className="max-w-7xl mx-auto">
          <ErrorBoundary>
            <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
              >
                <Outlet />
              </motion.div>
            </AnimatePresence>
          </ErrorBoundary>
        </div>
      </main>

      {/* Subtle Futuristic Footer */}
      <footer className="md:pl-[72px] py-6 px-4 text-center text-xs text-[#8F81A8] border-t border-purple-200/40">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>ProofyX — Next-Gen AI Authenticity & Forensics Platform</span>
          <span className="font-mono text-[11px] text-purple-700 bg-purple-100/60 px-2.5 py-0.5 rounded-full border border-purple-200">
            {loadedCount} Forensic Models Active
          </span>
        </div>
      </footer>

      <Toaster />
    </div>
  );
}
