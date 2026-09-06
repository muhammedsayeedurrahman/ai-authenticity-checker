import React, { useEffect, useState } from 'react';
import { Outlet, useLocation, NavLink, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Image as ImageIcon,
  Film,
  Mic,
  FileSearch,
  Layers,
  LayoutDashboard,
  Clock,
  Activity,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Menu,
  X,
  LogOut,
  User,
  Home,
} from 'lucide-react';
import ErrorBoundary from './ErrorBoundary';
import Toaster from './Toaster';
import useForensicStore from '../store/useForensicStore';
import useAuthStore from '../store/useAuthStore';
import { isAuthEnabled } from '../services/supabase';
import logo from '../assets/logo.jpeg';

const NAV_LINKS = [
  { to: '/dashboard',  label: 'Dashboard',      icon: LayoutDashboard, exact: true },
  { to: '/image',      label: 'Image AI',       icon: ImageIcon },
  { to: '/audio',      label: 'Audio Voice',    icon: Mic },
  { to: '/video',      label: 'Video Deepfake', icon: Film },
  { to: '/document',   label: 'Document',       icon: FileSearch },
  { to: '/multimodal', label: 'Multimodal',     icon: Layers },
  { to: '/history',    label: 'History',        icon: Clock },
];

export default function Layout() {
  const { fetchStatus, systemStatus } = useForensicStore();
  const { user, signOut } = useAuthStore();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const loadedCount = systemStatus?.loaded_models?.length || 7;

  return (
    <div className="min-h-screen flex flex-col relative overflow-x-hidden">
      {/* Top Floating Pill Navigation Header */}
      <header className="sticky top-4 z-50 px-4 sm:px-8 max-w-7xl mx-auto w-full">
        <nav className="pill-nav px-4 sm:px-6 py-3 flex items-center justify-between gap-4">
          {/* Brand Logo - Links to Dashboard */}
          <Link to="/dashboard" className="flex items-center gap-2.5 flex-shrink-0 group" title="Forensics Dashboard">
            <div className="w-8 h-8 rounded-full overflow-hidden border-2 border-purple-300 shadow-md group-hover:scale-105 transition-transform flex items-center justify-center bg-purple-100">
              <img src={logo} alt="ProofyX" className="w-full h-full object-cover" />
            </div>
            <span className="font-display font-black text-lg tracking-tight text-[#1E1238] flex items-center gap-1">
              PROOFY<span className="text-purple-600">X</span>
            </span>
          </Link>

          {/* Desktop Nav Links */}
          <div className="hidden lg:flex items-center gap-1.5">
            {NAV_LINKS.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.exact}
                className={({ isActive }) =>
                  `px-3.5 py-1.5 rounded-full text-xs font-bold transition-all ${
                    isActive
                      ? 'bg-[#4C1D95] text-white shadow-md shadow-purple-900/20'
                      : 'text-[#5B4E75] hover:text-[#1E1238] hover:bg-purple-100/50'
                  }`
                }
              >
                {link.label}
              </NavLink>
            ))}
          </div>

          {/* Right Action Buttons */}
          <div className="hidden sm:flex items-center gap-2.5">
            <Link
              to="/complaint"
              className="px-3.5 py-1.5 rounded-full text-xs font-semibold text-rose-700 bg-rose-50 border border-rose-200 hover:bg-rose-100 transition-colors flex items-center gap-1"
            >
              <ShieldAlert size={12} />
              Cyber Cell
            </Link>

            {/* Quick button to view Landing Page */}
            <Link
              to="/"
              className="btn-primary py-1.5 px-4 text-xs font-bold shadow-sm flex items-center gap-1.5"
              title="Return to Landing Page"
            >
              <Home size={13} />
              Landing Page
            </Link>
          </div>

          {/* Mobile Menu Toggle */}
          <button
            onClick={() => setMobileMenuOpen((prev) => !prev)}
            className="lg:hidden p-2 rounded-full hover:bg-purple-100 text-purple-900 transition-colors"
            aria-label="Toggle navigation"
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </nav>

        {/* Mobile Dropdown Menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="lg:hidden mt-2 p-4 rounded-3xl bg-white/95 backdrop-blur-xl border border-purple-200/60 shadow-xl space-y-1.5"
            >
              {NAV_LINKS.map((link) => {
                const Icon = link.icon;
                return (
                  <NavLink
                    key={link.to}
                    to={link.to}
                    end={link.exact}
                    onClick={() => setMobileMenuOpen(false)}
                    className={({ isActive }) =>
                      `flex items-center gap-3 px-4 py-2.5 rounded-2xl text-xs font-bold transition-all ${
                        isActive
                          ? 'bg-[#4C1D95] text-white shadow-md'
                          : 'text-[#5B4E75] hover:bg-purple-50'
                      }`
                    }
                  >
                    <Icon size={16} />
                    {link.label}
                  </NavLink>
                );
              })}
              <div className="pt-2 border-t border-purple-100 flex gap-2">
                <Link
                  to="/complaint"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex-1 text-center py-2 rounded-xl text-xs font-bold text-rose-700 bg-rose-50"
                >
                  Cyber Report
                </Link>
                <Link
                  to="/"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex-1 text-center py-2 rounded-xl text-xs font-bold text-white bg-purple-700 flex items-center justify-center gap-1"
                >
                  <Home size={13} />
                  Landing
                </Link>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {/* Main Page Area */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-8 py-6 relative z-10">
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
      </main>

      {/* Subtle Futuristic Footer */}
      <footer className="py-6 px-4 text-center text-xs text-[#8F81A8] border-t border-purple-200/40">
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
