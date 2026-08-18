import React, { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Image, Film, Mic, Layers, LayoutDashboard, Clock, Activity, LogOut } from 'lucide-react';
import useForensicStore from '../store/useForensicStore';
import useAuthStore from '../store/useAuthStore';
import { isAuthEnabled } from '../services/supabase';
import logo from '../assets/logo.jpeg';

const NAV_LINKS = [
  { to: '/',           icon: LayoutDashboard, label: 'Dashboard', exact: true },
  { to: '/image',      icon: Image,           label: 'Image' },
  { to: '/video',      icon: Film,            label: 'Video' },
  { to: '/audio',      icon: Mic,             label: 'Audio' },
  { to: '/multimodal', icon: Layers,          label: 'Multimodal' },
  { to: '/history',    icon: Clock,           label: 'History' },
];

// Desktop rests as an icon-only rail and expands to full width on
// hover/focus — mirrors the reference hover-expand pattern, built with
// what's already in this project (framer-motion) rather than the
// Next.js/shadcn component it was demonstrated in.
const COLLAPSED_WIDTH = 72;
const EXPANDED_WIDTH = 228;

function SidebarLabel({ showLabel, children }) {
  return (
    <motion.span
      animate={{ opacity: showLabel ? 1 : 0, width: showLabel ? 'auto' : 0 }}
      transition={{ duration: 0.2 }}
      className="whitespace-nowrap overflow-hidden"
    >
      {children}
    </motion.span>
  );
}

function SidebarLink({ to, exact, icon: Icon, label, onClick, showLabel }) {
  return (
    <NavLink
      to={to}
      end={exact}
      title={label}
      className={({ isActive }) => `nav-item w-full group ${isActive ? 'active' : ''}`}
      onClick={onClick}
    >
      {({ isActive }) => (
        <>
          {isActive && (
            <motion.span
              layoutId="sidebar-active"
              className="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full bg-accent shadow-glow-blue"
              transition={{ type: 'spring', stiffness: 350, damping: 30 }}
            />
          )}
          <Icon
            size={17}
            className="flex-shrink-0 transition-transform duration-200 group-hover:scale-110 group-hover:-translate-y-0.5"
          />
          <SidebarLabel showLabel={showLabel}>{label}</SidebarLabel>
        </>
      )}
    </NavLink>
  );
}

export default function Sidebar({ open = true, onClose = () => {} }) {
  const { systemStatus } = useForensicStore();
  const { user, signOut } = useAuthStore();
  const loadedCount = systemStatus?.loaded_models?.length || 0;

  // Desktop always shows the sidebar regardless of `open` — track the
  // breakpoint in JS so the spring transform below doesn't fight the
  // md-and-up "always visible" behavior via CSS specificity.
  // matchMedia is guarded — jsdom (tests) doesn't implement it.
  const hasMatchMedia = typeof window !== 'undefined' && typeof window.matchMedia === 'function';
  const [isDesktop, setIsDesktop] = useState(
    () => hasMatchMedia && window.matchMedia('(min-width: 768px)').matches,
  );

  useEffect(() => {
    if (!hasMatchMedia) return;
    const mql = window.matchMedia('(min-width: 768px)');
    const handler = (e) => setIsDesktop(e.matches);
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, [hasMatchMedia]);

  // Hover/focus-driven expand — desktop only. Mobile has no hover surface
  // and always shows the full drawer while open.
  const [isExpanded, setIsExpanded] = useState(false);
  const handleBlurCapture = (e) => {
    if (!e.currentTarget.contains(e.relatedTarget)) setIsExpanded(false);
  };

  const visible = isDesktop || open;
  const showLabel = !isDesktop || isExpanded;
  const width = !isDesktop ? EXPANDED_WIDTH : (isExpanded ? EXPANDED_WIDTH : COLLAPSED_WIDTH);

  return (
    <>
      {/* Mobile backdrop */}
      <AnimatePresence>
        {open && !isDesktop && (
          <motion.div
            className="md:hidden fixed inset-0 z-[45] bg-black/50"
            onClick={onClose}
            aria-hidden="true"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          />
        )}
      </AnimatePresence>

      <motion.aside
        className="flex flex-col h-screen fixed left-0 top-0 z-50 bg-[rgba(12,15,22,0.95)] backdrop-blur-xl border-r border-border-dim overflow-hidden"
        animate={{ x: visible ? 0 : '-100%', width }}
        transition={{ type: 'spring', stiffness: 300, damping: 32 }}
        onMouseEnter={() => isDesktop && setIsExpanded(true)}
        onMouseLeave={() => isDesktop && setIsExpanded(false)}
        onFocusCapture={() => isDesktop && setIsExpanded(true)}
        onBlurCapture={handleBlurCapture}
      >
        {/* Brand */}
        <div className="px-4 pt-5 pb-4 flex items-center gap-2.5">
          <img
            src={logo}
            alt="ProofyX"
            className="w-7 h-7 rounded-lg flex-shrink-0 object-cover transition-transform duration-150 hover:scale-105"
          />
          <SidebarLabel showLabel={showLabel}>
            <span className="text-sm font-bold tracking-[0.08em] uppercase gradient-text font-display">
              PROOFYX
            </span>
          </SidebarLabel>
        </div>

        <div className="mx-4 mb-3 divider" />

        {/* Main nav */}
        <nav className="flex-1 px-3 space-y-0.5 overflow-y-auto no-scrollbar" aria-label="Main navigation">
          {NAV_LINKS.map((link) => (
            <SidebarLink key={link.to} {...link} onClick={onClose} showLabel={showLabel} />
          ))}
        </nav>

        {/* Bottom section */}
        <div className="px-3 pb-4 mt-2 space-y-1">
          <div className="mx-1 mb-2 divider" />

          {/* System Status */}
          <SidebarLink to="/system" icon={Activity} label="System Status" onClick={onClose} showLabel={showLabel} />

          {/* Auth user */}
          {isAuthEnabled() && user && (
            <div className="flex items-center justify-between px-3 py-2 mt-1">
              <SidebarLabel showLabel={showLabel}>
                <p className="text-[13px] font-medium truncate min-w-0 text-text-2">
                  {user.email}
                </p>
              </SidebarLabel>
              <button
                onClick={signOut}
                title="Sign out"
                aria-label="Sign out"
                className="p-1 rounded transition-colors ml-2 flex-shrink-0 hover:bg-white/5 text-text-3"
              >
                <LogOut size={13} />
              </button>
            </div>
          )}

          {/* System status indicator */}
          <div className="flex items-center gap-2 px-3 py-2">
            <span className="relative flex-shrink-0">
              <span
                className={`block w-1.5 h-1.5 rounded-full ${loadedCount > 0 ? 'bg-risk-clear animate-pulse' : 'bg-risk-critical'}`}
              />
            </span>
            <SidebarLabel showLabel={showLabel}>
              <span className="text-[13px] text-text-3">
                {loadedCount} model{loadedCount !== 1 ? 's' : ''} loaded
              </span>
            </SidebarLabel>
          </div>
        </div>
      </motion.aside>
    </>
  );
}
