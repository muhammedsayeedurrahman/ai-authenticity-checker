import React from 'react';
import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Image, Film, Mic, Layers, LayoutDashboard, Clock, Settings, LogOut } from 'lucide-react';
import useForensicStore from '../store/useForensicStore';
import useAuthStore from '../store/useAuthStore';
import { isAuthEnabled } from '../services/supabase';
import logo from '../assets/logo.jpeg';

export default function Sidebar({ open = true, onClose = () => {} }) {
  const { systemStatus } = useForensicStore();
  const { user, signOut } = useAuthStore();
  const loadedCount = systemStatus?.loaded_models?.length || 0;

  const links = [
    { to: '/',           icon: <LayoutDashboard size={15} />, label: 'Dashboard',       exact: true },
    { to: '/image',      icon: <Image size={15} />,           label: 'Image Analysis'              },
    { to: '/video',      icon: <Film size={15} />,            label: 'Video Analysis'              },
    { to: '/audio',      icon: <Mic size={15} />,             label: 'Audio Analysis'              },
    { to: '/multimodal', icon: <Layers size={15} />,          label: 'Multimodal'                  },
    { to: '/history',    icon: <Clock size={15} />,           label: 'History'                     },
  ];

  const handleNavClick = () => {
    onClose();
  };

  return (
    <>
      {/* Mobile backdrop overlay */}
      {open && (
        <div
          className="md:hidden fixed inset-0 z-[45] bg-black/50"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`
          w-[200px] flex flex-col h-screen fixed left-0 top-0 z-50
          transition-transform duration-200 ease-out
          md:translate-x-0
          ${open ? 'translate-x-0' : '-translate-x-full'}
        `}
        style={{
          background: 'rgba(12,15,22,0.92)',
          backdropFilter: 'blur(16px) saturate(1.4)',
          borderRight: '1px solid rgba(59,130,246,0.10)',
          boxShadow: '4px 0 24px rgba(0,0,0,0.3)',
        }}
      >
        {/* Brand */}
        <div className="px-4 pt-5 pb-4 flex items-center gap-2.5">
          <img
            src={logo}
            alt="ProofyX"
            className="w-6 h-6 rounded-lg flex-shrink-0 object-cover"
          />
          <span
            className="text-[11px] font-bold tracking-[0.15em] uppercase gradient-text"
            style={{ fontFamily: "'Space Grotesk', sans-serif" }}
          >
            PROOFYX
          </span>
        </div>

        {/* Separator — gradient line */}
        <div
          className="mx-4 mb-3"
          style={{
            height: '1px',
            background: 'linear-gradient(90deg, rgba(59,130,246,0.3), rgba(6,182,212,0.2), transparent)',
          }}
        />

        {/* Navigation */}
        <nav className="flex-1 px-3 space-y-0.5 overflow-y-auto no-scrollbar" aria-label="Main navigation">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.exact}
              className={({ isActive }) =>
                `nav-item w-full ${isActive ? 'active' : ''}`
              }
              style={{ fontSize: '13px', fontWeight: 500 }}
              onClick={handleNavClick}
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.span
                      layoutId="sidebar-active"
                      className="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full"
                      style={{
                        background: 'var(--accent)',
                        boxShadow: '0 0 8px rgba(59,130,246,0.4)',
                      }}
                      transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                    />
                  )}
                  <span className="flex-shrink-0">{link.icon}</span>
                  <span>{link.label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Bottom section */}
        <div className="px-3 pb-4 mt-2 space-y-1">
          {/* Gradient separator */}
          <div
            className="mx-1 mb-2"
            style={{
              height: '1px',
              background: 'linear-gradient(90deg, rgba(59,130,246,0.3), rgba(6,182,212,0.2), transparent)',
            }}
          />

          {/* Settings */}
          <NavLink
            to="/settings"
            className={({ isActive }) => `nav-item w-full ${isActive ? 'active' : ''}`}
            style={{ fontSize: '13px', fontWeight: 500 }}
            onClick={handleNavClick}
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.span
                    layoutId="sidebar-active"
                    className="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full"
                    style={{
                      background: 'var(--accent)',
                      boxShadow: '0 0 8px rgba(59,130,246,0.4)',
                    }}
                    transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                  />
                )}
                <Settings size={15} />
                <span>Settings</span>
              </>
            )}
          </NavLink>

          {/* Auth user */}
          {isAuthEnabled() && user && (
            <div className="flex items-center justify-between px-3 py-2 mt-1">
              <p
                className="text-[11px] font-medium truncate min-w-0"
                style={{ color: 'var(--text-2, #8B95A5)' }}
              >
                {user.email}
              </p>
              <button
                onClick={signOut}
                title="Sign out"
                aria-label="Sign out"
                className="p-1 rounded transition-colors ml-2 flex-shrink-0"
                style={{ color: 'var(--text-3, #4A5264)' }}
              >
                <LogOut size={13} />
              </button>
            </div>
          )}

          {/* System status with glowing dot */}
          <div className="flex items-center gap-2 px-3 py-2">
            <span className="relative flex-shrink-0">
              <span
                className="block w-1.5 h-1.5 rounded-full"
                style={{ background: 'var(--risk-clear, #34D399)' }}
              />
              <span
                className="absolute inset-0 w-1.5 h-1.5 rounded-full animate-ping"
                style={{ background: 'var(--risk-clear, #34D399)', opacity: 0.4 }}
              />
            </span>
            <span
              className="text-[11px]"
              style={{ color: 'var(--text-3, #4A5264)' }}
            >
              {loadedCount} models &middot; Online
            </span>
          </div>
        </div>
      </aside>
    </>
  );
}
