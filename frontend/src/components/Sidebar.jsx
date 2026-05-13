import React from 'react';
import { NavLink } from 'react-router-dom';
import { Image, Film, Mic, Layers, Activity, Clock, Settings, LogOut } from 'lucide-react';
import useForensicStore from '../store/useForensicStore';
import useAuthStore from '../store/useAuthStore';
import { isAuthEnabled } from '../services/supabase';
import logo from '../assets/logo.jpeg';

export default function Sidebar() {
  const { systemStatus } = useForensicStore();
  const { user, signOut } = useAuthStore();
  const loadedCount = systemStatus?.loaded_models?.length || 0;

  const links = [
    { to: "/", icon: <Activity size={20} />, label: "Dashboard", exact: true },
    { to: "/image", icon: <Image size={20} />, label: "Image Analysis" },
    { to: "/video", icon: <Film size={20} />, label: "Video Analysis" },
    { to: "/audio", icon: <Mic size={20} />, label: "Audio Analysis" },
    { to: "/multimodal", icon: <Layers size={20} />, label: "Multimodal Fusion" },
    { to: "/history", icon: <Clock size={20} />, label: "History" },
  ];

  return (
    <aside className="w-64 bg-background-card border-r border-border-subtle flex flex-col h-screen fixed left-0 top-0 z-50">
      <div className="p-6 flex items-center gap-3">
        <img
          src={logo}
          alt="ProofyX Logo"
          className="w-12 h-12 rounded-xl shadow-glow-accent border-[2px] border-accent/30 bg-black/50"
        />
        <div>
          <h1 className="text-xl font-black bg-clip-text text-transparent bg-gradient-to-r from-accent to-indigo-400 tracking-wider">
            PROOFYX
          </h1>
          <p className="text-[10px] text-text-muted uppercase tracking-widest font-bold">Forensic Command</p>
        </div>
      </div>

      <nav className="flex-1 px-4 py-4 space-y-2 relative">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.exact}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-all duration-300 ${
                isActive
                  ? 'bg-accent/10 text-accent border border-accent/20 shadow-[inset_0_0_10px_rgba(99,102,241,0.1)]'
                  : 'text-text-secondary hover:text-text-primary hover:bg-background-card-hover border border-transparent'
              }`
            }
          >
            {link.icon}
            {link.label}
          </NavLink>
        ))}
      </nav>

      <div className="p-4 mt-auto space-y-2">
        {isAuthEnabled() && user && (
          <div className="flex items-center justify-between p-3 rounded-xl border border-border-subtle bg-background">
            <div className="truncate mr-2">
              <p className="text-xs font-bold text-text-primary truncate">{user.email}</p>
              <p className="text-[10px] text-text-muted">Authenticated</p>
            </div>
            <button
              onClick={signOut}
              className="p-1.5 rounded-lg hover:bg-red-500/10 text-text-secondary hover:text-red-400 transition-colors"
              title="Sign out"
            >
              <LogOut size={14} />
            </button>
          </div>
        )}
        <NavLink
          to="/settings"
          className={({isActive}) => `flex items-center gap-3 p-3 rounded-xl border transition-all ${
            isActive ? 'border-accent bg-accent/5' : 'border-border-subtle bg-background hover:border-border-hover'
          }`}
        >
          <Settings size={16} className="text-text-secondary" />
          <span className="text-xs font-bold text-text-primary">Settings</span>
        </NavLink>
        <div className="flex items-center gap-3 p-3 rounded-xl border border-border-subtle bg-background">
          <div className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-success opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-accent-success shadow-[0_0_8px_rgba(34,197,94,0.8)]"></span>
          </div>
          <div>
            <p className="text-xs font-bold text-text-primary">System Online</p>
            <p className="text-[10px] text-text-muted">{loadedCount} Models Active</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
