import React from 'react';
import { NavLink } from 'react-router-dom';
import { Image, Film, Mic, Layers, Activity } from 'lucide-react';
import useForensicStore from '../store/useForensicStore';
import logo from '../assets/logo.jpeg';

export default function Sidebar() {
  const { systemStatus } = useForensicStore();
  const loadedCount = systemStatus?.loaded_models?.length || 0;

  const links = [
    { to: "/", icon: <Activity size={20} />, label: "Dashboard", exact: true },
    { to: "/image", icon: <Image size={20} />, label: "Image Analysis" },
    { to: "/video", icon: <Film size={20} />, label: "Video Analysis" },
    { to: "/audio", icon: <Mic size={20} />, label: "Audio Analysis" },
    { to: "/multimodal", icon: <Layers size={20} />, label: "Multimodal Fusion" },
  ];

  return (
    <aside className="w-64 bg-background-card border-r border-border-subtle flex flex-col h-screen fixed left-0 top-0 z-50">
      <div className="p-6 flex items-center gap-3">
        <img 
          src={logo} 
          alt="ProofyX Logo" 
          className="w-12 h-12 rounded-xl shadow-glow-cyan border-[2px] border-accent-cyan/30 bg-black/50" 
        />
        <div>
          <h1 className="text-xl font-black bg-clip-text text-transparent bg-gradient-to-r from-accent-cyan to-accent-violet tracking-wider">
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
                  ? 'bg-gradient-to-r from-[rgba(0,240,255,0.1)] to-[rgba(168,85,247,0.1)] text-accent-cyan border border-border-glow shadow-[inset_0_0_10px_rgba(0,240,255,0.1)]'
                  : 'text-text-secondary hover:text-text-primary hover:bg-background-card-hover border border-transparent'
              }`
            }
          >
            {link.icon}
            {link.label}
          </NavLink>
        ))}
      </nav>

      <div className="p-4 mt-auto">
        <NavLink 
          to="/status"
          className={({isActive}) => `flex items-center gap-3 p-3 rounded-xl border transition-all ${
            isActive ? 'border-accent-cyan bg-[rgba(0,240,255,0.05)]' : 'border-border-subtle bg-background hover:border-border-glow'
          }`}
        >
          <div className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-green opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-accent-green shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
          </div>
          <div>
            <p className="text-xs font-bold text-text-primary">System Online</p>
            <p className="text-[10px] text-text-muted">{loadedCount} Models Active</p>
          </div>
        </NavLink>
      </div>
    </aside>
  );
}
