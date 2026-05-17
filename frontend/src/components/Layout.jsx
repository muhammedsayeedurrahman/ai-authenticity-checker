import React, { useEffect, useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Menu } from 'lucide-react';
import Sidebar from './Sidebar';
import NeuralBackground from './NeuralBackground';
import useForensicStore from '../store/useForensicStore';

export default function Layout() {
  const { fetchStatus } = useForensicStore();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return (
    <div
      className="flex min-h-screen overflow-hidden relative"
      style={{ background: 'var(--bg-void, #080A0F)' }}
    >
      {/* Skip-to-content link */}
      <a href="#main-content" className="skip-nav">
        Skip to content
      </a>

      {/* Ambient gradient washes */}
      <div
        className="fixed inset-0 pointer-events-none z-0"
        style={{
          background: `
            radial-gradient(ellipse 60% 50% at 20% 50%, rgba(59,130,246,0.08) 0%, transparent 70%),
            radial-gradient(ellipse 50% 60% at 80% 30%, rgba(6,182,212,0.06) 0%, transparent 70%)
          `,
        }}
      />

      <NeuralBackground />

      {/* Mobile hamburger */}
      <button
        className="md:hidden fixed top-3 left-3 z-[60] p-2 rounded-lg"
        style={{
          background: 'rgba(12,15,22,0.9)',
          border: '1px solid var(--border-dim)',
          color: 'var(--text-1)',
        }}
        onClick={() => setSidebarOpen((prev) => !prev)}
        aria-label="Toggle navigation menu"
      >
        <Menu size={18} />
      </button>

      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <main
        id="main-content"
        className="flex-1 overflow-y-auto h-screen relative z-10 pt-14 md:pt-0 md:ml-[200px]"
        style={{ padding: '32px 20px' }}
      >
        <div style={{ maxWidth: '1280px', margin: '0 auto' }}>
          <Outlet />
        </div>
      </main>
    </div>
  );
}
