import React, { useEffect, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu } from 'lucide-react';
import Sidebar from './Sidebar';
import NeuralBackground from './NeuralBackground';
import ErrorBoundary from './ErrorBoundary';
import Breadcrumbs from './Breadcrumbs';
import Toaster from './Toaster';
import useForensicStore from '../store/useForensicStore';

export default function Layout() {
  const { fetchStatus } = useForensicStore();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Feeds --tilt-x/--tilt-y (consumed by .card-hover:hover for a subtle
  // mouse-parallax tilt) and --spotlight-x/--spotlight-y (consumed by
  // NeuralBackground's cursor-following glow). One shared, rAF-throttled
  // listener rather than per-component JS.
  useEffect(() => {
    let rafId = null;
    let latestEvent = null;

    const applyPointerEffects = () => {
      rafId = null;
      if (!latestEvent) return;
      const nx = latestEvent.clientX / window.innerWidth - 0.5;
      const ny = latestEvent.clientY / window.innerHeight - 0.5;
      document.documentElement.style.setProperty('--tilt-x', `${(ny * -6).toFixed(2)}deg`);
      document.documentElement.style.setProperty('--tilt-y', `${(nx * 6).toFixed(2)}deg`);
      document.documentElement.style.setProperty('--spotlight-x', `${((nx + 0.5) * 100).toFixed(2)}%`);
      document.documentElement.style.setProperty('--spotlight-y', `${((ny + 0.5) * 100).toFixed(2)}%`);
    };

    const handlePointerMove = (e) => {
      latestEvent = e;
      if (rafId == null) rafId = requestAnimationFrame(applyPointerEffects);
    };

    window.addEventListener('pointermove', handlePointerMove, { passive: true });
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      if (rafId != null) cancelAnimationFrame(rafId);
    };
  }, []);

  return (
    <div className="flex min-h-screen overflow-hidden relative bg-bg-void">
      {/* Skip-to-content */}
      <a href="#main-content" className="skip-nav">Skip to content</a>

      <NeuralBackground />

      {/* Mobile hamburger */}
      <button
        className="md:hidden fixed top-3 left-3 z-[60] p-2 rounded-lg bg-[rgba(12,15,22,0.9)] border border-border-dim text-text-1"
        onClick={() => setSidebarOpen((prev) => !prev)}
        aria-label="Toggle navigation menu"
      >
        <Menu size={18} />
      </button>

      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <main
        id="main-content"
        className="flex-1 overflow-y-auto h-screen relative z-10 md:ml-[72px]"
        style={{ padding: '24px 20px', paddingTop: 'max(24px, env(safe-area-inset-top))' }}
      >
        <div className="max-w-[1280px] mx-auto pt-10 md:pt-0">
          <Breadcrumbs />
          <ErrorBoundary>
            <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 15 }}
                transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
              >
                <Outlet />
              </motion.div>
            </AnimatePresence>
          </ErrorBoundary>
        </div>
      </main>

      <Toaster />
    </div>
  );
}
