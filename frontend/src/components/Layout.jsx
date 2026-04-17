import React, { useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import useForensicStore from '../store/useForensicStore';

export default function Layout() {
  const { fetchStatus } = useForensicStore();

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return (
    <div className="flex bg-background min-h-screen text-text-primary overflow-hidden">
      {/* Background ambient light */}
      <div className="fixed top-[-20%] left-[-10%] w-[50%] h-[50%] bg-accent-cyan opacity-[0.03] blur-[120px] rounded-full pointer-events-none" />
      <div className="fixed bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-accent-violet opacity-[0.03] blur-[120px] rounded-full pointer-events-none" />

      <Sidebar />
      <main className="flex-1 ml-64 p-8 overflow-y-auto h-screen relative z-10">
        <div className="max-w-7xl mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
