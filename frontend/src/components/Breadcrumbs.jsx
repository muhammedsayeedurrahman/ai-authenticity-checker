import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ChevronRight, Home } from 'lucide-react';

const ROUTE_LABELS = {
  image: 'Image Analysis',
  video: 'Video Forensics',
  audio: 'Audio Analysis',
  multimodal: 'Multimodal Fusion',
  history: 'History',
  system: 'System Status',
};

export default function Breadcrumbs() {
  const { pathname } = useLocation();

  // Don't show on dashboard
  if (pathname === '/' || pathname === '/dashboard') return null;

  const segment = pathname.replace(/^\//, '').split('/')[0];
  const label = ROUTE_LABELS[segment];

  if (!label) return null;

  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 mb-4 text-xs">
      <Link
        to="/dashboard"
        className="flex items-center gap-1 text-text-3 hover:text-text-1 transition-colors"
      >
        <Home size={12} />
        <span>Dashboard</span>
      </Link>
      <ChevronRight size={10} className="text-text-3 opacity-50" />
      <span className="text-text-2 font-medium">{label}</span>
    </nav>
  );
}
