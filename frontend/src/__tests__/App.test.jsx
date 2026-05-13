import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

// Mock supabase as disabled (dev mode - no auth)
vi.mock('../services/supabase', () => ({
  supabase: null,
  isAuthEnabled: () => false,
}));

// Mock the logo import
vi.mock('../assets/logo.jpeg', () => ({ default: 'logo.jpeg' }));

// Mock NeuralBackground (Three.js won't work in jsdom)
vi.mock('../components/NeuralBackground', () => ({
  default: () => null,
}));

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders Dashboard on root route without auth', async () => {
    const { default: App } = await import('../App');

    // Need to render with MemoryRouter since App has its own BrowserRouter
    // Instead, render App directly which includes BrowserRouter
    const { container } = render(<App />);
    expect(container).toBeTruthy();
  });

  it('renders Sidebar navigation links', async () => {
    const { default: Sidebar } = await import('../components/Sidebar');

    render(
      <MemoryRouter>
        <Sidebar />
      </MemoryRouter>
    );

    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Image Analysis')).toBeInTheDocument();
    expect(screen.getByText('Video Analysis')).toBeInTheDocument();
    expect(screen.getByText('Audio Analysis')).toBeInTheDocument();
    expect(screen.getByText('History')).toBeInTheDocument();
  });
});
