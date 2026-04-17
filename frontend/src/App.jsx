import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import ImageAnalysis from './pages/ImageAnalysis';
import VideoAnalysis from './pages/VideoAnalysis';
import AudioAnalysis from './pages/AudioAnalysis';
import Multimodal from './pages/Multimodal';
import StatusPage from './pages/StatusPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="image" element={<ImageAnalysis />} />
          <Route path="video" element={<VideoAnalysis />} />
          <Route path="audio" element={<AudioAnalysis />} />
          <Route path="multimodal" element={<Multimodal />} />
          <Route path="status" element={<StatusPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
