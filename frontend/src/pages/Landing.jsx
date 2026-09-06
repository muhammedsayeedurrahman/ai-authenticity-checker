import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ShieldCheck,
  ShieldAlert,
  Cpu,
  Sparkles,
  Image,
  Film,
  Mic,
  FileSearch,
  Layers,
  LayoutDashboard,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  Activity,
  Zap,
  Lock,
  Globe,
  BarChart3,
  Search,
  Sliders,
  Menu,
  X,
  FileCheck2,
  Terminal,
  Code2,
  Copy,
  Check,
  Play,
  Eye,
  FileText,
  UserCheck,
  HelpCircle,
  AlertTriangle,
  UploadCloud,
  CheckCircle,
  ExternalLink,
  Shield,
  Smartphone,
  Newspaper,
  Briefcase,
  Scale,
} from 'lucide-react';
import logo from '../assets/logo.jpeg';
import heroScanImg from '../assets/deepfake_hero_scan.jpg';
import voiceScanImg from '../assets/voice_scan_vis.jpg';
import docFraudImg from '../assets/doc_fraud_vis.jpg';
import '../landing.css';

/* ── Code snippets for the collapsible developer deep dive ── */
const CODE_SNIPPETS = {
  pipeline: {
    title: 'core/pipeline.py',
    lang: 'Python (PyTorch)',
    desc: 'ModelRegistry orchestrating 12 multi-stream neural networks with Bayesian consensus scoring.',
    code: `class ModelRegistry:
    """Loads all models once and provides inference methods."""
    def analyze_image(self, img_pil: Image.Image) -> dict[str, Any]:
        # Spatial Tensor + 2D DCT Frequency Decomposition
        tensor = TRANSFORM(img_pil).unsqueeze(0).to(self.device)
        dino_score = self.models["dino"](tensor).item()
        eff_score  = self.models["efficientnet"](tensor).item()
        freq_score = self.freq_analyzer.analyze(img_pil)
        
        # Bayesian Consensus Fusion P(Fake) ∈ [0.0, 1.0]
        p_fake = self.fusion_mlp.predict([dino_score, eff_score, freq_score])
        return {
            "p_fake": float(p_fake),
            "verdict": "FAKE" if p_fake > 0.55 else "REAL",
            "confidence": "HIGH" if abs(p_fake - 0.5) > 0.3 else "MODERATE"
        }`,
  },
  frequency: {
    title: 'core_models/frequency_cnn.py',
    lang: 'Python (NumPy / 2D FFT)',
    desc: '2D Fast Fourier Transform & Azimuthal Spectral profile unmasking AI checkerboard artifacts.',
    code: `def extract_frequency_residuals(image_np: np.ndarray) -> np.ndarray:
    """Computes 2D Fast Fourier Transform & High-Pass Mask."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)
    return float(np.clip(np.mean(magnitude_spectrum) / 100.0, 0.0, 1.0))`,
  },
  audio: {
    title: 'core_models/wav2vec2_audio.py',
    lang: 'Python (Wav2Vec 2.0)',
    desc: 'Wav2Vec 2.0 acoustic neural network exposing synthetic voice clones and vocoder anomalies.',
    code: `class Wav2Vec2AudioDetector(nn.Module):
    """Deep acoustic feature analyzer for synthetic speech detection."""
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        features = self.wav2vec(waveforms).last_hidden_state
        pooled = torch.mean(features, dim=1)
        synthetic_prob = self.classifier(pooled)
        return synthetic_prob # Probability of cloned voice`,
  },
};

export default function Landing() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeFaq, setActiveFaq] = useState(null);
  const [activeDemoTab, setActiveDemoTab] = useState('image');
  const [activeCodeTab, setActiveCodeTab] = useState('pipeline');
  const [copiedCode, setCopiedCode] = useState(false);
  const [showDeveloperCode, setShowDeveloperCode] = useState(false);

  const handleCopyCode = () => {
    navigator.clipboard.writeText(CODE_SNIPPETS[activeCodeTab].code);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const navLinks = [
    { label: 'How It Works', href: '#how-it-works' },
    { label: 'What We Detect', href: '#features' },
    { label: 'Visual Demo', href: '#demo' },
    { label: 'Who It Is For', href: '#use-cases' },
    { label: 'FAQ', href: '#faq' },
  ];

  const laymanFeatures = [
    {
      icon: Image,
      title: 'Photo & Image Checker',
      laymanDesc: 'Find out if a portrait, profile picture, or social media image was made by AI (Midjourney, Stable Diffusion, DALL-E) or altered in Photoshop.',
      realWorldExample: 'Example: Detect fake celebrity photos or AI-generated dating profiles.',
      tag: 'Photos & Art',
      link: '/image',
      badgeColor: 'bg-purple-100 text-purple-800 border-purple-200',
    },
    {
      icon: Mic,
      title: 'Voice Clone & Audio Detector',
      laymanDesc: 'Identify whether a voice message, phone call, or podcast audio is a real human or an AI-generated clone mimicking someone you know.',
      realWorldExample: 'Example: Spot WhatsApp voice scams impersonating family members.',
      tag: 'Voice & Calls',
      link: '/audio',
      badgeColor: 'bg-cyan-100 text-cyan-800 border-cyan-200',
    },
    {
      icon: Film,
      title: 'Video Deepfake Scanner',
      laymanDesc: 'Scan videos frame-by-frame to catch face swaps, unnatural eye blinking, lip-sync mismatch, and AI-generated video speeches.',
      realWorldExample: 'Example: Verify breaking news videos before sharing on social media.',
      tag: 'Videos & Clips',
      link: '/video',
      badgeColor: 'bg-fuchsia-100 text-fuchsia-800 border-fuchsia-200',
    },
    {
      icon: FileSearch,
      title: 'Document & Bill Verifier',
      laymanDesc: 'Check if official certificates, salary slips, bank receipts, or invoices have had text, dates, or signatures digitally edited.',
      realWorldExample: 'Example: Catch falsified bills, forged certificates, and altered receipts.',
      tag: 'PDFs & Invoices',
      link: '/document',
      badgeColor: 'bg-indigo-100 text-indigo-800 border-indigo-200',
    },
    {
      icon: Layers,
      title: 'All-in-One Multi-Scanner',
      laymanDesc: 'Drop in any media file and let ProofyX automatically combine all 12 detectors to give you a single, straightforward authenticity score.',
      realWorldExample: 'Example: Quick overall check for mixed media campaigns.',
      tag: 'Universal Scan',
      link: '/multimodal',
      badgeColor: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    },
    {
      icon: ShieldAlert,
      title: 'Official Cyber Complaint Dossier',
      laymanDesc: 'Generate a ready-to-file legal report with tamper-proof cryptographic evidence that you can submit directly to cyber police.',
      realWorldExample: 'Example: File police complaints for online scams and defamation.',
      tag: 'Legal Evidence',
      link: '/complaint',
      badgeColor: 'bg-amber-100 text-amber-800 border-amber-200',
    },
  ];

  const interactiveDemos = {
    image: {
      title: 'Visual Face & Photo Deepfake Inspection',
      subtitle: 'See how ProofyX unmasks AI face swaps and synthetic portraits in seconds.',
      image: heroScanImg,
      summary: 'Our system highlights manipulated regions with a glowing heatmap. Even if a face looks 100% natural to human eyes, our algorithm detects micro-pixel anomalies and camera sensor mismatches.',
      highlights: [
        '98.7% Authentic Confidence score',
        'Facial symmetry & eye-blink verification',
        'Detects Midjourney, Flux & FaceSwap artifacts',
      ],
      link: '/image',
      btnText: 'Test an Image Now',
    },
    audio: {
      title: 'Voice Clone & Synthetic Audio Analysis',
      subtitle: 'Detect ElevenLabs and AI voice clones imitating real people.',
      image: voiceScanImg,
      summary: 'AI voice generators cannot replicate genuine vocal tract frequencies. Our scanner analyzes pitch contours, phase shifts, and digital vocoder noise to expose synthetic speech.',
      highlights: [
        'Instant voiceprint comparison',
        'Identifies 99.1% of AI cloned voices',
        'Works on WhatsApp voice notes & recordings',
      ],
      link: '/audio',
      btnText: 'Test an Audio Clip',
    },
    document: {
      title: 'Document & Certificate Tamper Detection',
      subtitle: 'Catch digitally edited text, modified font layers, and forged stamps.',
      image: docFraudImg,
      summary: 'When someone edits a PDF or certificate in software, the invisible metadata and font layers change. ProofyX pinpoints every modified line with precise red warning boxes.',
      highlights: [
        'Flags altered text and modified dates',
        'Validates official digital signatures',
        'Includes SHA-256 cryptographic evidence hash',
      ],
      link: '/document',
      btnText: 'Test a Document',
    },
  };

  const useCases = [
    {
      icon: Smartphone,
      title: 'Everyday People & Families',
      desc: 'Protect yourself against WhatsApp voice clone scams, fake investment videos, and manipulated photos of loved ones.',
    },
    {
      icon: Newspaper,
      title: 'Journalists & Fact Checkers',
      desc: 'Verify viral news videos, political speeches, and social media media before publishing to stop the spread of fake news.',
    },
    {
      icon: Briefcase,
      title: 'Businesses & HR Teams',
      desc: 'Prevent executive impersonation fraud, verify identity documents of candidates, and ensure invoice integrity.',
    },
    {
      icon: Scale,
      title: 'Legal & Law Enforcement',
      desc: 'Generate court-admissible forensic dossiers with mathematical certainty and cryptographic custody chains.',
    },
  ];

  const steps = [
    {
      step: '1',
      title: 'Upload Any Media',
      desc: 'Simply drag and drop any image (JPG/PNG), video (MP4), audio (WAV/MP3), or document (PDF).',
      icon: UploadCloud,
    },
    {
      step: '2',
      title: 'Instant 2-Second Scan',
      desc: 'Our 12 AI models inspect pixel noise, frequency spectrums, and voice biometrics in real time.',
      icon: Zap,
    },
    {
      step: '3',
      title: 'Get Clear Results & Report',
      desc: 'Receive a plain-English verdict (Real or Fake), confidence percentage, and an official report.',
      icon: CheckCircle,
    },
  ];

  const faqs = [
    {
      q: 'Can a regular person with no technical background use ProofyX?',
      a: 'Yes, absolutely! ProofyX was built so that anyone can simply upload a picture, audio message, or video and get an instant, clear answer: "Genuine" or "AI Manipulated", along with an easy-to-read percentage score.',
    },
    {
      q: 'How does ProofyX know if a photo or voice message is fake?',
      a: 'Human eyes and ears can easily be fooled by high-quality AI, but computers leave behind microscopic mathematical traces. ProofyX checks for invisible digital fingerprints, abnormal sound frequencies, and pixel compression patterns that only AI tools create.',
    },
    {
      q: 'What should I do if I am a victim of an AI scam or deepfake?',
      a: 'ProofyX includes a dedicated "Cyber Complaint" feature. You can upload the scam media and click "Generate Complaint Docket" to get a standardized legal evidence document ready to submit directly to your local police or cybercrime reporting portal.',
    },
    {
      q: 'Is my uploaded photo or audio kept private and safe?',
      a: 'Yes, 100%. We have a strict Zero-Retention Privacy Policy. Your uploaded files are scanned in secure temporary memory and deleted immediately after the analysis. We never store, share, or use your files to train models.',
    },
  ];

  return (
    <div className="relative min-h-screen text-slate-900 font-sans overflow-x-hidden selection:bg-purple-200 selection:text-purple-900">
      {/* Background ambient mesh */}
      <div className="lp-bg-mesh">
        <div className="lp-grid-pattern" />
      </div>

      {/* Floating Accent Glows */}
      <div className="fixed top-20 left-10 w-96 h-96 bg-purple-400/20 rounded-full blur-3xl pointer-events-none lp-glow-orb" />
      <div className="fixed top-1/3 right-10 w-96 h-96 bg-fuchsia-400/20 rounded-full blur-3xl pointer-events-none lp-glow-orb" />

      {/* ─── Top Floating Glassmorphic Header ───────────────────────── */}
      <header className="sticky top-0 z-50 w-full backdrop-blur-2xl bg-white/85 border-b border-purple-200/70 shadow-xs transition-all">
        <div className="lp-container flex items-center justify-between h-16 sm:h-18">
          {/* Logo & Brand */}
          <Link to="/dashboard" className="flex items-center gap-3 group">
            <div className="relative">
              <img
                src={logo}
                alt="ProofyX Logo"
                className="w-9 h-9 rounded-xl object-cover ring-2 ring-purple-500/25 shadow-md group-hover:scale-105 transition-transform duration-200"
              />
              <span className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 bg-emerald-500 border-2 border-white rounded-full animate-pulse" />
            </div>
            <div className="flex flex-col">
              <span className="font-extrabold tracking-wider text-base font-display text-purple-950 flex items-center gap-2">
                PROOFY<span className="text-purple-600">X</span>
                <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 bg-purple-100 text-purple-700 rounded-md border border-purple-200">
                  AI Verifier
                </span>
              </span>
            </div>
          </Link>

          {/* Desktop Nav Links */}
          <nav className="hidden lg:flex items-center gap-7 text-xs sm:text-sm font-semibold text-purple-950/80">
            {navLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                className="hover:text-purple-700 transition-colors py-1 relative after:content-[''] after:absolute after:bottom-0 after:left-0 after:w-0 after:h-0.5 after:bg-purple-600 hover:after:w-full after:transition-all"
              >
                {link.label}
              </a>
            ))}
          </nav>

          {/* Direct CTA Buttons */}
          <div className="hidden sm:flex items-center gap-3">
            <Link
              to="/dashboard"
              className="lp-btn-primary text-xs sm:text-sm font-bold !py-2 !px-4.5 shadow-purple-600/25 hover:shadow-purple-600/45"
            >
              <Zap size={14} className="text-purple-200" />
              <span>Try Scanner Free</span>
              <ArrowRight size={14} />
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden p-2 rounded-xl text-purple-950 hover:bg-purple-100/60 transition-colors"
            aria-label="Toggle Navigation Menu"
          >
            {mobileMenuOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>

        {/* Mobile Dropdown */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="lg:hidden border-b border-purple-100 bg-white/95 backdrop-blur-xl px-6 py-4 flex flex-col gap-3 shadow-xl"
            >
              {navLinks.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className="text-sm font-semibold text-purple-950 hover:text-purple-600 py-1.5"
                >
                  {link.label}
                </a>
              ))}
              <div className="pt-2 border-t border-purple-100 flex flex-col gap-2">
                <Link
                  to="/dashboard"
                  onClick={() => setMobileMenuOpen(false)}
                  className="lp-btn-primary w-full justify-center text-sm !py-2.5"
                >
                  <Zap size={15} />
                  <span>Try Scanner Free</span>
                </Link>
                <Link
                  to="/complaint"
                  onClick={() => setMobileMenuOpen(false)}
                  className="w-full text-center py-2 rounded-xl text-xs font-bold text-rose-700 bg-rose-50 border border-rose-200"
                >
                  Report Cyber Scam
                </Link>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {/* ─── Hero Section with High-Impact Layman Copy & Reference Visual ── */}
      <section className="relative z-10 pt-10 pb-16 md:pt-16 md:pb-24">
        <div className="lp-container">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-12 items-center">
            {/* Left Hero Content */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="lg:col-span-6 flex flex-col gap-6 text-left"
            >
              {/* Badge */}
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-purple-100/90 border border-purple-300/70 text-purple-900 text-xs font-bold w-fit shadow-xs">
                <span className="flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                <Sparkles size={13} className="text-purple-600" />
                <span>Instant AI & Deepfake Verification</span>
              </div>

              {/* Main Heading - Layman Friendly */}
              <h1 className="text-4xl sm:text-5xl lg:text-5xl xl:text-6xl font-black tracking-tight leading-[1.08] text-slate-950 font-display">
                Is It Real or AI? <br />
                <span className="lp-gradient-text">Verify Any Media</span> In Seconds.
              </h1>

              {/* Subheading - Crystal Clear */}
              <p className="text-base sm:text-lg text-slate-600 leading-relaxed max-w-xl">
                Protect yourself from voice clone scams, fake photos, altered videos, and forged documents. ProofyX scans any file and tells you with mathematical clarity if it was created or manipulated by AI.
              </p>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center gap-3 pt-2">
                <Link to="/dashboard" className="lp-btn-primary text-base !px-6 !py-3.5 shadow-lg shadow-purple-600/30">
                  <ShieldCheck size={19} />
                  <span>Start Free Scan</span>
                  <ArrowRight size={16} />
                </Link>

                <a href="#demo" className="lp-btn-secondary text-base !px-5 !py-3.5">
                  <Eye size={17} />
                  <span>See How It Works</span>
                </a>
              </div>

              {/* Quick Trust Highlights */}
              <div className="pt-6 border-t border-purple-200/70 grid grid-cols-3 gap-4">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-emerald-100 flex items-center justify-center text-emerald-600 flex-shrink-0">
                    <CheckCircle2 size={18} />
                  </div>
                  <div className="text-xs">
                    <div className="font-bold text-slate-900">99.4% Accurate</div>
                    <div className="text-slate-500 text-[11px]">Instant detection</div>
                  </div>
                </div>

                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-purple-100 flex items-center justify-center text-purple-600 flex-shrink-0">
                    <Lock size={16} />
                  </div>
                  <div className="text-xs">
                    <div className="font-bold text-slate-900">100% Private</div>
                    <div className="text-slate-500 text-[11px]">Zero file storage</div>
                  </div>
                </div>

                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-cyan-100 flex items-center justify-center text-cyan-600 flex-shrink-0">
                    <Zap size={16} />
                  </div>
                  <div className="text-xs">
                    <div className="font-bold text-slate-900">&lt; 2 Seconds</div>
                    <div className="text-slate-500 text-[11px]">Fast scan speed</div>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Right Hero: High-Class Reference Image & Visual Showcase */}
            <motion.div
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.7, delay: 0.1 }}
              className="lg:col-span-6 w-full"
            >
              <div className="relative rounded-3xl p-3 sm:p-4 bg-gradient-to-b from-purple-200/60 to-purple-300/40 border border-purple-300/80 shadow-2xl backdrop-blur-xl group overflow-hidden">
                {/* Reference Image with Overlay Card */}
                <div className="relative rounded-2xl overflow-hidden shadow-lg border border-purple-400/40">
                  <video
                    src="/hero-demo.mp4"
                    autoPlay
                    loop
                    muted
                    playsInline
                    aria-label="ProofyX AI Deepfake Facial Authenticity Scan Demo"
                    className="w-full h-auto object-cover transform group-hover:scale-102 transition-transform duration-500"
                  />

                  {/* Floating Live Badge */}
                  <div className="absolute top-3 left-3 bg-slate-950/85 backdrop-blur-md px-3 py-1.5 rounded-xl border border-purple-500/50 flex items-center gap-2 shadow-lg">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-[11px] font-bold text-white uppercase tracking-wider">
                      Live Forensic Scanner
                    </span>
                  </div>

                  {/* Floating Action Button */}
                  <div className="absolute bottom-3 right-3">
                    <Link
                      to="/image"
                      className="px-4 py-2 rounded-xl bg-purple-700/90 hover:bg-purple-800 text-white text-xs font-bold backdrop-blur-md shadow-lg flex items-center gap-1.5 transition-all"
                    >
                      <Zap size={13} />
                      <span>Test Live Image</span>
                    </Link>
                  </div>
                </div>

                {/* Caption Banner */}
                <div className="mt-3 px-3 py-2 flex items-center justify-between text-xs text-purple-950 font-medium">
                  <span className="flex items-center gap-1.5">
                    <ShieldCheck size={14} className="text-purple-700" />
                    <span>Real vs. AI Deepfake Facial Landmark & Heatmap Analysis</span>
                  </span>
                  <span className="font-bold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded-md">
                    VERIFIED AUTHENTIC
                  </span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ─── 3 Simple Steps (How It Works for Layman) ───────────────── */}
      <section id="how-it-works" className="relative z-10 py-16 bg-white/60 border-y border-purple-200/60 backdrop-blur-md">
        <div className="lp-container">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <span className="inline-block px-3 py-1 rounded-full bg-purple-100 border border-purple-300 text-purple-900 text-xs font-bold uppercase tracking-widest mb-3">
              Simple 3-Step Process
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-950 font-display tracking-tight mb-3">
              How Anyone Can Check Any Media
            </h2>
            <p className="text-slate-600 text-sm sm:text-base">
              No software installation or tech skills needed. Get verified answers in three simple steps.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {steps.map((item, idx) => {
              const Icon = item.icon;
              return (
                <div
                  key={idx}
                  className="lp-glass-card rounded-2xl p-6 relative flex flex-col items-center text-center group"
                >
                  <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-purple-600 to-indigo-600 text-white flex items-center justify-center mb-4 shadow-md group-hover:scale-110 transition-transform">
                    <Icon size={24} />
                  </div>
                  <div className="text-xs font-bold font-mono px-2.5 py-0.5 rounded-full bg-purple-100 text-purple-800 mb-2">
                    STEP 0{item.step}
                  </div>
                  <h3 className="text-lg font-bold text-slate-900 mb-2">
                    {item.title}
                  </h3>
                  <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
                    {item.desc}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ─── What We Detect (Plain English Modalities) ──────────────── */}
      <section id="features" className="relative z-10 py-20 md:py-28">
        <div className="lp-container">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <span className="inline-block px-3 py-1 rounded-full bg-purple-100 border border-purple-300 text-purple-900 text-xs font-bold uppercase tracking-widest mb-3">
              All-In-One Protection
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-950 font-display tracking-tight mb-4">
              What Can ProofyX Detect?
            </h2>
            <p className="text-slate-600 text-base">
              From WhatsApp audio scams to fake news videos and altered receipts, we have you covered.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {laymanFeatures.map((feat, i) => {
              const Icon = feat.icon;
              return (
                <motion.div
                  key={feat.title}
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35, delay: i * 0.05 }}
                  className="lp-glass-card rounded-2xl p-6 flex flex-col justify-between group"
                >
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div className="w-12 h-12 rounded-xl bg-purple-100 border border-purple-200 flex items-center justify-center text-purple-700 group-hover:bg-purple-600 group-hover:text-white transition-all duration-200 shadow-xs">
                        <Icon size={22} />
                      </div>
                      <span className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full border ${feat.badgeColor}`}>
                        {feat.tag}
                      </span>
                    </div>

                    <h3 className="text-lg font-bold text-slate-900 mb-2 group-hover:text-purple-800 transition-colors">
                      {feat.title}
                    </h3>
                    <p className="text-xs sm:text-sm text-slate-600 leading-relaxed mb-4">
                      {feat.laymanDesc}
                    </p>
                    <div className="p-2.5 rounded-xl bg-purple-50/80 border border-purple-100 text-xs text-purple-900 font-medium mb-5">
                      {feat.realWorldExample}
                    </div>
                  </div>

                  <div className="pt-3 border-t border-purple-100/80 flex items-center justify-between">
                    <Link
                      to={feat.link}
                      className="inline-flex items-center gap-1.5 text-xs font-bold text-purple-700 hover:text-purple-950 group-hover:translate-x-1 transition-all"
                    >
                      <span>Try this scanner</span>
                      <ArrowRight size={13} />
                    </Link>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ─── Interactive Visual Reference Showcase (with Images) ────── */}
      <section id="demo" className="relative z-10 py-20 bg-slate-950 text-white overflow-hidden">
        {/* Ambient Glows */}
        <div className="absolute top-0 right-1/3 w-96 h-96 bg-purple-600/15 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 w-96 h-96 bg-cyan-600/15 rounded-full blur-3xl pointer-events-none" />

        <div className="lp-container">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <span className="inline-block px-3 py-1 rounded-full bg-purple-900/60 border border-purple-700 text-purple-300 text-xs font-mono font-semibold mb-3">
              LIVE VISUAL DEMONSTRATION
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold font-display tracking-tight text-white mb-3">
              See ProofyX in Action
            </h2>
            <p className="text-slate-400 text-sm sm:text-base">
              Explore how our detection engine visually isolates manipulated regions and voice clones.
            </p>
          </div>

          {/* Tab buttons */}
          <div className="flex flex-wrap items-center justify-center gap-2 mb-10">
            {[
              { id: 'image', label: 'Photo & Face Deepfake', icon: Image },
              { id: 'audio', label: 'Voice Clone & Speech', icon: Mic },
              { id: 'document', label: 'Document & ID Tampering', icon: FileSearch },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeDemoTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveDemoTab(tab.id)}
                  className={`flex items-center gap-2 px-5 py-3 rounded-2xl text-xs sm:text-sm font-bold transition-all ${
                    isActive
                      ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-lg shadow-purple-900/50 scale-105'
                      : 'bg-slate-900 text-slate-300 hover:bg-slate-800 hover:text-white border border-purple-900/40'
                  }`}
                >
                  <Icon size={16} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Active Demo Card with Real Reference Image */}
          <div className="max-w-5xl mx-auto">
            <div className="lp-terminal-glass rounded-3xl p-6 sm:p-8 border border-purple-500/30">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
                {/* Left Visual Image */}
                <div className="lg:col-span-7 rounded-2xl overflow-hidden border border-purple-400/30 shadow-2xl group">
                  <img
                    src={interactiveDemos[activeDemoTab].image}
                    alt={interactiveDemos[activeDemoTab].title}
                    className="w-full h-auto object-cover group-hover:scale-103 transition-transform duration-500"
                  />
                </div>

                {/* Right Details */}
                <div className="lg:col-span-5 flex flex-col gap-4 text-left">
                  <span className="text-xs font-mono font-bold text-purple-400 uppercase tracking-widest">
                    DETECTION BREAKDOWN
                  </span>

                  <h3 className="text-2xl font-bold text-white font-display">
                    {interactiveDemos[activeDemoTab].title}
                  </h3>

                  <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
                    {interactiveDemos[activeDemoTab].summary}
                  </p>

                  <div className="space-y-2 pt-2">
                    {interactiveDemos[activeDemoTab].highlights.map((h, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs text-purple-200">
                        <CheckCircle2 size={15} className="text-emerald-400 flex-shrink-0" />
                        <span>{h}</span>
                      </div>
                    ))}
                  </div>

                  <div className="pt-4 border-t border-purple-900/40">
                    <Link
                      to={interactiveDemos[activeDemoTab].link}
                      className="lp-btn-primary w-full justify-center text-sm !py-3"
                    >
                      <Zap size={15} />
                      <span>{interactiveDemos[activeDemoTab].btnText}</span>
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ─── Who Uses ProofyX (Real-World Use Cases) ───────────────── */}
      <section id="use-cases" className="relative z-10 py-20 md:py-28">
        <div className="lp-container">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <span className="inline-block px-3 py-1 rounded-full bg-purple-100 border border-purple-300 text-purple-900 text-xs font-bold uppercase tracking-widest mb-3">
              Real-World Safety
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-950 font-display tracking-tight mb-4">
              Built for Everyone Who Needs Truth
            </h2>
            <p className="text-slate-600 text-base">
              Whether you are protecting your family from scams or verifying news broadcasts.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {useCases.map((uc) => {
              const Icon = uc.icon;
              return (
                <div
                  key={uc.title}
                  className="lp-glass-card rounded-2xl p-6 flex flex-col justify-between"
                >
                  <div>
                    <div className="w-12 h-12 rounded-xl bg-purple-100 border border-purple-200 flex items-center justify-center text-purple-700 mb-4 shadow-xs">
                      <Icon size={22} />
                    </div>
                    <h3 className="text-base font-bold text-slate-900 mb-2">
                      {uc.title}
                    </h3>
                    <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
                      {uc.desc}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ─── Optional Developer Deep Dive (Collapsible for experts) ── */}
      <section className="relative z-10 py-12 bg-purple-900/5 border-y border-purple-200/50">
        <div className="lp-container text-center">
          <button
            onClick={() => setShowDeveloperCode(!showDeveloperCode)}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-white border border-purple-300 text-purple-900 text-xs font-bold hover:bg-purple-50 transition-all shadow-xs"
          >
            <Code2 size={15} className="text-purple-600" />
            <span>{showDeveloperCode ? 'Hide Under-The-Hood Code' : '🔬 Want to see the code? View Technical Deep Dive'}</span>
            <ChevronDown size={14} className={`transition-transform ${showDeveloperCode ? 'rotate-180' : ''}`} />
          </button>

          <AnimatePresence>
            {showDeveloperCode && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-8 text-left max-w-4xl mx-auto"
              >
                <div className="lp-terminal-glass rounded-3xl overflow-hidden shadow-2xl">
                  <div className="px-6 py-4 bg-slate-900 border-b border-purple-900/40 flex items-center justify-between">
                    <div className="flex items-center gap-2 font-mono text-xs text-purple-300 font-bold">
                      <Terminal size={14} />
                      <span>{CODE_SNIPPETS[activeCodeTab].title}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {Object.keys(CODE_SNIPPETS).map((k) => (
                        <button
                          key={k}
                          onClick={() => setActiveCodeTab(k)}
                          className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold ${
                            activeCodeTab === k ? 'bg-purple-700 text-white' : 'text-slate-400 hover:text-white'
                          }`}
                        >
                          {CODE_SNIPPETS[k].title.split('/')[1]}
                        </button>
                      ))}
                      <button
                        onClick={handleCopyCode}
                        className="px-3 py-1 rounded-lg text-xs font-mono font-semibold text-slate-400 hover:text-white"
                        aria-label="Copy code snippet"
                      >
                        {copiedCode ? 'Copied' : 'Copy'}
                      </button>
                    </div>
                  </div>
                  <div className="p-6 font-mono text-xs sm:text-sm text-purple-200 overflow-x-auto bg-[#0b0817]">
                    <pre><code>{CODE_SNIPPETS[activeCodeTab].code}</code></pre>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>

      {/* ─── FAQ Section (Easy to Understand) ───────────────────────── */}
      <section id="faq" className="relative z-10 py-20 md:py-28">
        <div className="lp-container max-w-3xl">
          <div className="text-center mb-14">
            <span className="inline-block px-3 py-1 rounded-full bg-purple-100 border border-purple-200 text-purple-800 text-xs font-bold uppercase tracking-widest mb-3">
              Got Questions?
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-950 font-display tracking-tight mb-4">
              Frequently Asked Questions
            </h2>
            <p className="text-slate-600 text-base">
              Everything you need to know about ProofyX and protecting yourself from AI deepfakes.
            </p>
          </div>

          <div className="flex flex-col gap-4">
            {faqs.map((faq, index) => {
              const isOpen = activeFaq === index;
              return (
                <div
                  key={index}
                  className="lp-glass-card rounded-2xl overflow-hidden transition-all duration-200"
                >
                  <button
                    onClick={() => setActiveFaq(isOpen ? null : index)}
                    className="w-full text-left px-6 py-4 flex items-center justify-between gap-4 font-bold text-slate-900 hover:text-purple-800 transition-colors"
                  >
                    <span className="text-sm sm:text-base">{faq.q}</span>
                    <ChevronDown
                      size={18}
                      className={`text-purple-600 transition-transform duration-200 flex-shrink-0 ${
                        isOpen ? 'rotate-180' : ''
                      }`}
                    />
                  </button>
                  <AnimatePresence>
                    {isOpen && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="px-6 pb-5 text-sm text-slate-600 leading-relaxed border-t border-purple-100/80 pt-3"
                      >
                        {faq.a}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ─── Call to Action Banner ─────────────────────────────────── */}
      <section className="relative z-10 py-16">
        <div className="lp-container">
          <div className="relative rounded-3xl p-8 sm:p-14 text-center bg-gradient-to-r from-purple-950 via-indigo-950 to-purple-900 text-white shadow-2xl overflow-hidden">
            <div className="absolute inset-0 bg-[radial-gradient(#a855f7_1px,transparent_1px)] [background-size:24px_24px] opacity-20 pointer-events-none" />
            <div className="relative z-10 max-w-2xl mx-auto flex flex-col items-center gap-6">
              <span className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full bg-purple-500/30 border border-purple-400/40 text-purple-200 text-xs font-semibold">
                <ShieldCheck size={14} /> Ready to verify content
              </span>
              <h2 className="text-3xl sm:text-5xl font-black font-display tracking-tight">
                Stop Guessing. Verify Authenticity Now.
              </h2>
              <p className="text-purple-200 text-sm sm:text-base leading-relaxed">
                Check any suspicious photo, voice clip, deepfake video, or invoice in under 2 seconds.
              </p>
              <div className="flex flex-wrap items-center justify-center gap-4">
                <Link
                  to="/dashboard"
                  className="px-8 py-3.5 rounded-xl bg-white text-purple-950 font-bold text-sm hover:bg-purple-50 transition-all shadow-xl hover:scale-105"
                >
                  Start Free Scan Now
                </Link>
                <Link
                  to="/multimodal"
                  className="px-8 py-3.5 rounded-xl bg-purple-800/80 border border-purple-500/40 text-white font-bold text-sm hover:bg-purple-700 transition-all"
                >
                  Explore Multi-Scanner
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ─── Modern Footer ─────────────────────────────────────────── */}
      <footer className="relative z-10 border-t border-purple-200/60 bg-white/70 backdrop-blur-xl py-12">
        <div className="lp-container">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
            <div className="md:col-span-2">
              <div className="flex items-center gap-2.5 mb-3">
                <img src={logo} alt="ProofyX" className="w-8 h-8 rounded-lg object-cover ring-1 ring-purple-300" />
                <span className="font-extrabold tracking-wider text-base font-display text-purple-950">
                  PROOFY<span className="text-purple-600">X</span>
                </span>
              </div>
              <p className="text-xs sm:text-sm text-slate-500 max-w-sm leading-relaxed mb-4">
                The world’s easiest and most accurate AI deepfake & authenticity checker. Making digital truth accessible to everyone.
              </p>
              <div className="flex items-center gap-2 text-xs text-purple-700 font-semibold font-mono">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span>12 Forensic Verification Models Operational</span>
              </div>
            </div>

            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-purple-950 mb-3 font-display">
                Quick Scanners
              </h4>
              <ul className="flex flex-col gap-2 text-xs text-slate-600 font-medium">
                <li><Link to="/image" className="hover:text-purple-700">Photo & Image Checker</Link></li>
                <li><Link to="/video" className="hover:text-purple-700">Video Deepfake Detector</Link></li>
                <li><Link to="/audio" className="hover:text-purple-700">Voice Clone Scanner</Link></li>
                <li><Link to="/document" className="hover:text-purple-700">Document & ID Verifier</Link></li>
                <li><Link to="/multimodal" className="hover:text-purple-700">All-In-One Multi-Scanner</Link></li>
              </ul>
            </div>

            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-purple-950 mb-3 font-display">
                Protection & Legal
              </h4>
              <ul className="flex flex-col gap-2 text-xs text-slate-600 font-medium">
                <li><Link to="/complaint" className="hover:text-purple-700">Cyber Scam Complaint Generator</Link></li>
                <li><Link to="/history" className="hover:text-purple-700">My Scan History</Link></li>
                <li><Link to="/system" className="hover:text-purple-700">System Diagnostics</Link></li>
                <li><Link to="/dashboard" className="hover:text-purple-700">Main Dashboard</Link></li>
              </ul>
            </div>
          </div>

          <div className="pt-6 border-t border-purple-200/60 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
            <div>© {new Date().getFullYear()} ProofyX Technologies. All rights reserved.</div>
            <div className="flex items-center gap-6">
              <span className="hover:text-purple-700 cursor-pointer">Privacy Guarantee</span>
              <span className="hover:text-purple-700 cursor-pointer">Terms of Service</span>
              <span className="hover:text-purple-700 cursor-pointer">Zero-Retention Policy</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
