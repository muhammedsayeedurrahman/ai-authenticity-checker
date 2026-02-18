"""
Audio deepfake detection pipeline.

Integrates the zo9999 CNN mel-spectrogram model with:
  - Audio loading and format conversion
  - Mel-spectrogram preprocessing (matching zo9999 exactly)
  - Windowed analysis for long audio (3-5s segments)
  - Spectral artifact detection (explainability)
  - Unified JSON output compatible with the multimodal pipeline
"""

import os
import sys
import tempfile
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import librosa
import soundfile as sf

from core_models.audio_deepfake_model import AudioDeepfakeCNN

# ================= CONFIG =================
SAMPLE_RATE = 22050          # librosa default, matches zo9999 training
N_MELS = 91                  # mel bands (matches zo9999)
MAX_TIME_STEPS = 150         # time steps per segment (matches zo9999)
HOP_LENGTH = 512             # librosa default
N_FFT = 2048                 # librosa default
SEGMENT_DURATION = 3.5       # seconds per analysis window (~150 time steps)
SEGMENT_OVERLAP = 0.5        # 50% overlap between windows
MODEL_PATH = os.path.join(ROOT_DIR, "models", "audio_deepfake_model.pth")

# Supported audio formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
# ==========================================


def load_audio(file_path):
    """
    Load audio file and return waveform at 22050 Hz.

    Supports WAV, MP3, FLAC, M4A, OGG, AAC, WMA.

    Returns:
        (waveform, sample_rate) tuple. Waveform is mono float32 numpy array.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    # librosa handles most formats via soundfile/audioread
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(waveform) == 0:
        raise ValueError("Audio file is empty or could not be decoded")

    return waveform, sr


def generate_mel_spectrogram(waveform, sr=SAMPLE_RATE):
    """
    Generate mel-spectrogram from waveform.

    Matches zo9999 preprocessing exactly:
      - n_mels=91, n_fft=2048, hop_length=512
      - power_to_db with ref=np.max
      - Pad/truncate to 150 time steps

    Args:
        waveform: mono float32 numpy array
        sr: sample rate (default 22050)

    Returns:
        (n_mels, MAX_TIME_STEPS) numpy array in dB scale
    """
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or truncate to fixed width
    if mel_db.shape[1] < MAX_TIME_STEPS:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, MAX_TIME_STEPS - mel_db.shape[1])),
            mode="constant",
        )
    else:
        mel_db = mel_db[:, :MAX_TIME_STEPS]

    return mel_db


def preprocess_audio(waveform, sr=SAMPLE_RATE):
    """
    Full preprocessing: waveform -> model-ready tensor.

    Returns:
        torch tensor of shape (1, 1, 91, 150)
    """
    mel_db = generate_mel_spectrogram(waveform, sr)
    # Add batch and channel dims: (1, 1, 91, 150)
    tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


def segment_audio(waveform, sr=SAMPLE_RATE, segment_sec=SEGMENT_DURATION,
                  overlap=SEGMENT_OVERLAP):
    """
    Split audio into overlapping segments for windowed analysis.

    Args:
        waveform: mono float32 numpy array
        sr: sample rate
        segment_sec: segment duration in seconds
        overlap: overlap ratio (0.0 - 0.9)

    Returns:
        list of (start_time, end_time, waveform_segment) tuples
    """
    segment_samples = int(segment_sec * sr)
    step_samples = int(segment_samples * (1 - overlap))

    if step_samples < 1:
        step_samples = segment_samples

    segments = []
    total_samples = len(waveform)

    pos = 0
    while pos < total_samples:
        end = min(pos + segment_samples, total_samples)
        seg = waveform[pos:end]

        # Only analyze segments with at least 0.5s of audio
        if len(seg) >= int(0.5 * sr):
            start_time = pos / sr
            end_time = end / sr
            segments.append((start_time, end_time, seg))

        pos += step_samples

    return segments


def analyze_spectral_artifacts(mel_db):
    """
    Detect spectral artifacts indicative of AI-generated audio.

    Analyzes:
      1. Harmonic structure regularity (vocoders produce unnaturally regular harmonics)
      2. High-frequency energy distribution (AI audio often lacks natural high-freq detail)
      3. Spectral flatness (AI audio tends toward flatter spectra in certain bands)
      4. Phase consistency (indirectly via mel energy distribution)

    Args:
        mel_db: (91, 150) mel-spectrogram in dB scale

    Returns:
        dict with artifact indicators and evidence list
    """
    evidence = []
    scores = {}

    # 1. Harmonic regularity — check variance across mel bands
    band_means = mel_db.mean(axis=1)  # (91,)
    band_stds = mel_db.std(axis=1)

    # AI vocoders produce very uniform energy across time in each band
    temporal_uniformity = 1.0 - np.clip(band_stds.mean() / 15.0, 0.0, 1.0)
    scores["harmonic_regularity"] = float(temporal_uniformity)
    if temporal_uniformity > 0.6:
        evidence.append("unnatural harmonic structure")

    # 2. High-frequency energy — real speech has more natural HF variation
    low_bands = mel_db[:30, :].mean()   # low freq
    high_bands = mel_db[60:, :].mean()  # high freq
    hf_ratio = (high_bands - low_bands) / (abs(low_bands) + 1e-8)

    # AI audio often has less natural HF rolloff variation
    hf_artifact = np.clip((hf_ratio + 0.8) / 0.4, 0.0, 1.0)
    scores["high_freq_artifact"] = float(hf_artifact)
    if hf_artifact > 0.5:
        evidence.append("spectral artifacts")

    # 3. Spectral flatness — measure across time frames
    frame_energies = mel_db.mean(axis=0)  # (150,)
    non_silent = frame_energies[frame_energies > mel_db.min() + 10]

    if len(non_silent) > 5:
        flatness = float(np.exp(np.mean(np.log(np.abs(non_silent) + 1e-8))) /
                        (np.mean(np.abs(non_silent)) + 1e-8))
        flatness_score = np.clip(flatness, 0.0, 1.0)
    else:
        flatness_score = 0.5
    scores["spectral_flatness"] = float(flatness_score)

    # 4. Phase consistency proxy — temporal smoothness of energy
    if len(frame_energies) > 3:
        diffs = np.abs(np.diff(frame_energies))
        smoothness = 1.0 - np.clip(diffs.std() / 10.0, 0.0, 1.0)
        scores["phase_consistency"] = float(smoothness)
        if smoothness > 0.7:
            evidence.append("phase inconsistency")
    else:
        scores["phase_consistency"] = 0.5

    # Combined artifact score
    combined = (
        0.30 * scores["harmonic_regularity"] +
        0.30 * scores["high_freq_artifact"] +
        0.20 * scores["spectral_flatness"] +
        0.20 * scores["phase_consistency"]
    )
    scores["combined_artifact_score"] = float(combined)

    if not evidence:
        if combined > 0.5:
            evidence.append("subtle vocoder artifacts detected")
        else:
            evidence.append("no significant spectral artifacts")

    return scores, evidence


class AudioAnalyzer:
    """
    Full audio deepfake detection pipeline.

    Usage:
        analyzer = AudioAnalyzer(device)
        result = analyzer.analyze("audio.wav")
        print(result)  # Unified JSON output
    """

    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        """Load the audio deepfake CNN model."""
        if not os.path.exists(MODEL_PATH):
            print(f"Audio model not found at {MODEL_PATH} (train first)")
            return

        try:
            self.model = AudioDeepfakeCNN().to(self.device)
            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            )
            self.model.eval()
            self.model_loaded = True
            print("Audio deepfake model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load audio model: {e}")
            self.model = None

    def analyze(self, audio_path, progress_callback=None):
        """
        Full audio analysis pipeline.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, M4A, etc.)
            progress_callback: Optional callable(current, total, message)

        Returns:
            dict with unified JSON output format
        """
        if not self.model_loaded:
            return {
                "media_type": "audio",
                "error": "Audio model not loaded. Train with: python training/train_audio_deepfake.py",
                "authenticity_score": None,
            }

        try:
            # Step 1: Load audio
            if progress_callback:
                progress_callback(0, 5, "Loading audio...")
            waveform, sr = load_audio(audio_path)
            duration = len(waveform) / sr

            # Step 2: Segment audio
            if progress_callback:
                progress_callback(1, 5, "Segmenting audio...")
            segments = segment_audio(waveform, sr)

            if not segments:
                return {
                    "media_type": "audio",
                    "error": "Audio too short to analyze (minimum 0.5 seconds)",
                    "authenticity_score": None,
                }

            # Step 3: Analyze each segment
            segment_results = []
            suspicious_timestamps = []
            all_evidence = set()

            for i, (start_t, end_t, seg_wave) in enumerate(segments):
                if progress_callback:
                    progress_callback(
                        2 + i, 2 + len(segments),
                        f"Analyzing segment {i+1}/{len(segments)} ({start_t:.1f}s - {end_t:.1f}s)"
                    )

                # Generate mel-spectrogram
                mel_db = generate_mel_spectrogram(seg_wave, sr)

                # Run CNN model
                tensor = torch.tensor(
                    mel_db, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    probs = self.model(tensor)
                    fake_prob = probs[0, 0].item()
                    real_prob = probs[0, 1].item()

                # Spectral artifact analysis
                artifact_scores, evidence = analyze_spectral_artifacts(mel_db)
                all_evidence.update(evidence)

                seg_result = {
                    "start_time": round(start_t, 2),
                    "end_time": round(end_t, 2),
                    "fake_probability": round(fake_prob, 4),
                    "real_probability": round(real_prob, 4),
                    "artifact_scores": artifact_scores,
                    "evidence": evidence,
                }
                segment_results.append(seg_result)

                # Track suspicious segments
                if fake_prob > 0.5:
                    suspicious_timestamps.append(round(start_t, 1))

            # Step 4: Aggregate results
            if progress_callback:
                progress_callback(len(segments) + 2, len(segments) + 3, "Computing final score...")

            fake_probs = [s["fake_probability"] for s in segment_results]
            avg_fake = sum(fake_probs) / len(fake_probs)

            # Weighted average: higher-confidence segments get more weight
            weights = [abs(p - 0.5) + 0.5 for p in fake_probs]
            total_w = sum(weights)
            weighted_fake = sum(p * w for p, w in zip(fake_probs, weights)) / total_w

            # Use weighted average as primary score
            final_fake_prob = weighted_fake

            # Authenticity score: 0 = definitely fake, 100 = definitely real
            authenticity_score = round((1.0 - final_fake_prob) * 100, 1)

            # Determine label and confidence
            if final_fake_prob > 0.7:
                label = "Likely Fake"
                confidence = "High"
                manipulation_type = "AI voice cloning / TTS"
            elif final_fake_prob > 0.5:
                label = "Possibly Fake"
                confidence = "Medium"
                manipulation_type = "Suspected AI generation"
            elif final_fake_prob > 0.3:
                label = "Uncertain"
                confidence = "Low"
                manipulation_type = "Inconclusive"
            else:
                label = "Likely Real"
                confidence = "High"
                manipulation_type = "None detected"

            # Build explanation
            explanation = self._build_explanation(
                final_fake_prob, all_evidence, segment_results
            )

            return {
                "media_type": "audio",
                "authenticity_score": authenticity_score,
                "fake_probability": round(final_fake_prob, 4),
                "label": label,
                "modality_scores": {
                    "image": None,
                    "video": None,
                    "audio": authenticity_score,
                },
                "manipulation_type": manipulation_type,
                "confidence": confidence,
                "evidence": sorted(all_evidence),
                "timestamps": suspicious_timestamps,
                "explanation": explanation,
                "duration_sec": round(duration, 2),
                "segments_analyzed": len(segment_results),
                "segment_results": segment_results,
            }

        except Exception as e:
            return {
                "media_type": "audio",
                "error": str(e),
                "authenticity_score": None,
            }

    def _build_explanation(self, fake_prob, evidence, segment_results):
        """Build human-readable explanation."""
        if fake_prob > 0.7:
            base = "Audio shows strong deepfake characteristics."
        elif fake_prob > 0.5:
            base = "Audio shows some deepfake indicators."
        elif fake_prob > 0.3:
            base = "Audio analysis is inconclusive."
        else:
            base = "Audio appears to be authentic human speech."

        evidence_list = sorted(evidence - {"no significant spectral artifacts"})
        if evidence_list:
            base += f" Detected: {', '.join(evidence_list)}."

        # Note segment consistency
        fake_probs = [s["fake_probability"] for s in segment_results]
        if len(fake_probs) > 1:
            variance = float(np.var(fake_probs))
            if variance > 0.05:
                base += " Inconsistent scores across segments suggest partial manipulation."

        return base
