"""
Mobile optimization utilities for the Audio Deepfake CNN.

Provides:
  - ONNX export (for cross-platform mobile inference)
  - INT8 quantization (PyTorch dynamic quantization)
  - Model size reporting

Usage:
    python utils/audio_export.py                    # Export ONNX
    python utils/audio_export.py --quantize         # Export quantized PyTorch
"""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.quantization
from core_models.audio_deepfake_model import AudioDeepfakeCNN

MODEL_PATH = os.path.join(ROOT_DIR, "models", "audio_deepfake_model.pth")
ONNX_PATH = os.path.join(ROOT_DIR, "models", "audio_deepfake_model.onnx")
QUANTIZED_PATH = os.path.join(ROOT_DIR, "models", "audio_deepfake_model_int8.pth")


def export_onnx(model_path=MODEL_PATH, output_path=ONNX_PATH):
    """Export the audio model to ONNX format for mobile deployment."""
    model = AudioDeepfakeCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.eval()

    # Dummy input: (batch=1, channels=1, n_mels=91, time_steps=150)
    dummy_input = torch.randn(1, 1, 91, 150)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["mel_spectrogram"],
        output_names=["probabilities"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size"},
            "probabilities": {0: "batch_size"},
        },
        opset_version=13,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model exported to: {output_path} ({size_mb:.1f} MB)")
    return output_path


def quantize_dynamic(model_path=MODEL_PATH, output_path=QUANTIZED_PATH):
    """Apply dynamic INT8 quantization for smaller model size."""
    model = AudioDeepfakeCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.eval()

    # Dynamic quantization on Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    torch.save(quantized_model.state_dict(), output_path)

    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100

    print(f"Quantized model saved to: {output_path}")
    print(f"  Original: {orig_size:.1f} MB -> Quantized: {quant_size:.1f} MB ({reduction:.0f}% reduction)")
    return output_path


def report_model_size(model_path=MODEL_PATH):
    """Print model parameter count and file size."""
    model = AudioDeepfakeCNN()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Audio Deepfake CNN:")
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Model file size     : {size_mb:.1f} MB")
    else:
        print(f"  Model file          : Not found ({model_path})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audio model export utilities")
    parser.add_argument("--quantize", action="store_true", help="Export INT8 quantized model")
    parser.add_argument("--onnx", action="store_true", help="Export ONNX model (default)")
    parser.add_argument("--info", action="store_true", help="Print model info only")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Train first with:")
        print("  python training/train_audio_deepfake.py")
        sys.exit(1)

    report_model_size()
    print()

    if args.info:
        sys.exit(0)

    if args.quantize:
        quantize_dynamic()

    if args.onnx or not args.quantize:
        export_onnx()
