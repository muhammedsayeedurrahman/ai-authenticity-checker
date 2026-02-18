"""
Standalone audio deepfake inference script.

Usage:
    python inference/predict_audio.py -f audio.wav
    python inference/predict_audio.py -f audio.mp3 --json
"""

import sys
import os
import argparse
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.audio_analyzer import AudioAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Audio deepfake detection")
    parser.add_argument("-f", "--file", required=True, help="Path to audio file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    analyzer = AudioAnalyzer()
    result = analyzer.analyze(args.file)

    if args.json:
        # Remove segment_results for cleaner JSON output
        output = {k: v for k, v in result.items() if k != "segment_results"}
        print(json.dumps(output, indent=2))
    else:
        if "error" in result:
            print(f"\nError: {result['error']}")
            sys.exit(1)

        print("\n=== AUDIO DEEPFAKE DETECTION ===")
        print(f"File               : {args.file}")
        print(f"Duration           : {result['duration_sec']}s")
        print(f"Segments Analyzed  : {result['segments_analyzed']}")
        print()
        print(f"Authenticity Score : {result['authenticity_score']}%")
        print(f"Fake Probability   : {result['fake_probability']:.4f}")
        print(f"Label              : {result['label']}")
        print(f"Confidence         : {result['confidence']}")
        print(f"Manipulation Type  : {result['manipulation_type']}")
        print()
        print(f"Evidence           : {', '.join(result['evidence'])}")
        if result['timestamps']:
            print(f"Suspicious Times   : {result['timestamps']}")
        print()
        print(f"Explanation        : {result['explanation']}")


if __name__ == "__main__":
    main()
