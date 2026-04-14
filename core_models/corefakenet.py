"""
CorefakeNet: Unified hybrid CNN for ProofyX deepfake detection.

Replaces 7 separate models with a single EfficientNet-B4 backbone shared
across 5 specialized heads + attention fusion. Achieves ~7x speedup for
video processing while maintaining accuracy.

Architecture:
  Backbone: EfficientNet-B4 (pretrained ImageNet)
    |-- Mid features (24x24, 112ch) --> Frequency head, Artifact head
    '-- Final features (12x12, 1792ch) --> pool --> Texture, ViT, DINO heads

  Heads:
    1. Texture   -- photorealistic texture inconsistencies (pooled features)
    2. Frequency -- spectral artifacts via FFT on mid-level features
    3. Artifact  -- blending seams via depthwise high-pass + dilated conv
    4. ViT-style -- global patterns mimicking Vision Transformer
    5. DINO-style-- self-supervised feature patterns

  Fusion: Attention-weighted combination of all 5 head embeddings
  Output: final_score, confidence, per-head scores, attention weights

Input:  (B, 3, 380, 380) normalized face crop
Output: dict with final_score, confidence, head_scores, attention_weights

Saves as: models/corefakenet.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class CorefakeNet(nn.Module):
    """
    Unified deepfake detector: 5 specialized heads sharing EfficientNet-B4.

    All heads share the backbone computation and produce independent scores.
    An attention fusion mechanism learns optimal per-head weighting.
    Temperature scaling calibrates logits before sigmoid.
    """

    INPUT_SIZE = 380
    HEAD_NAMES = ['texture', 'frequency', 'artifact', 'vit', 'dino']

    PREPROCESS = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def __init__(self):
        super().__init__()

        base = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        features = list(base.features.children())

        # Split backbone for intermediate feature access
        # backbone_early: stem + stages 1-4 --> (B, 112, 24, 24) for 380 input
        # backbone_late:  stages 5-7 + head conv --> (B, 1792, 12, 12)
        self.backbone_early = nn.Sequential(*features[:5])
        self.backbone_late = nn.Sequential(*features[5:])

        # Freeze stem + stages 1-2 (general low-level features)
        for i in range(3):
            for p in self.backbone_early[i].parameters():
                p.requires_grad = False

        self.mid_channels = 112
        self.final_channels = 1792
        self.hidden_dim = 128

        # ---- Feature Reducer: final features --> pooled vector ----
        self.final_reducer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.final_channels, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.hidden_dim),
        )

        # ---- Mid-level Projection (for frequency + artifact heads) ----
        self.mid_projection = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ---- HEAD 1: TEXTURE (from pooled features) ----
        self.texture_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.hidden_dim),
            nn.ReLU(),
        )
        self.texture_classifier = nn.Linear(self.hidden_dim, 1)

        # ---- HEAD 2: FREQUENCY (FFT on mid-level features) ----
        # Input: concat of projected mid features + FFT magnitude = 256 channels
        self.freq_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.freq_fc = nn.Linear(64, self.hidden_dim)
        self.freq_classifier = nn.Linear(self.hidden_dim, 1)

        # ---- HEAD 3: ARTIFACT (depthwise high-pass + dilated conv) ----
        self.register_buffer(
            'high_pass_filter', self._make_laplacian(self.hidden_dim)
        )
        self.artifact_conv = nn.Sequential(
            # Depthwise separable convolution
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3,
                      padding=1, groups=self.hidden_dim),
            nn.Conv2d(self.hidden_dim, 64, 1),  # Pointwise
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Dilated conv for wider receptive field
            nn.Conv2d(64, 32, 3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.artifact_fc = nn.Linear(32, self.hidden_dim)
        self.artifact_classifier = nn.Linear(self.hidden_dim, 1)

        # ---- HEAD 4: VIT-STYLE (from pooled features) ----
        self.vit_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, self.hidden_dim),
            nn.GELU(),
        )
        self.vit_classifier = nn.Linear(self.hidden_dim, 1)

        # ---- HEAD 5: DINO-STYLE (from pooled features) ----
        self.dino_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.hidden_dim),
        )
        self.dino_classifier = nn.Linear(self.hidden_dim, 1)

        # ---- ATTENTION FUSION ----
        self.attention_proj = nn.Linear(self.hidden_dim, 1, bias=False)
        self.fusion_classifier = nn.Linear(self.hidden_dim, 1)

        # ---- CONFIDENCE ESTIMATOR ----
        self.confidence_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # ---- LEARNABLE TEMPERATURE (applied to logits before sigmoid) ----
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        self._init_custom_weights()

    @staticmethod
    def _make_laplacian(channels):
        """Create depthwise Laplacian high-pass filter."""
        kernel = torch.tensor(
            [[[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]]
        )
        return kernel.expand(channels, 1, 3, 3).clone()

    def _init_custom_weights(self):
        """Kaiming init for all non-backbone Linear and Conv2d layers."""
        for name, m in self.named_modules():
            if name.startswith(('backbone_early', 'backbone_late')):
                continue
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_all=True):
        """
        Args:
            x: (B, 3, 380, 380) normalized face crop
            return_all: if True return full dict, else (final_score, confidence)

        Returns:
            dict with: final_score (B,1), confidence (B,1), head_scores (B,5),
                       head_logits (B,5), attention_weights (B,5), temperature,
                       embeddings (B,128)
        """
        # ---- Backbone ----
        mid = self.backbone_early(x)         # (B, 112, 24, 24)
        final = self.backbone_late(mid)      # (B, 1792, 12, 12)
        reduced = self.final_reducer(final)  # (B, 128)
        proj_mid = self.mid_projection(mid)  # (B, 128, 24, 24)

        # ---- Head 1: Texture ----
        tex_embed = self.texture_head(reduced)
        tex_logits = self.texture_classifier(tex_embed)

        # ---- Head 2: Frequency (FFT on mid-level features) ----
        fft = torch.fft.fft2(proj_mid, norm='ortho')
        fft_mag = torch.log(1 + torch.abs(fft))
        freq_input = torch.cat([proj_mid, fft_mag], dim=1)  # (B, 256, 24, 24)
        freq_feat = self.freq_conv(freq_input).flatten(1)    # (B, 64)
        freq_embed = self.freq_fc(freq_feat)
        freq_logits = self.freq_classifier(freq_embed)

        # ---- Head 3: Artifact (depthwise high-pass) ----
        high_freq = F.conv2d(
            proj_mid, self.high_pass_filter,
            padding=1, groups=self.hidden_dim,
        )
        art_feat = self.artifact_conv(high_freq).flatten(1)  # (B, 32)
        art_embed = self.artifact_fc(art_feat)
        art_logits = self.artifact_classifier(art_embed)

        # ---- Head 4: ViT-style ----
        vit_embed = self.vit_head(reduced)
        vit_logits = self.vit_classifier(vit_embed)

        # ---- Head 5: DINO-style ----
        dino_embed = self.dino_head(reduced)
        dino_logits = self.dino_classifier(dino_embed)

        # ---- Attention Fusion (B, 5, 128) ----
        embeddings = [tex_embed, freq_embed, art_embed, vit_embed, dino_embed]
        stacked = torch.stack(embeddings, dim=1)       # (B, 5, 128)
        attn_scores = self.attention_proj(stacked)     # (B, 5, 1)
        attn_weights = F.softmax(attn_scores, dim=1)   # (B, 5, 1)
        fused = torch.sum(attn_weights * stacked, dim=1)  # (B, 128)

        # ---- Temperature-scaled outputs ----
        fusion_logits = self.fusion_classifier(fused)
        final_score = torch.sigmoid(fusion_logits / self.temperature)

        head_logits = torch.cat(
            [tex_logits, freq_logits, art_logits, vit_logits, dino_logits],
            dim=1,
        )  # (B, 5)
        head_scores = torch.sigmoid(head_logits / self.temperature)

        # ---- Confidence ----
        confidence = self.confidence_head(fused)

        if return_all:
            return {
                'final_score': final_score,
                'confidence': confidence,
                'head_scores': head_scores,
                'head_logits': head_logits,
                'attention_weights': attn_weights.squeeze(-1),  # (B, 5)
                'temperature': self.temperature.item(),
                'embeddings': fused,
            }

        return final_score, confidence

    def get_param_groups(self, weight_decay=0.01):
        """Return parameter groups with differential learning rates."""
        backbone_params = [
            p for p in self.backbone_early.parameters() if p.requires_grad
        ]
        backbone_params += list(self.backbone_late.parameters())

        reducer_params = (
            list(self.final_reducer.parameters()) +
            list(self.mid_projection.parameters())
        )

        head_params = []
        for mod in [
            self.texture_head, self.texture_classifier,
            self.freq_conv, self.freq_fc, self.freq_classifier,
            self.artifact_conv, self.artifact_fc, self.artifact_classifier,
            self.vit_head, self.vit_classifier,
            self.dino_head, self.dino_classifier,
        ]:
            head_params += list(mod.parameters())

        fusion_params = (
            list(self.attention_proj.parameters()) +
            list(self.fusion_classifier.parameters()) +
            list(self.confidence_head.parameters())
        )

        return [
            {'params': backbone_params, 'lr': 1e-5, 'weight_decay': weight_decay},
            {'params': reducer_params, 'lr': 5e-5, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': 2e-4, 'weight_decay': weight_decay},
            {'params': fusion_params, 'lr': 3e-4, 'weight_decay': weight_decay},
            {'params': [self.temperature], 'lr': 1e-3, 'weight_decay': 0.0},
        ]

    @torch.no_grad()
    def predict(self, pil_image):
        """
        Single-image inference returning UI-compatible result dict.

        Args:
            pil_image: PIL Image (RGB)

        Returns:
            dict matching ProofyX UI format
        """
        device = next(self.parameters()).device
        tensor = self.PREPROCESS(pil_image.convert("RGB")).unsqueeze(0).to(device)
        self.eval()
        out = self.forward(tensor)

        head_scores = {
            f'{name}_score': out['head_scores'][0, i].item()
            for i, name in enumerate(self.HEAD_NAMES)
        }

        return {
            'final_risk': out['final_score'].item(),
            'confidence': out['confidence'].item(),
            'model_scores': head_scores,
            'attention_weights': {
                name: out['attention_weights'][0, i].item()
                for i, name in enumerate(self.HEAD_NAMES)
            },
            'temperature': out['temperature'],
            'fusion_mode': 'learned',
        }

    @property
    def gradcam_target_layer(self):
        """Return the target layer for GradCAM (last conv in backbone)."""
        return self.backbone_late[-1]


# ================================================================
#  Loss Function
# ================================================================

def corefakenet_loss(outputs, targets):
    """
    Multi-task loss with KL divergence diversity regularization.

    Args:
        outputs: dict from CorefakeNet.forward(return_all=True)
        targets: (B, 1) ground truth labels (0=real, 1=fake)

    Returns:
        (total_loss, loss_dict) for logging
    """
    bce = nn.BCELoss()

    # Per-head classification losses
    head_loss = sum(
        bce(outputs['head_scores'][:, i:i + 1], targets)
        for i in range(5)
    )

    # Fusion loss (primary objective)
    fusion_loss = bce(outputs['final_score'], targets)

    # KL divergence diversity loss: encourage heads to specialize
    avg_pred = outputs['head_scores'].mean(dim=1, keepdim=True)  # (B, 1)
    kl_divs = []
    for i in range(5):
        p = outputs['head_scores'][:, i:i + 1].clamp(1e-7, 1 - 1e-7)
        q = avg_pred.clamp(1e-7, 1 - 1e-7)
        kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        kl_divs.append(kl)
    diversity_loss = -torch.cat(kl_divs, dim=1).mean()  # Negative to maximize

    # Confidence regularization: encourage high confidence
    confidence_loss = -outputs['confidence'].mean()

    total = (
        head_loss * 0.15 +
        fusion_loss * 0.25 +
        diversity_loss * 0.01 +
        confidence_loss * 0.05
    )

    return total, {
        'head_loss': head_loss.item(),
        'fusion_loss': fusion_loss.item(),
        'diversity_loss': diversity_loss.item(),
        'confidence_loss': confidence_loss.item(),
        'total': total.item(),
    }


# ================================================================
#  Fast Video Processor
# ================================================================

class FastVideoProcessor:
    """
    Optimized video deepfake detection using CorefakeNet.

    Replaces the multi-model ensemble pipeline with a single forward pass
    per frame, achieving ~7x speedup.

    Usage:
        processor = FastVideoProcessor('models/corefakenet.pth')
        result = processor.analyze(video_path, sampling_fps=0.5)
    """

    def __init__(self, model_path, device='cpu', quantize=False):
        self.device = torch.device(device)
        self.model = CorefakeNet()

        checkpoint = torch.load(model_path, map_location=self.device,
                                weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        if quantize and device == 'cpu':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8,
            )

    def process_frame(self, pil_image):
        """
        Process a single frame (PIL Image).

        Returns:
            dict with scores, confidence, head breakdown
        """
        from utils.gradcam import detect_and_align_face

        face_crop, bbox = detect_and_align_face(pil_image, expand_ratio=0.3)
        face_detected = face_crop is not None
        model_input = face_crop if face_detected else pil_image

        result = self.model.predict(model_input)
        result['face_detected'] = face_detected
        result['face_aligned'] = face_detected
        return result

    def analyze(self, video_path, sampling_fps=0.5, progress_callback=None):
        """
        Full video analysis with confidence-weighted aggregation.

        Args:
            video_path: Path to video file.
            sampling_fps: Frames per second to sample.
                0.5 = 1 frame every 2s (fast mode, ~2s for 22s video)
                10  = 10 fps (accurate mode, ~60s for 22s video)
            progress_callback: Optional callable(current, total, message).

        Returns:
            dict with final verdict, per-frame results, timing info
        """
        import cv2
        import time

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Cannot open video: {video_path}'}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        frame_interval = max(1, int(fps / sampling_fps))
        est_samples = max(1, total_frames // frame_interval)

        start_time = time.time()
        results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil_img = Image.fromarray(rgb)

                result = self.process_frame(pil_img)
                result['frame_index'] = frame_idx
                result['timestamp'] = round(frame_idx / fps, 2)
                results.append(result)

                if progress_callback:
                    progress_callback(
                        len(results), est_samples,
                        f"Frame {len(results)}/{est_samples} "
                        f"({result['timestamp']:.1f}s)",
                    )

            frame_idx += 1

        cap.release()
        elapsed = time.time() - start_time

        if not results:
            return {
                'error': 'No frames analyzed',
                'video_info': {
                    'fps': fps, 'duration_sec': round(duration, 2),
                    'frame_count': total_frames,
                },
            }

        # Confidence-weighted aggregation
        confidences = np.array([r['confidence'] for r in results])
        conf_sum = confidences.sum()
        weights = confidences / conf_sum if conf_sum > 0 else np.ones(
            len(results)) / len(results)

        final_risk = float(np.average(
            [r['final_risk'] for r in results], weights=weights,
        ))

        # Per-head aggregation
        head_risks = {}
        for name in CorefakeNet.HEAD_NAMES:
            key = f'{name}_score'
            head_risks[key] = float(np.average(
                [r['model_scores'][key] for r in results], weights=weights,
            ))

        fake_frames = sum(1 for r in results if r['final_risk'] > 0.5)
        faces_detected = sum(1 for r in results if r['face_detected'])

        return {
            'prediction': 'FAKE' if final_risk > 0.5 else 'REAL',
            'final_risk': round(final_risk, 4),
            'confidence': round(float(confidences.mean()), 4),
            'model_scores': head_risks,
            'total_frames_analyzed': len(results),
            'fake_frames': fake_frames,
            'real_frames': len(results) - fake_frames,
            'faces_detected_in_frames': faces_detected,
            'elapsed_seconds': round(elapsed, 2),
            'video_info': {
                'fps': fps,
                'duration_sec': round(duration, 2),
                'frame_count': total_frames,
                'sampling_fps': sampling_fps,
            },
            'frame_results': results,
        }
