"""
GradCAM visualization for deepfake detection.

Provides artifact-focused Grad-CAM for all model types:
  - FaceDeepfakeModel (ResNet50)
  - ViT (gradient-based, not attention rollout)
  - EfficientNet
  - DINOv2

Heatmaps are generated on face-aligned crops and overlaid
only on the face region of the original image.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Hooks into the target convolutional layer to capture activations
    and gradients, then produces a heatmap.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate GradCAM heatmap for the given input tensor.

        Args:
            input_tensor: preprocessed image tensor.
            target_class: index of class to explain (None = use predicted class).

        Returns:
            heatmap: numpy array [H, W] normalized to [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        if target_class is None:
            if output.dim() == 2 and output.shape[1] > 1:
                target_class = output.argmax(dim=1).item()
            else:
                # Single output (sigmoid) — backward on the output directly
                target_class = None

        self.model.zero_grad()
        if target_class is not None and output.dim() == 2:
            score = output[0, target_class]
        else:
            score = output.squeeze()
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            return np.zeros((7, 7))

        # Handle different activation shapes
        if self.gradients.dim() == 3:
            # Transformer: (batch, seq_len, hidden) -> reshape to spatial
            grads = self.gradients
            acts = self.activations
            # Global average pooling over sequence
            weights = grads.mean(dim=1, keepdim=True)  # (B, 1, H)
            cam = (weights * acts).sum(dim=2)  # (B, seq_len)
            cam = cam[0, 1:]  # Remove CLS token
            grid_size = int(np.sqrt(cam.shape[0]))
            if grid_size * grid_size != cam.shape[0]:
                grid_size = int(np.ceil(np.sqrt(cam.shape[0])))
                cam_padded = torch.zeros(grid_size * grid_size)
                cam_padded[:cam.shape[0]] = cam
                cam = cam_padded
            cam = cam.reshape(grid_size, grid_size).cpu().numpy()
        else:
            # CNN: (batch, channels, H, W)
            weights = self.gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def cleanup(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


# ================================================================
#  Face alignment utilities
# ================================================================

def detect_and_align_face(pil_image, expand_ratio=0.3):
    """
    Detect and crop the face region from a PIL image.

    Args:
        pil_image: Input PIL image.
        expand_ratio: How much to expand the bounding box (0.3 = 30% padding).

    Returns:
        (face_crop, bbox) where bbox is (x1, y1, x2, y2) or (None, None).
    """
    from pipeline.face_gate import _ensure_model_files, _PROTOTXT, _CAFFEMODEL
    _ensure_model_files()

    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]

    net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _CAFFEMODEL)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    best_conf = 0.0
    best_box = None
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > 0.5 and confidence > best_conf:
            best_conf = confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None, None

    x1, y1, x2, y2 = best_box
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * expand_ratio)
    pad_y = int(bh * expand_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    face_crop = pil_image.crop((x1, y1, x2, y2))
    return face_crop, (x1, y1, x2, y2)


# ================================================================
#  Per-model Grad-CAM generators
# ================================================================

def get_gradcam_for_face_model(face_model, input_tensor):
    """
    Grad-CAM on FaceDeepfakeModel (ResNet50 layer4).
    Target: the "fake" direction (1 - P(real)).
    """
    target_layer = face_model.features[7]  # ResNet50 layer4

    cam = GradCAM(face_model, target_layer)
    try:
        heatmap = cam.generate(input_tensor)
    finally:
        cam.cleanup()

    return heatmap


def get_gradcam_for_vit(vit_model, vit_processor, pil_image, device):
    """
    Proper Grad-CAM for ViT deepfake detector.
    Hooks into the last encoder layer's output LayerNorm to get
    gradient-weighted activations (NOT attention rollout).

    Returns:
        heatmap: numpy array [H, W] normalized to [0, 1], or None.
    """
    try:
        inputs = vit_processor(images=pil_image.convert("RGB"), return_tensors="pt").to(device)

        # Target: last encoder block's layernorm_after (output of the block)
        encoder_layers = vit_model.vit.encoder.layer
        target_layer = encoder_layers[-1].layernorm_after

        cam = GradCAM(vit_model, target_layer)
        try:
            # Enable gradients
            pixel_values = inputs["pixel_values"].requires_grad_(True)
            output = vit_model(pixel_values=pixel_values)
            logits = output.logits

            # Find deepfake class index
            deepfake_idx = None
            for k, v in vit_model.config.id2label.items():
                if "fake" in v.lower() or "deep" in v.lower():
                    deepfake_idx = k
                    break
            if deepfake_idx is None:
                deepfake_idx = 1

            # Backward pass
            vit_model.zero_grad()
            score = logits[0, deepfake_idx]
            score.backward()

            if cam.gradients is None or cam.activations is None:
                return None

            # Activations shape: (1, seq_len, hidden_dim)
            grads = cam.gradients  # (1, seq_len, hidden)
            acts = cam.activations  # (1, seq_len, hidden)

            # Weight activations by gradients
            weights = grads.mean(dim=2, keepdim=True)  # (1, seq_len, 1)
            weighted = (weights * acts).sum(dim=2)  # (1, seq_len)

            # Remove CLS token, reshape to grid
            patch_cam = weighted[0, 1:].cpu().numpy()
            patch_cam = np.maximum(patch_cam, 0)  # ReLU

            grid_size = int(np.sqrt(len(patch_cam)))
            if grid_size * grid_size == len(patch_cam):
                heatmap = patch_cam.reshape(grid_size, grid_size)
            else:
                # Handle non-square
                heatmap = patch_cam[:grid_size * grid_size].reshape(grid_size, grid_size)

            # Resize to image size
            heatmap = cv2.resize(heatmap, (224, 224))

            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

            # Light smoothing to reduce patchiness
            heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

            return heatmap

        finally:
            cam.cleanup()

    except Exception as e:
        print(f"ViT Grad-CAM failed: {e}")
        return None


def get_gradcam_for_efficientnet(eff_model, input_tensor):
    """
    Grad-CAM on EfficientNet (last feature block).
    """
    try:
        target_layer = eff_model.model.features[-1]
        cam = GradCAM(eff_model, target_layer)
        try:
            heatmap = cam.generate(input_tensor)
        finally:
            cam.cleanup()
        return heatmap
    except Exception as e:
        print(f"EfficientNet Grad-CAM failed: {e}")
        return None


def get_gradcam_for_dino(dino_model, input_tensor):
    """
    Grad-CAM on DINOv2 (last encoder block norm).
    """
    try:
        target_layer = dino_model.backbone.encoder.layer[-1].norm1
        cam = GradCAM(dino_model, target_layer)
        try:
            heatmap = cam.generate(input_tensor)
        finally:
            cam.cleanup()
        return heatmap
    except Exception as e:
        print(f"DINOv2 Grad-CAM failed: {e}")
        return None


# ================================================================
#  Preprocessing
# ================================================================

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================================================================
#  Heatmap overlay and visualization
# ================================================================

def create_heatmap_overlay(pil_image, heatmap, alpha=0.6, size=(224, 224)):
    """
    Overlay a GradCAM heatmap on the original image.

    Args:
        pil_image: Original PIL image.
        heatmap: 2D numpy array [h, w] normalized to [0, 1].
        alpha: Blend factor.
        size: Output size.

    Returns:
        PIL Image with heatmap overlay.
    """
    img = np.array(pil_image.convert("RGB").resize(size))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Gamma correction for better artifact visibility
    heatmap_resized = np.power(heatmap_resized, 0.7)

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_HOT
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * img)
    return Image.fromarray(overlay)


def create_face_region_overlay(pil_image, heatmap, face_bbox, alpha=0.6, size=(400, 400)):
    """
    Overlay heatmap ONLY on the face region of the original image.

    Args:
        pil_image: Original full image.
        heatmap: 2D heatmap generated from face crop.
        face_bbox: (x1, y1, x2, y2) face bounding box in original image coords.
        alpha: Blend factor.
        size: Output canvas size.

    Returns:
        PIL Image with heatmap overlay only on face region.
    """
    img = np.array(pil_image.convert("RGB").resize(size))
    orig_w, orig_h = pil_image.size
    scale_x = size[0] / orig_w
    scale_y = size[1] / orig_h

    x1, y1, x2, y2 = face_bbox
    sx1 = int(x1 * scale_x)
    sy1 = int(y1 * scale_y)
    sx2 = int(x2 * scale_x)
    sy2 = int(y2 * scale_y)

    # Clamp
    sx1, sy1 = max(0, sx1), max(0, sy1)
    sx2, sy2 = min(size[0], sx2), min(size[1], sy2)

    face_w = sx2 - sx1
    face_h = sy2 - sy1
    if face_w < 5 or face_h < 5:
        return create_heatmap_overlay(pil_image, heatmap, alpha, size)

    # Resize heatmap to face region size
    heatmap_resized = cv2.resize(heatmap, (face_w, face_h))
    heatmap_resized = np.power(heatmap_resized, 0.7)

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_HOT
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay only on face region
    canvas = img.copy()
    face_region = canvas[sy1:sy2, sx1:sx2]
    blended = np.uint8(alpha * heatmap_colored + (1 - alpha) * face_region)
    canvas[sy1:sy2, sx1:sx2] = blended

    return Image.fromarray(canvas)


def merge_heatmaps(heatmaps, weights=None):
    """
    Merge multiple heatmaps with optional weights.

    Args:
        heatmaps: list of 2D numpy arrays (can contain None).
        weights: list of floats (same length as heatmaps).

    Returns:
        Merged heatmap normalized to [0, 1].
    """
    valid = [(h, w) for h, w in zip(heatmaps, weights or [1.0]*len(heatmaps))
             if h is not None]

    if not valid:
        return np.zeros((7, 7))

    # Resize all to common size
    target_size = (224, 224)
    merged = np.zeros(target_size, dtype=np.float32)
    total_weight = 0.0

    for h, w in valid:
        resized = cv2.resize(h.astype(np.float32), target_size)
        merged += w * resized
        total_weight += w

    if total_weight > 0:
        merged /= total_weight

    if merged.max() > 0:
        merged = merged / merged.max()

    return merged


# ================================================================
#  Side-by-side comparison
# ================================================================

def create_side_by_side(pil_image, heatmap, face_bbox=None,
                        label_left="Input", label_right="Detection Output"):
    """
    Create a side-by-side comparison: original | heatmap overlay.
    If face_bbox is provided, overlay only on the face region.
    """
    size = (400, 400)
    original = np.array(pil_image.convert("RGB").resize(size))

    if face_bbox is not None:
        overlay_pil = create_face_region_overlay(pil_image, heatmap, face_bbox, alpha=0.5, size=size)
    else:
        overlay_pil = create_heatmap_overlay(pil_image, heatmap, alpha=0.5, size=size)
    overlay = np.array(overlay_pil)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    text_color = (0, 0, 0)

    label_height = 40
    gap = 6
    canvas_h = size[1] + label_height
    canvas_w = size[0] * 2 + gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    left_size = cv2.getTextSize(label_left, font, font_scale, thickness)[0]
    right_size = cv2.getTextSize(label_right, font, font_scale, thickness)[0]
    cv2.putText(canvas, label_left,
                (size[0] // 2 - left_size[0] // 2, 28),
                font, font_scale, text_color, thickness)
    cv2.putText(canvas, label_right,
                (size[0] + gap + size[0] // 2 - right_size[0] // 2, 28),
                font, font_scale, text_color, thickness)

    canvas[label_height:, :size[0]] = original
    canvas[label_height:, size[0] + gap:] = overlay

    return Image.fromarray(canvas)


# ================================================================
#  Main entry point
# ================================================================

def generate_gradcam_image(pil_image, face_model, device,
                           vit_model=None, vit_processor=None,
                           eff_model=None, dino_model=None):
    """
    Full pipeline: face-aligned Grad-CAM from all available models.

    1. Detect and crop face
    2. Run Grad-CAM on face crop for each model
    3. Merge heatmaps with model weights
    4. Overlay only on face region

    Returns:
        PIL Image (side-by-side) or None.
    """
    # Step 1: Face detection and alignment
    face_crop, face_bbox = detect_and_align_face(pil_image)
    use_face_crop = face_crop is not None

    # Image to use for model input
    model_input = face_crop if use_face_crop else pil_image

    heatmaps = []
    weights = []
    model_names = []

    # Step 2: Grad-CAM from each model on the face crop
    if face_model is not None:
        try:
            tensor = _preprocess(model_input.convert("RGB")).unsqueeze(0).to(device)
            hm = get_gradcam_for_face_model(face_model, tensor)
            heatmaps.append(hm)
            weights.append(0.3)
            model_names.append("Face")
        except Exception as e:
            print(f"Face GradCAM failed: {e}")

    if vit_model is not None and vit_processor is not None:
        hm = get_gradcam_for_vit(vit_model, vit_processor, model_input, device)
        if hm is not None:
            heatmaps.append(hm)
            weights.append(0.4)
            model_names.append("ViT")

    if eff_model is not None:
        try:
            tensor = _preprocess(model_input.convert("RGB")).unsqueeze(0).to(device)
            hm = get_gradcam_for_efficientnet(eff_model, tensor)
            if hm is not None:
                heatmaps.append(hm)
                weights.append(0.2)
                model_names.append("Eff")
        except Exception as e:
            print(f"EfficientNet GradCAM failed: {e}")

    if dino_model is not None:
        try:
            tensor = _preprocess(model_input.convert("RGB")).unsqueeze(0).to(device)
            hm = get_gradcam_for_dino(dino_model, tensor)
            if hm is not None:
                heatmaps.append(hm)
                weights.append(0.1)
                model_names.append("DINO")
        except Exception as e:
            print(f"DINO GradCAM failed: {e}")

    # Step 3: Merge heatmaps
    if not heatmaps:
        # Fallback: uniform low-confidence heatmap
        merged = np.ones((224, 224)) * 0.2
        model_label = "No Model"
    else:
        merged = merge_heatmaps(heatmaps, weights)
        model_label = "+".join(model_names)

    # Step 4: Overlay on face region only
    return create_side_by_side(
        pil_image, merged, face_bbox=face_bbox,
        label_left="Input Image",
        label_right=f"Artifact Map ({model_label})"
    )


def generate_gradcam_video_frames(video_frames_pil, face_model, device,
                                   vit_model=None, vit_processor=None):
    """
    Generate face-aligned GradCAM overlays for video frames.
    """
    if face_model is None and vit_model is None:
        return []

    results = []
    prev_heatmap = None

    for frame_idx, pil_img, timestamp in video_frames_pil:
        face_crop, face_bbox = detect_and_align_face(pil_img)
        model_input = face_crop if face_crop is not None else pil_img

        heatmap = None

        if face_model is not None:
            try:
                tensor = _preprocess(model_input.convert("RGB")).unsqueeze(0).to(device)
                heatmap = get_gradcam_for_face_model(face_model, tensor)
            except Exception:
                pass

        if heatmap is None and vit_model is not None and vit_processor is not None:
            heatmap = get_gradcam_for_vit(vit_model, vit_processor, model_input, device)

        if heatmap is None:
            heatmap = prev_heatmap if prev_heatmap is not None else np.ones((7, 7)) * 0.2

        # Temporal smoothing: blend with previous frame's heatmap
        if prev_heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            prev_resized = cv2.resize(prev_heatmap, (224, 224))
            heatmap = 0.7 * heatmap_resized + 0.3 * prev_resized
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

        prev_heatmap = heatmap

        # Overlay on face region only
        if face_bbox is not None:
            overlay = create_face_region_overlay(pil_img, heatmap, face_bbox, alpha=0.5, size=(400, 400))
        else:
            overlay = create_heatmap_overlay(pil_img, heatmap, alpha=0.5, size=(400, 400))

        results.append((frame_idx, overlay, timestamp))

    return results
