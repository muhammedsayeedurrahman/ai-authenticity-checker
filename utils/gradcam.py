"""
GradCAM visualization for deepfake detection.

Generates heatmaps showing which regions of the image the model
focuses on when making its prediction, and creates side-by-side
comparisons (original | heatmap overlay).
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

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """
        Generate GradCAM heatmap for the given input tensor.

        Args:
            input_tensor: [1, 3, H, W] preprocessed image tensor.

        Returns:
            heatmap: numpy array [H, W] normalized to [0, 1].
        """
        self.model.eval()

        # Enable gradients temporarily
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        # Backward pass on the output score
        self.model.zero_grad()
        output.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = torch.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def cleanup(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


def get_gradcam_for_face_model(face_model, input_tensor):
    """
    Generate GradCAM using the FaceDeepfakeModel's last conv layer (layer4).
    """
    # FaceDeepfakeModel.features is nn.Sequential(*list(resnet.children())[:-1])
    # layer4 is at index 7 in this sequential
    target_layer = face_model.features[7]  # ResNet50 layer4

    cam = GradCAM(face_model, target_layer)
    try:
        heatmap = cam.generate(input_tensor)
    finally:
        cam.cleanup()

    return heatmap


def get_gradcam_for_dino_model(dino_model, input_tensor):
    """
    Generate GradCAM using DINOv2AuthModel's feature extractor.
    DINOv2 is transformer-based, so we target the last norm layer.
    """
    try:
        target_layer = dino_model.backbone.encoder.layer[-1].norm1
    except (AttributeError, IndexError):
        return None

    cam = GradCAM(dino_model, target_layer)
    try:
        heatmap = cam.generate(input_tensor)
    finally:
        cam.cleanup()

    return heatmap


_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def create_heatmap_overlay(pil_image, heatmap, alpha=0.5):
    """
    Overlay a GradCAM heatmap on the original image.

    Args:
        pil_image: Original PIL image.
        heatmap: 2D numpy array [h, w] normalized to [0, 1].
        alpha: Blend factor (0=original, 1=full heatmap).

    Returns:
        PIL Image with heatmap overlay.
    """
    img = np.array(pil_image.convert("RGB").resize((224, 224)))

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Apply colormap (jet: blue=low, red=high)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * img)

    return Image.fromarray(overlay)


def create_side_by_side(pil_image, heatmap, label_left="Input", label_right="Detection Output"):
    """
    Create a side-by-side comparison: original | heatmap overlay.
    Style: white background, black text labels (VeridisQuo-inspired).

    Args:
        pil_image: Original PIL image.
        heatmap: 2D numpy array from GradCAM.
        label_left: Label for left image.
        label_right: Label for right image.

    Returns:
        PIL Image with side-by-side comparison.
    """
    size = (400, 400)
    original = np.array(pil_image.convert("RGB").resize(size))
    overlay = np.array(create_heatmap_overlay(pil_image, heatmap, alpha=0.45).resize(size))

    # Style settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    text_color = (0, 0, 0)  # Black text
    bg_color = (255, 255, 255)  # White background

    # Create canvas with space for labels
    label_height = 40
    gap = 6
    canvas_h = size[1] + label_height
    canvas_w = size[0] * 2 + gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)  # White bg

    # Draw labels (centered above each image)
    left_size = cv2.getTextSize(label_left, font, font_scale, thickness)[0]
    right_size = cv2.getTextSize(label_right, font, font_scale, thickness)[0]
    cv2.putText(canvas, label_left,
                (size[0] // 2 - left_size[0] // 2, 28),
                font, font_scale, text_color, thickness)
    cv2.putText(canvas, label_right,
                (size[0] + gap + size[0] // 2 - right_size[0] // 2, 28),
                font, font_scale, text_color, thickness)

    # Place images
    canvas[label_height:, :size[0]] = original
    canvas[label_height:, size[0] + gap:] = overlay

    return Image.fromarray(canvas)


def generate_gradcam_image(pil_image, face_model, device):
    """
    Full pipeline: take a PIL image, run GradCAM on face model,
    return side-by-side comparison.

    Args:
        pil_image: Input PIL image.
        face_model: Loaded FaceDeepfakeModel.
        device: torch device.

    Returns:
        PIL Image (side-by-side) or None if model unavailable.
    """
    if face_model is None:
        return None

    tensor = _preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)
    heatmap = get_gradcam_for_face_model(face_model, tensor)

    return create_side_by_side(pil_image, heatmap,
                               label_left="Manipulated Video",
                               label_right="AI Auth Checker output")


def generate_gradcam_video_frames(video_frames_pil, face_model, device):
    """
    Generate GradCAM overlays for a list of video frames.

    Args:
        video_frames_pil: List of (frame_index, PIL.Image, timestamp).
        face_model: Loaded FaceDeepfakeModel.
        device: torch device.

    Returns:
        List of (frame_index, PIL overlay image, timestamp).
    """
    if face_model is None:
        return []

    results = []
    for frame_idx, pil_img, timestamp in video_frames_pil:
        tensor = _preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)
        heatmap = get_gradcam_for_face_model(face_model, tensor)
        overlay = create_heatmap_overlay(pil_img, heatmap)
        results.append((frame_idx, overlay, timestamp))

    return results
