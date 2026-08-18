import cv2
import os
import urllib.request

# Paths for DNN face detector model files
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "face_detector")
_PROTOTXT = os.path.join(_MODEL_DIR, "deploy.prototxt")
_CAFFEMODEL = os.path.join(_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def _ensure_model_files():
    """Download DNN face detector model files if not present."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if not os.path.exists(_PROTOTXT):
        print("Downloading DNN face detector prototxt...")
        urllib.request.urlretrieve(_PROTOTXT_URL, _PROTOTXT)
    if not os.path.exists(_CAFFEMODEL):
        print("Downloading DNN face detector caffemodel...")
        urllib.request.urlretrieve(_CAFFEMODEL_URL, _CAFFEMODEL)


_net_cache = None


def get_face_net():
    """Return the cached OpenCV DNN face detector, loading it once.

    Parsing the Caffe model from disk is expensive - reloading it per
    call (the previous behavior here, in video_analyzer.py's
    FaceExtractor, and in utils/gradcam.py's detect_and_align_face) is
    what made per-frame video analysis and training-time face alignment
    dramatically slower than necessary.
    """
    global _net_cache
    if _net_cache is None:
        _ensure_model_files()
        _net_cache = cv2.dnn.readNetFromCaffe(_PROTOTXT, _CAFFEMODEL)
    return _net_cache


def face_present(image_path, confidence_threshold=0.5):
    """Detect faces using OpenCV DNN (much more accurate than Haar cascade)."""
    net = get_face_net()

    img = cv2.imread(image_path)
    if img is None:
        return False

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            return True

    return False
