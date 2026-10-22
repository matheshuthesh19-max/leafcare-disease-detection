
"""ML-based leaf disease detector with leaf-image validation."""

import os
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_SIZE = (224, 224)

DISEASE_MODEL_PATH = BASE_DIR / "model.keras"
DISEASE_LABELS_PATH = BASE_DIR / "class_names.json"
LEAF_MODEL_PATH = BASE_DIR / "leaf_validator.keras"
LEAF_LABELS_PATH = BASE_DIR / "leaf_classes.json"

# Starting thresholds only. Validate with real test images.
LEAF_THRESHOLD = 0.75
DISEASE_THRESHOLD = 0.55

DISEASE_LIBRARY = {
    "Healthy": {
        "name": "ஆரோக்கியமான இலை (Healthy Leaf)",
        "advice": "வழக்கமான நீர்ப்பாசனம் மற்றும் பராமரிப்பைத் தொடரவும். "
                  "இலைகளைத் தொடர்ந்து பரிசோதிக்கவும்."
    },
    "Leaf_Spot": {
        "name": "இலைப் புள்ளி நோய் (Leaf Spot Disease)",
        "advice": "பாதிக்கப்பட்ட இலைகளைப் பரிசோதிக்கவும். இலைகள் ஈரமாக "
                  "இருப்பதைத் தவிர்த்து, சரியான நோய் மேலாண்மை முறையைப் "
                  "பயிர் நிபுணரிடம் உறுதிப்படுத்தவும்."
    },
    "Rust": {
        "name": "இலை துரு நோய் (Leaf Rust)",
        "advice": "துரு போன்ற அறிகுறிகள் உள்ள இலைகளைப் பரிசோதிக்கவும். "
                  "நோயை உறுதிப்படுத்திய பின்பே பொருத்தமான சிகிச்சையைத் "
                  "தேர்வு செய்யவும்."
    },
    "Blight": {
        "name": "இலைக் கருகல் நோய் (Leaf Blight)",
        "advice": "பாதிக்கப்பட்ட இலைகளைத் தொடர்ந்து கண்காணிக்கவும். "
                  "நோயை உறுதிப்படுத்தி, பயிருக்கு ஏற்ற மேலாண்மை முறையை "
                  "பின்பற்றவும்."
    },
    "Powdery_Mildew": {
        "name": "மாவுப்பூஞ்சை நோய் (Powdery Mildew)",
        "advice": "வெள்ளை மாவு போன்ற படலம் உள்ளதா எனப் பரிசோதிக்கவும். "
                  "நோயை உறுதிப்படுத்திய பிறகு சரியான சிகிச்சையைத் தேர்வு "
                  "செய்யவும்."
    },
    "Downy_Mildew": {
        "name": "டவுனி மில்டியூ நோய் (Downy Mildew)",
        "advice": "இலைகளில் மஞ்சள் புள்ளிகள் அல்லது அடிப்புறப் பூஞ்சை "
                  "அறிகுறிகள் உள்ளதா எனப் பரிசோதிக்கவும்."
    },
    "Mosaic_Virus": {
        "name": "மொசைக் வைரஸ் அறிகுறி (Mosaic Virus)",
        "advice": "இலைகளில் ஒழுங்கற்ற வெளிர் மற்றும் அடர் நிற வடிவங்கள் "
                  "உள்ளதா எனப் பரிசோதிக்கவும். நிபுணரிடம் உறுதிப்படுத்தவும்."
    },
    "Anthracnose": {
        "name": "ஆந்த்ராக்னோஸ் நோய் (Anthracnose)",
        "advice": "கருமையான புள்ளிகள் உள்ளதா எனப் பரிசோதிக்கவும். "
                  "படத்தின் அடிப்படையில் மட்டும் நோயை உறுதி செய்ய முடியாது."
    },
    "Septoria_Leaf_Spot": {
        "name": "செப்டோரியா இலைப் புள்ளி நோய் (Septoria Leaf Spot)",
        "advice": "இலைப் புள்ளிகளின் வடிவத்தையும் பரவலையும் கண்காணிக்கவும். "
                  "சரியான நோய் அடையாளத்தை நிபுணரிடம் உறுதிப்படுத்தவும்."
    },
    "Leaf_Mold": {
        "name": "இலைப் பூஞ்சை நோய் (Leaf Mold)",
        "advice": "இலையின் இருபுறமும் பூஞ்சை அறிகுறிகள் உள்ளனவா எனப் "
                  "பரிசோதித்து, பொருத்தமான மேலாண்மை முறையை உறுதிப்படுத்தவும்."
    },
}

_leaf_model = None
_leaf_classes = None
_disease_model = None
_disease_classes = None


def _load_model(model_path, labels_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    model = tf.keras.models.load_model(model_path)

    with open(labels_path, "r", encoding="utf-8") as file:
        classes = json.load(file)

    if len(classes) != model.output_shape[-1]:
        raise ValueError(f"Model and labels mismatch: {model_path.name}")

    return model, classes


def _load_models():
    global _leaf_model, _leaf_classes
    global _disease_model, _disease_classes

    if _leaf_model is None:
        _leaf_model, _leaf_classes = _load_model(
            LEAF_MODEL_PATH, LEAF_LABELS_PATH
        )

        if "Leaf" not in _leaf_classes:
            raise ValueError(
                "leaf_classes.json must contain the exact class name 'Leaf'."
            )

    if _disease_model is None:
        _disease_model, _disease_classes = _load_model(
            DISEASE_MODEL_PATH, DISEASE_LABELS_PATH
        )

    return (
        _leaf_model, _leaf_classes,
        _disease_model, _disease_classes
    )


def _save(path_no_ext, suffix, image):
    filename = f"{os.path.basename(path_no_ext)}_{suffix}.jpg"
    out_path = os.path.join(os.path.dirname(path_no_ext), filename)

    if not cv2.imwrite(out_path, image):
        raise ValueError(f"Could not save image stage: {suffix}")

    return filename


def _invalid_result():
    return {
        "status": "Invalid",
        "disease_name": "❌ இது சரியான இலை படம் அல்ல",
        "advice": "தயவுசெய்து தெளிவான இலைப் படத்தை மட்டும் பதிவேற்றவும்.",
        "severity": "-",
        "diseased_percent": 0,
        "healthy_percent": 0,
        "spot_count": 0,
        "gray_file": "",
        "segmentation_file": "",
        "highlighted_file": "",
        "valid_image": False,
        "confidence": 0.0,
    }


def analyze_leaf(image_path: str) -> dict:
    if not os.path.isfile(image_path):
        return _invalid_result()

    img = cv2.imread(image_path)
    if img is None:
        return _invalid_result()

    try:
        (
            leaf_model, leaf_classes,
            disease_model, disease_classes
        ) = _load_models()
    except (OSError, ValueError, FileNotFoundError):
        # Do not silently return a fake disease prediction.
        raise

    img = cv2.resize(img, (420, 420))
    path_no_ext = os.path.splitext(image_path)[0]

    # OpenCV uses BGR; Keras directory datasets use RGB.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model_image = cv2.resize(rgb, IMG_SIZE).astype(np.float32)
    model_image = np.expand_dims(model_image, axis=0)

    # Stage 1: validate leaf vs non-leaf.
    leaf_probs = leaf_model.predict(model_image, verbose=0)[0]
    leaf_idx = int(np.argmax(leaf_probs))
    leaf_label = leaf_classes[leaf_idx]
    leaf_confidence = float(leaf_probs[leaf_idx])

    if leaf_label != "Leaf" or leaf_confidence < LEAF_THRESHOLD:
        return _invalid_result()

    # Stage 2: classify disease using the trained model.
    probs = disease_model.predict(model_image, verbose=0)[0]
    best_idx = int(np.argmax(probs))
    predicted_class = disease_classes[best_idx]
    confidence = float(probs[best_idx])

    # Build visual stages for the existing results page.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_file = _save(path_no_ext, "gray", gray)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv, np.array([25, 40, 40]), np.array([95, 255, 255])
    )

    # Approximate foreground mask for display only.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if cv2.countNonZero(mask) < (420 * 420) * 0.05:
        mask = np.full(gray.shape, 255, dtype=np.uint8)

    segmentation_file = _save(path_no_ext, "segmented", mask)

    # Approximate discoloured pixels. This is NOT a disease measurement.
    yellow_brown = cv2.inRange(
        hsv, np.array([8, 40, 40]), np.array([30, 255, 220])
    )
    dark = cv2.inRange(
        hsv, np.array([0, 0, 0]), np.array([180, 255, 65])
    )

    disease_mask = cv2.bitwise_or(yellow_brown, dark)
    disease_mask = cv2.bitwise_and(disease_mask, mask)
    disease_mask = cv2.morphologyEx(
        disease_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    leaf_area = max(cv2.countNonZero(mask), 1)
    diseased_area = cv2.countNonZero(disease_mask)
    diseased_percent = round(
        min(diseased_area / leaf_area * 100, 100), 2
    )
    healthy_percent = round(100 - diseased_percent, 2)

    contours, _ = cv2.findContours(
        disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > 15]

    highlighted = img.copy()
    cv2.drawContours(highlighted, contours, -1, (0, 0, 255), 2)
    highlighted_file = _save(path_no_ext, "highlighted", highlighted)

    if confidence < DISEASE_THRESHOLD:
        return {
            "status": "Uncertain",
            "disease_name": "நோயை உறுதியாகக் கண்டறிய முடியவில்லை",
            "advice": "தெளிவான இலைப் படத்தை மீண்டும் பதிவேற்றவும். "
                      "தேவைப்பட்டால் நிபுணரிடம் உறுதிப்படுத்தவும்.",
            "severity": "உறுதிப்படுத்தப்படவில்லை",
            "diseased_percent": diseased_percent,
            "healthy_percent": healthy_percent,
            "spot_count": len(contours),
            "gray_file": gray_file,
            "segmentation_file": segmentation_file,
            "highlighted_file": highlighted_file,
            "valid_image": True,
            "confidence": confidence,
        }

    info = DISEASE_LIBRARY.get(predicted_class)

    if info is None:
        raise ValueError(
            f"Unknown disease class in class_names.json: {predicted_class}"
        )

    if predicted_class == "Healthy":
        status = "Healthy"
        severity = "இல்லை (None)"
    else:
        status = "Diseased"
        if diseased_percent < 12:
            severity = "லேசானது (Mild)"
        elif diseased_percent < 25:
            severity = "மிதமானது (Moderate)"
        else:
            severity = "அதிகமானது (High)"

    return {
        "status": status,
        "disease_name": info["name"],
        "advice": info["advice"],
        "severity": severity,
        "diseased_percent": diseased_percent,
        "healthy_percent": healthy_percent,
        "spot_count": len(contours),
        "gray_file": gray_file,
        "segmentation_file": segmentation_file,
        "highlighted_file": highlighted_file,
        "valid_image": True,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2),
        "predicted_class": predicted_class,
    }
