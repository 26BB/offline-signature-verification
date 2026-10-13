import cv2
import numpy as np
from PIL import Image

def preprocess_signature(image_input, target_size=(128, 128)):
    """
    Cleans and standardizes an input signature image:
    1. Grayscale conversion
    2. Otsu thresholding + binarization inversion
    3. Tight bounding-box contour crop
    4. Aspect-ratio preserving resize with zero padding
    5. Normalization to [0.0, 1.0]

    Args:
        image_input: File path (str) or PIL Image or numpy array
        target_size: Tuple (height, width), default (128, 128)
    Returns:
        np.ndarray with shape target_size, float32, range [0, 1]
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_input}")
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("L"))
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # 1. Otsu thresholding with inversion (strokes become white/255, background black/0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Morphological noise removal (opening removes small specks)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 3. Find contours to crop tight bounding box
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        # Ensure non-empty crop
        if w > 0 and h > 0:
            cropped = cleaned[y:y+h, x:x+w]
        else:
            cropped = cleaned
    else:
        cropped = cleaned

    # 4. Aspect-ratio preserving resize with centered padding
    h_c, w_c = cropped.shape
    if h_c == 0 or w_c == 0:
        return np.zeros(target_size, dtype=np.float32)

    scale = min(target_size[0] / h_c, target_size[1] / w_c)
    new_h = max(1, int(h_c * scale))
    new_w = max(1, int(w_c * scale))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_h = (target_size[0] - new_h) // 2
    pad_w = (target_size[1] - new_w) // 2

    canvas = np.zeros(target_size, dtype=np.uint8)
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

    # 5. Normalize to [0.0, 1.0]
    normalized = canvas.astype(np.float32) / 255.0
    return normalized
