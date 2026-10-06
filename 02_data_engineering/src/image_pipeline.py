"""
Image Preprocessing and Label Conversion Pipeline for Computer Vision.

This module prepares raw image and annotation data for object detection models (e.g., YOLO).
Key capabilities:
1. Aspect-ratio preserving image resize with letterbox padding.
2. Pixel normalization to [0.0, 1.0].
3. Conversion of Pascal VOC XML annotations (<xmin>, <ymin>, <xmax>, <ymax>) 
   into normalized YOLO format (<class_id> <x_center> <y_center> <width> <height>).
"""

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import numpy as np


def letterbox_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    fill_value: int = 0
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resizes an image using letterboxing (padding) to maintain the original aspect ratio.

    Args:
        image: Input image array of shape (H, W, C) or (H, W).
        target_size: Desired (target_height, target_width).
        fill_value: Pixel value for padding (default 0 for black).

    Returns:
        padded_image: The resized and padded image array.
        scale: Scaling factor applied to the original dimensions.
        (pad_w, pad_h): Padding added to left and top (in pixels).
    """
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size

    # Compute scale to fit inside target bounding box
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    # Nearest neighbor / bilinear simulation for resizing using pure numpy if cv2 unavailable
    # Nearest neighbor coordinate mapping:
    row_indices = (np.linspace(0, orig_h - 1, new_h)).astype(int)
    col_indices = (np.linspace(0, orig_w - 1, new_w)).astype(int)
    resized = image[np.ix_(row_indices, col_indices)]

    # Compute symmetric padding
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    if image.ndim == 3:
        channels = image.shape[2]
        padded_image = np.full((target_h, target_w, channels), fill_value, dtype=image.dtype)
        padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized
    else:
        padded_image = np.full((target_h, target_w), fill_value, dtype=image.dtype)
        padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return padded_image, scale, (pad_w, pad_h)


def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel values from [0, 255] integer range to [0.0, 1.0] floating point.
    """
    return image.astype(np.float32) / 255.0


def voc_box_to_yolo(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int
) -> tuple[float, float, float, float]:
    """
    Converts Pascal VOC bounding box [xmin, ymin, xmax, ymax]
    to normalized YOLO format [x_center, y_center, width, height].

    All YOLO coordinates are normalized to the [0.0, 1.0] range relative to image width/height.
    """
    xmin, ymin, xmax, ymax = box

    # Clip to image boundaries
    xmin = max(0.0, min(float(xmin), float(image_width)))
    xmax = max(0.0, min(float(xmax), float(image_width)))
    ymin = max(0.0, min(float(ymin), float(image_height)))
    ymax = max(0.0, min(float(ymax), float(image_height)))

    box_w = xmax - xmin
    box_h = ymax - ymin
    x_center = xmin + (box_w / 2.0)
    y_center = ymin + (box_h / 2.0)

    # Normalize by image dimensions
    norm_x = x_center / image_width
    norm_y = y_center / image_height
    norm_w = box_w / image_width
    norm_h = box_h / image_height

    return (norm_x, norm_y, norm_w, norm_h)


def parse_voc_xml(xml_path: Path, class_mapping: dict[str, int]) -> list[dict[str, Any]]:
    """
    Parses Pascal VOC XML annotation file and returns a list of bounding boxes with class IDs.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_tag = root.find("size")
    if size_tag is not None:
        width = int(size_tag.findtext("width", "0"))
        height = int(size_tag.findtext("height", "0"))
    else:
        width, height = 0, 0

    annotations = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        if class_name not in class_mapping:
            continue

        class_id = class_mapping[class_name]
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))

            annotations.append({
                "class_id": class_id,
                "class_name": class_name,
                "box": (xmin, ymin, xmax, ymax),
                "image_width": width,
                "image_height": height
            })

    return annotations


def convert_annotations_to_yolo_txt(
    annotations: list[dict[str, Any]],
    output_path: Path
) -> None:
    """
    Writes a list of annotations to a YOLO-compatible text file.
    Each line contains: <class_id> <x_center> <y_center> <width> <height>
    """
    lines = []
    for ann in annotations:
        class_id = ann["class_id"]
        box = ann["box"]
        img_w = ann["image_width"]
        img_h = ann["image_height"]

        if img_w <= 0 or img_h <= 0:
            continue

        yolo_box = voc_box_to_yolo(box, img_w, img_h)
        lines.append(f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
