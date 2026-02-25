"""Draw bounding-box annotations on images for visual debugging."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from app.schemas import Detection

# Distinct colors per class (extend as needed)
_PALETTE: dict[str, str] = {
    "black_spot": "#FF4136",
    "downy_mildew": "#0074D9",
}
_DEFAULT_COLOR = "#2ECC40"


def annotate_image(
    image: Image.Image,
    detections: list[Detection],
) -> Image.Image:
    """Return a copy of *image* with bounding boxes and labels drawn."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for det in detections:
        color = _PALETTE.get(det.label, _DEFAULT_COLOR)
        box = (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2)
        draw.rectangle(box, outline=color, width=3)

        label_text = f"{det.label} {det.confidence:.2f}"
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Background rectangle for readability
        bg_x = det.bbox.x1
        bg_y = max(det.bbox.y1 - text_h - 6, 0)
        draw.rectangle(
            (bg_x, bg_y, bg_x + text_w + 6, bg_y + text_h + 6),
            fill=color,
        )
        draw.text((bg_x + 3, bg_y + 2), label_text, fill="white", font=font)

    return img
