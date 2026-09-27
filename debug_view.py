"""Run the nameplate detector on every image in screenshots/ and save annotated
copies to logs/debug_view/.

  green   target (whitelisted) + click point
  yellow  read by OCR but not in the whitelist (e.g. player names)
  cyan    passed the box checks (only with --no-ocr)
  gray    rejected: self / zone / no_box / unread / click_in_zone

Usage: python debug_view.py [--no-ocr] [image ...]
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2

from capture import load_image
from config import ROOT, load_config
from detector import find_targets

COLORS = {
    "target": (0, 255, 0),
    "not_whitelisted": (0, 220, 255),
    "candidate": (255, 255, 0),
}
GRAY = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(img, cfg, targets, labels):
    view = img.copy()
    x, y, w, h = cfg["detector"]["self_exclusion"]
    cv2.rectangle(view, (x, y), (x + w, y + h), (255, 0, 255), 1)
    for label in labels:
        x, y, w, h = label.box
        color = COLORS.get(label.status, GRAY)
        cv2.rectangle(view, (x - 2, y - 2), (x + w + 2, y + h + 2), color, 1)
        caption = f"{label.text} {label.score:.2f}" if label.text else label.status
        if label.status not in COLORS and label.text:
            caption += f" [{label.status}]"
        cv2.putText(view, caption, (x, y + h + 13), FONT, 0.38, color, 1, cv2.LINE_AA)
    for i, t in enumerate(targets):
        cx, cy = t.click
        cv2.drawMarker(view, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
        cv2.line(view, (t.label[0] + t.label[2] // 2, t.label[1] + t.label[3]), (cx, cy), (0, 255, 0), 1)
        cv2.putText(view, f"#{i + 1}", (cx + 8, cy + 4), FONT, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--no-ocr", action="store_true", help="test box detection only")
    args = ap.parse_args()

    cfg = load_config()
    paths = [Path(p) for p in args.images] or sorted(
        p for p in (ROOT / "screenshots").iterdir() if p.suffix.lower() in (".png", ".jpg", ".bmp"))
    out_dir = ROOT / "logs" / "debug_view"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        img = load_image(path)
        targets, labels = find_targets(img, cfg, return_labels=True, use_ocr=not args.no_ocr)
        counts = Counter(l.status for l in labels)
        print(f"\n{path.name}: {len(targets)} target(s)  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        for i, t in enumerate(targets):
            print(f"  #{i + 1} {t.text!r} ({t.score:.2f}) label={t.label} click={t.click} dist={t.distance:.0f}")
        for l in labels:
            if l.status == "not_whitelisted" or (l.status == "candidate"):
                print(f"  - {l.status}: {l.text!r} {l.box}")
        cv2.imwrite(str(out_dir / path.name), annotate(img, cfg, targets, labels))
    print(f"\nAnnotated images: {out_dir}")


if __name__ == "__main__":
    main()
