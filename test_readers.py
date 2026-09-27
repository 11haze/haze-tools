"""Run the readers on every image in screenshots/ and print the results.
Annotated copies are saved to logs/readers_debug/ so you can check them visually.

Usage: python test_readers.py [image ...]
"""
import sys
from pathlib import Path

import cv2

import readers
from capture import load_image
from config import ROOT, load_config


def fmt(v):
    return "  -  " if v is None else f"{v:5.1f}"


def main():
    cfg = load_config()
    paths = [Path(p) for p in sys.argv[1:]] or sorted(
        p for p in (ROOT / "screenshots").iterdir() if p.suffix.lower() in (".png", ".jpg", ".bmp"))
    out_dir = ROOT / "logs" / "readers_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = (cfg["window"]["width"], cfg["window"]["height"])

    print(f"{'image':32} {'HP%':>5} {'MP%':>5} {'target':>6} {'tgtHP%':>6}  bar")
    for path in paths:
        img = load_image(path)
        hp = readers.get_hp_percent(img, cfg)
        mp = readers.get_mp_percent(img, cfg)
        box = readers.find_target_box(img, cfg)
        selected = box is not None
        bar = readers.find_target_hp_bar(img, cfg)
        tgt = bar[4] if (selected and bar) else None
        size_note = "" if (img.shape[1], img.shape[0]) == expected else "  (size mismatch, may be off)"
        print(f"{path.name:32} {fmt(hp)} {fmt(mp)} {'yes' if selected else 'no':>6} {fmt(tgt):>6}"
              f"  box={box or '-'} bar={bar[:4] if bar else '-'}{size_note}")

        view = img.copy()
        for name in ("hp_bar", "mp_bar"):
            rect = cfg["regions"].get(name)
            if rect:
                x, y, w, h = rect
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if box:
            x, y, w, h = box
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if bar:
            x, y, w, h, pct = bar
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.putText(view, f"{pct:.0f}%", (x + w + 4, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(view, f"HP {fmt(hp)}  MP {fmt(mp)}  target {'yes' if selected else 'no'}  tgtHP {fmt(tgt)}",
                    (220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / path.name), view)
    print(f"\nAnnotated images: {out_dir}")


if __name__ == "__main__":
    main()
