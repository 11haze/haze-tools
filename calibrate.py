"""Calibration tool (Prompt 2).

Drag rectangles for the HP/MP bars, target frame, target HP bar and no-click
zones, and click pixels to sample bar colors (saved as HSV ranges).
Results are written into config.yaml (regions, no_click_zones, colors).

Usage:
  python calibrate.py                  # cycle through images in screenshots/
  python calibrate.py path\\to\\img.png
  python calibrate.py --live           # capture the game now (also saved to screenshots/)

Coordinates only line up with the bot if the image is a full client-area
capture (same size as window.width x window.height), so --live is the most
reliable source.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import ROOT, load_config, update_sections

REGION_ITEMS = ["hp_bar", "mp_bar", "target_frame_search"]
COLOR_ITEMS = ["hp_fill", "hp_empty", "mp_fill", "mp_empty", "target_hp_fill", "target_hp_empty"]
PAD_H, PAD_S, PAD_V = 8, 40, 40   # tolerance added around sampled HSV values
PATCH = 2                          # each click samples a 5x5 pixel patch
PANEL_H = 84
WIN = "calibrate"
FONT = cv2.FONT_HERSHEY_SIMPLEX

HELP = """Controls
  [ / ]      previous / next item (bars, zones, colors)
  , / .      previous / next image
  drag       set the rectangle for a region or zone item
  click      add a color sample for a color item (click several spots)
  n          add a new no-click zone (type its name in the console)
  d          delete / clear the current item
  s          save to config.yaml
  q / Esc    quit (press twice if there are unsaved changes)"""


def hsv_range(pixels):
    """Min/max of sampled HSV pixels plus tolerance. OpenCV hue runs 0-179 and
    wraps at red, so h_min > h_max means the range wraps around 0."""
    h = pixels[:, 0].astype(int)
    s = pixels[:, 1].astype(int)
    v = pixels[:, 2].astype(int)
    if h.max() - h.min() > 90:     # samples straddle the red wrap point
        h = np.where(h < 90, h + 180, h)
    if h.max() - h.min() + 2 * PAD_H >= 179:
        h_rng = [0, 179]
    else:
        h_rng = [int((h.min() - PAD_H) % 180), int((h.max() + PAD_H) % 180)]
    return {
        "h": h_rng,
        "s": [int(max(s.min() - PAD_S, 0)), int(min(s.max() + PAD_S, 255))],
        "v": [int(max(v.min() - PAD_V, 0)), int(min(v.max() + PAD_V, 255))],
    }


def put_label(img, text, org, color, scale=0.45):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y - th - 3), (x + tw + 4, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 1), FONT, scale, color, 1, cv2.LINE_AA)


def live_capture(cfg):
    from capture import GameCapture
    cap = GameCapture(cfg)
    cap.activate()
    time.sleep(0.5)
    cap.refresh_rect()
    out = ROOT / "screenshots" / f"calib_{datetime.now():%Y%m%d_%H%M%S}.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), cap.grab())
    print(f"Captured {out}")
    return out


class Calibrator:
    def __init__(self, cfg, images):
        self.cfg = cfg
        self.images = images
        self.expected = (cfg["window"]["width"], cfg["window"]["height"])
        self.regions = dict(cfg.get("regions") or {})
        self.zones = dict(cfg.get("no_click_zones") or {})
        self.colors = dict(cfg.get("colors") or {})
        self.samples = {}        # color item -> list of HSV pixel arrays
        self.sample_pts = {}     # color item -> list of (image index, x, y)
        self.item_idx = 0
        self.img_idx = 0
        self.drag_start = self.drag_end = None
        self.mouse = (0, 0)
        self.dirty = False
        self.quit_armed = False
        self.message = ""
        self.load_image()

    @property
    def items(self):
        return ([("region", n) for n in REGION_ITEMS]
                + [("zone", n) for n in self.zones]
                + [("color", n) for n in COLOR_ITEMS])

    @property
    def current(self):
        self.item_idx %= len(self.items)
        return self.items[self.item_idx]

    def load_image(self):
        path = self.images[self.img_idx]
        self.img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if self.img is None:
            raise FileNotFoundError(path)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, w = self.img.shape[:2]
        if (w, h) != self.expected:
            self.message = (f"WARNING: image is {w}x{h}, game is {self.expected[0]}x{self.expected[1]} "
                            f"- coordinates may be off. Prefer --live.")
        else:
            self.message = ""

    # ---- input -------------------------------------------------------------

    def on_mouse(self, event, x, y, flags, _):
        h, w = self.img.shape[:2]
        x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
        self.mouse = (x, y)
        kind, name = self.current

        if kind == "color":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.add_sample(name, x, y)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x0, y0 = self.drag_start
            self.drag_start = None
            rect = [min(x0, x), min(y0, y), abs(x - x0), abs(y - y0)]
            if rect[2] < 3 or rect[3] < 3:
                self.message = "Rectangle too small - drag a larger area."
                return
            (self.regions if kind == "region" else self.zones)[name] = rect
            self.dirty = True
            self.message = f"{name} = {rect}"

    def add_sample(self, name, x, y):
        patch = self.hsv[max(y - PATCH, 0):y + PATCH + 1, max(x - PATCH, 0):x + PATCH + 1]
        self.samples.setdefault(name, []).append(patch.reshape(-1, 3))
        self.sample_pts.setdefault(name, []).append((self.img_idx, x, y))
        self.colors[name] = hsv_range(np.vstack(self.samples[name]))
        self.dirty = True
        c = self.colors[name]
        self.message = f"{name}: {len(self.samples[name])} sample(s) -> H{c['h']} S{c['s']} V{c['v']}"

    def new_zone(self):
        self.message = "Type the new zone name in the console..."
        cv2.imshow(WIN, self.render())
        cv2.waitKey(1)
        name = input("New no-click zone name: ").strip().replace(" ", "_")
        if not name:
            self.message = "Cancelled."
        elif name in self.zones:
            self.message = f"Zone {name!r} already exists."
        else:
            self.zones[name] = None
            self.item_idx = self.items.index(("zone", name))
            self.message = f"Drag a rectangle for {name}."

    def clear_current(self):
        kind, name = self.current
        if kind == "zone":
            self.zones.pop(name, None)
        elif kind == "region":
            self.regions.pop(name, None)
        else:
            self.colors.pop(name, None)
            self.samples.pop(name, None)
            self.sample_pts.pop(name, None)
        self.dirty = True
        self.message = f"Cleared {name}."

    def save(self):
        update_sections({
            "regions": self.regions,
            "no_click_zones": {k: v for k, v in self.zones.items() if v},
            "colors": self.colors,
        })
        self.dirty = False
        self.message = "Saved to config.yaml."

    # ---- drawing -----------------------------------------------------------

    def render(self):
        h, w = self.img.shape[:2]
        kind, name = self.current
        view = self.img.copy()

        shade = view.copy()
        for rect in self.zones.values():
            if rect:
                x, y, rw, rh = rect
                cv2.rectangle(shade, (x, y), (x + rw, y + rh), (0, 0, 200), -1)
        view = cv2.addWeighted(shade, 0.25, view, 0.75, 0)

        for group, color in ((self.zones, (60, 60, 255)), (self.regions, (60, 220, 60))):
            for n, rect in group.items():
                if not rect:
                    continue
                x, y, rw, rh = rect
                active = n == name and kind != "color"
                cv2.rectangle(view, (x, y), (x + rw, y + rh),
                              (0, 255, 255) if active else color, 2 if active else 1)
                put_label(view, n, (x, max(y - 2, 12)), (0, 255, 255) if active else color)

        if self.drag_start and self.drag_end:
            cv2.rectangle(view, self.drag_start, self.drag_end, (0, 255, 255), 1)

        if kind == "color":
            for idx, x, y in self.sample_pts.get(name, []):
                if idx == self.img_idx:
                    cv2.circle(view, (x, y), 4, (0, 255, 255), 1)

        canvas = np.zeros((h + PANEL_H, w, 3), np.uint8)
        canvas[:h] = view
        mx, my = self.mouse
        hsv = self.hsv[my, mx]
        hint = "click pixels to sample" if kind == "color" else "drag a rectangle"
        done = name in (self.colors if kind == "color" else self.regions if kind == "region" else
                        {k: v for k, v in self.zones.items() if v})
        lines = [
            (f"[{self.item_idx + 1}/{len(self.items)}] {kind.upper()}: {name}"
             f"  ({'set' if done else 'not set'})  - {hint}", (0, 255, 255)),
            (f"image {self.img_idx + 1}/{len(self.images)}: {self.images[self.img_idx].name}"
             f"   cursor ({mx},{my})  HSV ({hsv[0]},{hsv[1]},{hsv[2]})"
             f"{'   * UNSAVED *' if self.dirty else ''}", (220, 220, 220)),
            ("[ ] item   , . image   n new zone   d clear   s save   q quit", (160, 160, 160)),
            (self.message, (120, 200, 255)),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(canvas, text, (8, h + 18 + i * 19), FONT, 0.45, color, 1, cv2.LINE_AA)
        return canvas

    # ---- main loop ---------------------------------------------------------

    def run(self):
        print(HELP)
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        while True:
            cv2.imshow(WIN, self.render())
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 255:
                continue
            if key in (ord("q"), 27):
                if not self.dirty or self.quit_armed:
                    break
                self.quit_armed = True
                self.message = "Unsaved changes! Press q again to quit without saving, or s to save."
                continue
            self.quit_armed = False
            if key in (ord("]"), 9):
                self.item_idx = (self.item_idx + 1) % len(self.items)
                self.message = ""
            elif key == ord("["):
                self.item_idx = (self.item_idx - 1) % len(self.items)
                self.message = ""
            elif key in (ord("."), ord(",")):
                step = 1 if key == ord(".") else -1
                self.img_idx = (self.img_idx + step) % len(self.images)
                self.load_image()
            elif key == ord("n"):
                self.new_zone()
            elif key == ord("d"):
                self.clear_current()
            elif key == ord("s"):
                self.save()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Define screen regions and bar colors.")
    ap.add_argument("image", nargs="?", help="image file (default: all images in screenshots/)")
    ap.add_argument("--live", action="store_true", help="capture the game window first")
    args = ap.parse_args()
    cfg = load_config()

    if args.live:
        images = [live_capture(cfg)]
    elif args.image:
        images = [Path(args.image)]
    else:
        folder = ROOT / "screenshots"
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if not images:
        sys.exit("No images found. Put screenshots in screenshots/ or use --live.")

    Calibrator(cfg, images).run()


if __name__ == "__main__":
    main()
