"""Find mobs by their nameplates (Prompt 4).

1. Find bright text of any color that sits on a small dark label box
   (high contrast against the box - no hue filter, since mob name color
   changes with level difference).
2. Drop labels over your own character and inside no-click zones.
3. OCR each label and keep only names in targets.whitelist (config.yaml).
   Player names can also be white on a dark box, so this is the real filter.
4. Return targets sorted by distance from screen center, each with a click
   point below the label (the mob's body), skipping click points in no-click zones.
"""
import difflib
import re
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np

# Text strokes: bright pixels much brighter than their darkest neighbor.
TEXT_MIN_V = 150
TEXT_MIN_CONTRAST = 100
TEXT_HEIGHT = (7, 20)
TEXT_WIDTH = (10, 200)
TEXT_MAX_FILL = 0.75          # solid blobs (HP bar fill, bright UI) are not text
WORD_GAP = 8                  # max px between pieces of one name
# Label box: the pixels around the text are darker than normal ground. The box is
# translucent, so on bright maps (MP_Campus pavement) it's gray (~V 60-90), not black.
# OCR + whitelist is the real filter; this check just skips obvious non-labels.
BOX_PAD = 3
BOX_MAX_MEDIAN_V = 100


@dataclass
class Label:
    box: tuple                # x, y, w, h of the text
    text: str = ""
    score: float = 0.0
    status: str = ""          # target / not_whitelisted / self / zone / no_box / click_in_zone / unread


@dataclass
class Target:
    name: str                 # whitelist entry it matched
    text: str                 # raw OCR text
    score: float
    label: tuple              # x, y, w, h
    click: tuple              # x, y client coordinates
    distance: float = field(default=0.0)


# ---- OCR ----------------------------------------------------------------------

_ocr_engine = None


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise RuntimeError("OCR engine missing. Install it with:\n"
                               "  .venv\\Scripts\\python.exe -m pip install rapidocr-onnxruntime") from e
        _ocr_engine = RapidOCR()
    return _ocr_engine


def warm_up():
    """Load the OCR model now, so the first real scan isn't delayed by it."""
    blank = np.full((40, 120, 3), 255, np.uint8)
    _get_ocr()(blank, use_det=False, use_cls=False, use_rec=True)


def relocate(frame, target, cfg, radius=90):
    """Find where a target's nameplate is now (mobs walk between scan and click).
    Uses only the fast box search - no OCR - and matches a label of nearly the same
    size near the old position, so it won't jump to item drop labels.
    Returns an updated Target, or None if the nameplate isn't visible."""
    x, y, w, h = target.label
    ox, oy = x + w / 2, y + h / 2
    pad = radius + 10                     # search only around the last position (~2ms)
    region = (x - pad, y - pad, w + 2 * pad, h + 2 * pad)
    near = []
    for (bx, by, bw, bh), on_box in find_label_boxes(frame, region):
        if not on_box or abs(bw - w) > 6 or abs(bh - h) > 3:
            continue
        d = float(np.hypot(bx + bw / 2 - ox, by + bh / 2 - oy))
        if d <= radius:
            near.append((d, (bx, by, bw, bh)))
    # Same-size labels of OTHER mobs can be close by, so confirm the name too.
    # OCR results are cached by text pattern, so this is usually ~free.
    wl = cfg.get("targets", {}).get("whitelist", [])
    best = next(((d, b) for d, b in sorted(near) if match_whitelist(ocr_label(frame, b)[0], wl) == target.name),
                None)
    if best is None:
        return None
    bx, by, bw, bh = best[1]
    click = (int(bx + bw / 2), int(by + bh / 2 + cfg.get("detector", {}).get("click_offset_y", 45)))
    if _in_any_zone(*click, cfg):
        return None
    return Target(target.name, target.text, target.score, (bx, by, bw, bh), click, target.distance)


_ocr_cache = OrderedDict()
OCR_CACHE_SIZE = 512


_LEVEL_PREFIX = re.compile(r"^\s*[lL1I][vVyY][.,]?\s*\d+\s*")


def target_box_name(frame, box, cfg):
    """Read the name line of the target info box ("Lv.10 Chief Hooligan").
    Returns (whitelist entry or None, raw OCR text)."""
    x, y, w, h = box
    inner = frame[y:y + h, x:x + w]            # search only inside the box
    lines = [(bx + x, by + y, bw, bh) for (bx, by, bw, bh), _ in find_label_boxes(inner)
             if by + bh <= h / 2 + 4]
    if not lines:
        return None, ""
    top = min(lines, key=lambda b: (b[1], b[0]))
    row = [b for b in lines if abs(b[1] - top[1]) <= 3]
    x0, y0 = min(b[0] for b in row), min(b[1] for b in row)
    x1, y1 = max(b[0] + b[2] for b in row), max(b[1] + b[3] for b in row)
    text, _ = ocr_label(frame, (x0, y0, x1 - x0, y1 - y0))
    name = _LEVEL_PREFIX.sub("", text)
    return match_whitelist(name, cfg.get("targets", {}).get("whitelist", [])), text


def ocr_label(frame, box):
    """Read one label. Returns (text, score).
    Results are cached by the label's text-pixel pattern: nameplates use a pixel
    font, so the same name gives the same pattern frame after frame."""
    x, y, w, h = box
    crop = frame[max(y - 2, 0):y + h + 2, max(x - 2, 0):x + w + 2]
    key = (crop.shape, np.packbits(crop.max(axis=2) >= TEXT_MIN_V).tobytes())
    if key in _ocr_cache:
        _ocr_cache.move_to_end(key)
        return _ocr_cache[key]
    result = _ocr_crop(crop)
    _ocr_cache[key] = result
    if len(_ocr_cache) > OCR_CACHE_SIZE:
        _ocr_cache.popitem(last=False)
    return result


def _ocr_crop(crop):
    # Brightest channel turns any text color into white-on-black; invert to black-on-white.
    gray = 255 - crop.max(axis=2)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    gray = cv2.copyMakeBorder(gray, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    result, _ = _get_ocr()(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), use_det=False, use_cls=False, use_rec=True)
    best = ("", 0.0)
    for item in result or []:
        text = next((v for v in item if isinstance(v, str)), "")
        score = next((float(v) for v in item if isinstance(v, (float, np.floating))), 0.0)
        if score > best[1]:
            best = (text.strip(), score)
    return best


# ---- matching -----------------------------------------------------------------

_CONFUSABLE = str.maketrans({"1": "i", "l": "i", "|": "i", "!": "i", "0": "o"})


def _norm(s):
    return s.lower().replace(" ", "").translate(_CONFUSABLE)


def match_whitelist(text, whitelist):
    """Whitelist entry matching the OCR text, or None.
    "Hum" matches exactly "Hum"; "Die*" matches anything starting with "Die".
    Case and spaces are ignored, and names of 5+ letters tolerate a small OCR misread."""
    t = _norm(text)
    for entry in whitelist:
        prefix = entry.endswith("*")
        n = _norm(entry.rstrip("*"))
        if not n:
            continue
        if t == n or (prefix and t.startswith(n)):
            return entry
        if len(n) >= 5:
            seen = t[:len(n)] if prefix else t
            if difflib.SequenceMatcher(None, seen, n).ratio() >= 0.85:
                return entry
    return None


# ---- geometry helpers ---------------------------------------------------------

def _in_rect(px, py, rect):
    x, y, w, h = rect
    return x <= px < x + w and y <= py < y + h


def _in_any_zone(px, py, cfg):
    return any(_in_rect(px, py, r) for r in cfg.get("no_click_zones", {}).values())


# ---- detection ----------------------------------------------------------------

def _merge_words(pieces, max_gap=WORD_GAP):
    """Merge text pieces on the same line into one label ("Die-" + "16")."""
    pieces = sorted(pieces)
    merged = []
    for x, y, w, h in pieces:
        for i, (mx, my, mw, mh) in enumerate(merged):
            same_line = abs((y + h / 2) - (my + mh / 2)) <= 3
            if same_line and mx <= x <= mx + mw + max_gap:
                x0, y0 = mx, min(my, y)
                merged[i] = (x0, y0, max(mx + mw, x + w) - x0, max(my + mh, y + h) - y0)
                break
        else:
            merged.append((x, y, w, h))
    return merged


def find_label_boxes(frame, region=None):
    """Candidate labels: bright text blobs surrounded by a dark box.
    With region=(x, y, w, h), only that part of the frame is searched (much faster);
    returned boxes are still in full-frame coordinates."""
    if region is not None:
        H, W = frame.shape[:2]
        rx, ry, rw, rh = region
        rx, ry = max(int(rx), 0), max(int(ry), 0)
        crop = frame[ry:min(ry + int(rh), H), rx:min(rx + int(rw), W)]
        if crop.size == 0:
            return []
        return [((x + rx, y + ry, w, h), ok) for (x, y, w, h), ok in find_label_boxes(crop)]
    v = frame.max(axis=2)                               # HSV value channel
    darkest = cv2.erode(v, np.ones((5, 5), np.uint8))   # darkest pixel nearby
    text = ((v >= TEXT_MIN_V) & (v.astype(int) - darkest >= TEXT_MIN_CONTRAST)).astype(np.uint8)
    # Join letters into words. Horizontal only, so text never fuses with the
    # target HP bar drawn a few pixels below it.
    joined = cv2.morphologyEx(text, cv2.MORPH_CLOSE, np.ones((1, 4), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    H, W = v.shape
    pieces = [tuple(int(a) for a in s[:4]) for s in stats[1:]
              if TEXT_HEIGHT[0] <= s[3] <= TEXT_HEIGHT[1] and s[2] >= 3]
    boxes = []
    for x, y, w, h in _merge_words(pieces):
        if not TEXT_WIDTH[0] <= w <= TEXT_WIDTH[1]:
            continue
        if text[y:y + h, x:x + w].mean() > TEXT_MAX_FILL:
            continue
        x0, y0 = max(x - BOX_PAD, 0), max(y - BOX_PAD, 0)
        x1, y1 = min(x + w + BOX_PAD, W), min(y + h + BOX_PAD, H)
        around = v[y0:y1, x0:x1]
        not_text = cv2.dilate(text[y0:y1, x0:x1], np.ones((3, 3), np.uint8)) == 0
        on_box = not_text.any() and float(np.median(around[not_text])) <= BOX_MAX_MEDIAN_V
        boxes.append(((int(x), int(y), int(w), int(h)), bool(on_box)))
    return boxes


def find_targets(frame, cfg, return_labels=False, use_ocr=True):
    """Whitelisted mobs sorted nearest-to-the-player first (detector.player_position).
    With return_labels=True, also returns every candidate Label (for debugging).
    use_ocr=False stops before OCR (labels that pass get status "candidate")."""
    det = cfg.get("detector", {})
    whitelist = cfg.get("targets", {}).get("whitelist", [])
    self_rect = det.get("self_exclusion")
    offset = det.get("click_offset_y", 45)
    min_score = det.get("ocr_min_score", 0.5)
    H, W = frame.shape[:2]
    # Distance is measured from your character's body to each mob's body (click point).
    cx, cy = det.get("player_position") or (W / 2, H / 2)

    labels, targets = [], []
    for box, on_dark_box in find_label_boxes(frame):
        label = Label(box)
        labels.append(label)
        x, y, w, h = box
        lx, ly = x + w / 2, y + h / 2
        if not on_dark_box:
            label.status = "no_box"
            continue
        if self_rect and _in_rect(lx, ly, self_rect):
            label.status = "self"
            continue
        if _in_any_zone(lx, ly, cfg):
            label.status = "zone"
            continue
        if not use_ocr:
            label.status = "candidate"
            continue

        label.text, label.score = ocr_label(frame, box)
        if label.score < min_score:
            label.status = "unread"
            continue
        name = match_whitelist(label.text, whitelist)
        if not name:
            label.status = "not_whitelisted"
            continue

        click = (int(lx), int(ly + offset))
        if not (0 <= click[0] < W and 0 <= click[1] < H) or _in_any_zone(*click, cfg):
            label.status = "click_in_zone"
            continue
        label.status = "target"
        dist = float(np.hypot(click[0] - cx, click[1] - cy))
        targets.append(Target(name, label.text, label.score, box, click, dist))

    targets.sort(key=lambda t: t.distance)
    return (targets, labels) if return_labels else targets
