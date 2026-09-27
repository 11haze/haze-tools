"""Game-state readers. All functions take a BGR frame (client area) and the config.

- get_hp_percent / get_mp_percent: fixed bars (regions.hp_bar / mp_bar).
- is_target_selected: the target info box ("Lv.301 Die-8 / 17622/44675 (39%)"),
  searched for inside regions.target_frame_search because its position differs
  between UI layouts.
- get_target_hp_percent: the floating bar under the targeted mob's nameplate. It
  moves with the mob, so it is searched for instead of read from a fixed region.
"""
import cv2
import numpy as np

# Target info box: light-gray 1px border around a dark translucent fill.
# Height is fixed (~43px); width follows the target's name.
BORDER_MAX_S, BORDER_V = 60, (90, 230)   # border is translucent - floor color tints it
BOX_MAX_MEDIAN_V = 100        # translucent: over bright ground the inside is gray, not black
BOX_MIN_CONTRAST = 50         # border line must be this much brighter than the inside
TBOX_WIDTH = (100, 320)
TBOX_HEIGHT = (38, 160)       # rows between top and bottom border; quest mobs add
                              # "You have quests in progress / Quest: ..." lines

# Floating target HP bar: pure-black 1px top/bottom border, red fill, gray empty.
BAR_BLACK_V = 2              # border is pure black; the box behind it can be nearly black
BAR_WIDTH = (110, 160)       # full bar incl. borders is ~127px; narrower = partly covered
BAR_HEIGHT = (5, 11)          # distance between top and bottom border rows
BAR_MIN_COVERAGE = 0.85       # share of columns that must be fill or empty


def hsv_mask(hsv, rng):
    """Boolean mask of pixels inside an HSV range. h[0] > h[1] wraps around red."""
    (h0, h1), (s0, s1), (v0, v1) = rng["h"], rng["s"], rng["v"]
    if h0 <= h1:
        m = cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
    else:
        m = cv2.inRange(hsv, (h0, s0, v0), (179, s1, v1)) | cv2.inRange(hsv, (0, s0, v0), (h1, s1, v1))
    return m > 0


def _crop(frame, rect):
    x, y, w, h = rect
    return frame[y:y + h, x:x + w]


def _bar_percent(frame, rect, fill_rng):
    """Percent of a left-to-right bar that is filled. Uses the rightmost filled
    column, so number text drawn over the bar doesn't lower the reading."""
    hsv = cv2.cvtColor(_crop(frame, rect), cv2.COLOR_BGR2HSV)
    filled_cols = hsv_mask(hsv, fill_rng).mean(axis=0) >= 0.25
    if not filled_cols.any():
        return 0.0
    return 100.0 * (np.flatnonzero(filled_cols)[-1] + 1) / filled_cols.size


def _fixed_bar(frame, cfg, region, color):
    rect = cfg["regions"].get(region)
    rng = cfg["colors"].get(color)
    if not rect or not rng:
        return None
    return _bar_percent(frame, rect, rng)


def get_hp_percent(frame, cfg):
    return _fixed_bar(frame, cfg, "hp_bar", "hp_fill")


def get_mp_percent(frame, cfg):
    return _fixed_bar(frame, cfg, "mp_bar", "mp_fill")


_last_box = None


def find_target_box(frame, cfg):
    """Locate the target info box inside regions.target_frame_search.
    Its position and width change with the UI layout and the target's name,
    so it is searched for. Returns (x, y, w, h) in client coordinates, or None.
    The spot where it was last found is checked first (~1ms instead of ~40ms)."""
    global _last_box
    H, W = frame.shape[:2]
    area = cfg["regions"].get("target_frame_search") or (0, H // 2, W, H - H // 2)
    if _last_box:
        x, y, w, h = _last_box
        # Width follows the name; height grows with quest lines (box grows upward).
        near = (max(x - 20, 0), max(y - 130, 0), w + 220, h + 150)
        box = _search_box(frame, near)
        if box:
            _last_box = box
            return box
    box = _search_box(frame, area)
    if box:
        _last_box = box
    return box


def _search_box(frame, region):
    H, W = frame.shape[:2]
    sx, sy, sw, sh = region
    sw, sh = min(sw, W - sx), min(sh, H - sy)
    if sw <= 0 or sh <= 0:
        return None
    hsv = cv2.cvtColor(frame[sy:sy + sh, sx:sx + sw], cv2.COLOR_BGR2HSV)
    border = (hsv[..., 1] < BORDER_MAX_S) & (hsv[..., 2] >= BORDER_V[0]) & (hsv[..., 2] <= BORDER_V[1])
    # Bridge 1-3px breaks in the border lines (corners, bright spots behind the box).
    border = cv2.morphologyEx(border.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8)) > 0
    value = hsv[..., 2]
    rows = border.shape[0]

    best = None
    for y in np.flatnonzero(border.sum(axis=1) >= TBOX_WIDTH[0]):
        for x0, run_end in _black_runs(border[y], TBOX_WIDTH[0]):
            # The top line can run past the box when gray ground sits next to it,
            # so the box's width comes from its vertical side lines, not the run.
            reach = min(run_end, x0 + TBOX_WIDTH[1])
            # Just inside a real box the fill is dark; on flat gray ground it isn't.
            # Checking this first skips ground rows cheaply.
            if y + 2 >= rows or border[y + 2, x0 + 3:x0 + TBOX_WIDTH[0] - 3].mean() >= 0.5:
                continue
            for dh in range(TBOX_HEIGHT[0], TBOX_HEIGHT[1] + 1):
                y2 = y + dh
                if y2 >= rows:
                    break
                if border[y2, x0 + 2:x0 + TBOX_WIDTH[0] - 2].mean() < 0.8:
                    continue
                cols = border[y:y2 + 1, max(x0 - 1, 0):reach + 1].mean(axis=0)
                if cols[:3].max() < 0.8:                       # left side line
                    continue
                right = [i for i in np.flatnonzero(cols >= 0.8) if i >= TBOX_WIDTH[0]]
                if not right:
                    continue
                x1 = max(x0 - 1, 0) + int(right[0]) + 1        # first side line far enough right
                if border[y2, x0 + 2:x1 - 2].mean() < 0.8:     # bottom line spans the box
                    continue
                inside = value[y + 3:y2 - 2, x0 + 3:x1 - 3]
                if not inside.size:
                    continue
                # A real box is a thin light line around a clearly darker inside.
                # Flat gray ground (pavement) matches the border colors everywhere, so
                # require the row just inside to NOT look like border, and real contrast.
                inner_med = float(np.median(inside))
                edge_med = float(np.median(value[y, x0:x1]))
                thin = border[y + 2, x0 + 3:x1 - 3].mean() < 0.5
                if thin and inner_med <= BOX_MAX_MEDIAN_V and edge_med - inner_med >= BOX_MIN_CONTRAST:
                    cand = (int(sx + x0), int(sy + y), int(x1 - x0), int(dh + 1))
                    if best is None or cand[2] > best[2]:
                        best = cand
                    break
    return best


def is_target_selected(frame, cfg):
    return find_target_box(frame, cfg) is not None


def _black_runs(row, min_w):
    """(x0, x1) of each run of True in a 1-D bool array at least min_w long."""
    padded = np.concatenate(([False], row, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_w]


def find_target_hp_bar(frame, cfg):
    """Locate the floating target HP bar. Returns (x, y, w, h, percent) or None."""
    fill_rng = cfg["colors"].get("target_hp_fill")
    empty_rng = cfg["colors"].get("target_hp_empty")
    if not fill_rng or not empty_rng:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    black = hsv[..., 2] < BAR_BLACK_V
    for x, y, w, h in cfg.get("no_click_zones", {}).values():
        black[y:y + h, x:x + w] = False   # UI panels have black borders too
    fill = hsv_mask(hsv, fill_rng)
    empty = hsv_mask(hsv, empty_rng)

    best = None
    for y in np.flatnonzero(black[1:].sum(axis=1) >= BAR_WIDTH[0]) + 1:
        for x0, x1 in _black_runs(black[y], BAR_WIDTH[0]):
            # Top border: a thin black line, so the row above is mostly not black.
            if x1 - x0 > BAR_WIDTH[1] or black[y - 1, x0:x1].mean() > 0.5:
                continue
            for dy in range(BAR_HEIGHT[0], BAR_HEIGHT[1] + 1):
                y2 = y + dy
                if y2 >= black.shape[0] or black[y2, x0:x1].mean() < 0.8:
                    continue
                inner = slice(y + 1, y2)
                f = fill[inner, x0:x1].mean(axis=0) >= 0.5
                e = empty[inner, x0:x1].mean(axis=0) >= 0.5
                covered = (f | e).mean()
                if covered < BAR_MIN_COVERAGE:
                    continue
                pct = 100.0 * f.sum() / max((f | e).sum(), 1)
                cand = (int(x0), int(y), int(x1 - x0), int(dy), pct)
                if best is None or cand[2] > best[2]:
                    best = cand
                break
    return best


def get_target_hp_percent(frame, cfg):
    """Target HP percent, or None if no target is selected or its bar isn't visible."""
    if not is_target_selected(frame, cfg):
        return None
    bar = find_target_hp_bar(frame, cfg)
    return bar[4] if bar else None
