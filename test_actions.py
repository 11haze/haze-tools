"""Test the action layer.

  python test_actions.py --dry [image]   offline: detect targets in a screenshot and log
                                         what would be sent, without sending any input
  python test_actions.py --live          one real fight: acquire the nearest target and
                                         attack until it dies (see combat: in config.yaml).
                                         F12 = stop
"""
import argparse
import sys
import time
from pathlib import Path

from actions import Actions, GameNotFocused
from capture import GameCapture, load_image
from config import ROOT, load_config
from detector import find_targets, warm_up
from killswitch import start_kill_switch
from session_log import SessionLog


class ImageCapture:
    """Stands in for GameCapture in dry runs: a fixed image, client == screen coords."""
    def __init__(self, img):
        self.img = img

    def grab(self):
        return self.img

    def to_screen(self, x, y):
        return int(x), int(y)

    def is_foreground(self):
        return True


def dry_run(cfg, log, image):
    img = load_image(image)
    act = Actions(cfg, ImageCapture(img), log, dry_run=True)
    targets = find_targets(img, cfg)
    log.log("SCAN", f"{len(targets)} target(s): " + ", ".join(f"{t.text}@{t.click}" for t in targets))
    if not targets:
        return
    act.acquire(targets)

    # Rotation/cooldown check with a simulated clock: skills 1,2,3 have cooldowns 1.5/3/6.
    for s in act.skills:
        s.last_used = float("-inf")
    act._rotation = 0
    order = []
    for step in range(12):
        now = step * 0.5
        skill = act.next_skill(now)
        if skill:
            skill.last_used = now
            act._rotation = (act.skills.index(skill) + 1) % len(act.skills)
        order.append(f"t={now:.1f}s:{skill.key if skill else '-'}")
    log.log("ROTATION_SIM", " ".join(order))
    act.check_potions(img)


def live_run(cfg, log):
    cap = GameCapture(cfg)
    if cap.input_blocked:
        log.log("ERROR", "The game runs as administrator but this script doesn't, so Windows "
                         "blocks its keys and clicks. Run the terminal as administrator.")
        return
    act = Actions(cfg, cap, log)
    warm_up()
    cap.activate()
    time.sleep(1.0)
    cap.refresh_rect()
    frame = cap.grab()
    targets = find_targets(frame, cfg)
    log.log("SCAN", f"{len(targets)} target(s): " + ", ".join(f"{t.text}@{t.click}" for t in targets))
    if not targets:
        log.log("SCAN_EMPTY", f"frame saved: {act._save_debug_frame(frame, 'scan_empty')}")
        return
    try:
        target = act.acquire(targets)
        if not target:
            return
        if act.fight(target) != "timeout":
            act.loot()
    except GameNotFocused:
        log.log("STOP", "game lost focus")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry", action="store_true")
    mode.add_argument("--live", action="store_true")
    ap.add_argument("image", nargs="?", default=str(ROOT / "screenshots" / "calib_20260926_203655.png"))
    args = ap.parse_args()

    cfg = load_config()
    log = SessionLog(cfg["logging"]["dir"])
    start_kill_switch(log, cfg.get("kill_switch", "f12"))
    log.state = "TEST"
    try:
        if args.dry:
            dry_run(cfg, log, Path(args.image))
        else:
            live_run(cfg, log)
    except Exception as e:
        log.log("ERROR", f"{type(e).__name__}: {e}")
        raise
    finally:
        log.log("END", str(log.path))
        log.close()


if __name__ == "__main__":
    sys.exit(main())
