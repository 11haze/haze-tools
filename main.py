"""Entry point.

  python main.py           farm: find, attack and loot whitelisted mobs until the game
                           window stops being the foreground window (or F12)
  python main.py --check   setup check: find the window, save one capture, wait for F12
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

from actions import Actions
from brain import Brain
from capture import GameCapture
from config import load_config
from detector import warm_up
from killswitch import start_kill_switch
from session_log import SessionLog


def open_game(cfg, log):
    try:
        cap = GameCapture(cfg)
    except RuntimeError as e:
        log.log("ERROR", str(e))
        return None
    if not cap.activate():
        log.log("WARN", "could not bring the game to the front - click it within 3s")
    time.sleep(3 if not cap.is_foreground() else 0.5)  # let it redraw on top
    if not cap.refresh_rect():
        log.log("WARN", f"client area is {cap.width}x{cap.height}, "
                        f"config expects {cfg['window']['width']}x{cfg['window']['height']}")
    log.log("WINDOW_FOUND", f"client at ({cap.left},{cap.top}) {cap.width}x{cap.height}")
    return cap


def setup_check(cfg, log, cap):
    out = Path(cfg["logging"]["dir"]) / "capture_test.png"
    cv2.imwrite(str(out), cap.grab())
    log.log("CAPTURE_TEST", f"saved {out}")
    print("Setup OK. Press F12 to exit (tests the kill switch).")
    while True:
        time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="setup check only")
    args = ap.parse_args()

    cfg = load_config()
    log = SessionLog(cfg["logging"]["dir"])
    start_kill_switch(log, cfg.get("kill_switch", "f12"))
    log.log("START", f"log={log.path}")
    try:
        if not args.check:
            warm_up()                 # load OCR before the game is focused
        cap = open_game(cfg, log)
        if cap is None:
            return 1
        if args.check:
            setup_check(cfg, log, cap)
        if cap.input_blocked:
            log.log("ERROR", "The game runs as administrator but this script doesn't, so Windows "
                             "blocks its keys and clicks. Run the terminal as administrator.")
            return 1
        Brain(cfg, cap, log, Actions(cfg, cap, log)).run()
    except KeyboardInterrupt:
        log.log("STOP", "Ctrl+C")
    except Exception as e:
        log.log("ERROR", f"{type(e).__name__}: {e}")
        raise
    finally:
        log.log("END", str(log.path))
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
