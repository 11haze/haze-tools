"""Farming loop (Prompt 6, part 1): SEARCH -> ACQUIRE -> ATTACK -> LOOT -> SEARCH ...

Runs until the game window stops being the foreground window, or F12.
PATROL (walking around when nothing is in sight) comes later; for now SEARCH
waits and rescans.
"""
import time
import traceback
from enum import Enum

import numpy as np

from actions import GameNotFocused
from detector import find_targets


class State(str, Enum):
    SEARCH = "SEARCH"
    ACQUIRE = "ACQUIRE"
    ATTACK = "ATTACK"
    LOOT = "LOOT"


class Brain:
    def __init__(self, cfg, capture, log, actions):
        self.cfg = cfg
        self.capture = capture
        self.log = log
        self.act = actions
        f = cfg.get("farming", {})
        self.scan_interval = f.get("scan_interval", 0.5)
        self.skip_seconds = f.get("skip_failed_seconds", 5.0)
        self.state = State.SEARCH
        self.targets = []
        self.target = None
        self.last_name = None
        self.kills = 0
        self._failed = []           # (time, click) of targets that couldn't be selected
        self._last_idle_log = 0.0
        self._no_target_since = None
        self.sidestep_after = cfg.get("sidestep", {}).get("no_target_after", 1.5)

    def set_state(self, state):
        if state != self.state:
            self.log.log("STATE", f"{self.state.value} -> {state.value}", state=state.value)
        self.state = state
        self.log.state = state.value

    # ---- main loop --------------------------------------------------------

    def run(self):
        self.log.log("FARM_START", f"whitelist={self.cfg['targets']['whitelist']}")
        self.set_state(State.SEARCH)
        errors = []
        try:
            while True:
                if not self.capture.is_foreground():
                    self.log.log("STOP", "game is no longer the foreground window")
                    break
                try:
                    self.step()
                except GameNotFocused:
                    raise
                except Exception as e:
                    # A bug in one step shouldn't end a long session: log it, save the
                    # screen, and start over from SEARCH. Too many in a row -> stop.
                    now = time.monotonic()
                    errors = [t for t in errors if now - t < 60] + [now]
                    frame = self.act._save_debug_frame(self.capture.grab(), "error")
                    self.log.log("ERROR", f"{type(e).__name__}: {e} | "
                                          f"{traceback.format_exc(limit=3).strip().splitlines()[-3:]} "
                                          f"(frame: {frame})")
                    if len(errors) >= 5:
                        self.log.log("STOP", "5 errors within a minute - stopping")
                        raise
                    self.set_state(State.SEARCH)
        except GameNotFocused:
            self.log.log("STOP", "game lost focus")
        self.log.log("FARM_END", f"kills={self.kills}")

    def step(self):
        if self.state == State.SEARCH:
            self.search()
        elif self.state == State.ACQUIRE:
            self.acquire()
        elif self.state == State.ATTACK:
            self.attack()
        elif self.state == State.LOOT:
            self.loot()

    # ---- states -----------------------------------------------------------

    def search(self):
        frame = self.capture.grab()
        self.act.check_potions(frame)
        targets = [t for t in find_targets(frame, self.cfg) if not self._recently_failed(t)]
        now = time.monotonic()
        if not targets:
            self._no_target_since = self._no_target_since or now
            # Nothing targetable, yet HP is dropping: a mob is hitting you and its
            # nameplate is probably merged with yours or your pet's. Step aside.
            if (now - self._no_target_since >= self.sidestep_after and self.act.being_hit()
                    and self.act.can_sidestep()):
                self.act.sidestep(frame, "taking damage but no target visible")
                return
            if now - self._last_idle_log >= 5.0:
                self.log.log("NO_TARGETS", "nothing whitelisted in sight - waiting")
                self._last_idle_log = now
            time.sleep(self.scan_interval)
            return
        self._no_target_since = None
        # find_targets already sorts nearest-to-the-player first, so mobs attacking
        # you or standing next to you are taken before anything farther away.
        self.log.log("SCAN", f"{len(targets)} target(s), nearest first: "
                             + ", ".join(f"{t.text}@{t.click} {t.distance:.0f}px" for t in targets[:5]))
        self.targets = targets
        self.set_state(State.ACQUIRE)

    def acquire(self):
        target = self.act.acquire(self.targets)
        if target:
            self.target = target
            self.set_state(State.ATTACK)
            return
        now = time.monotonic()
        self._failed += [(now, t.click) for t in self.targets[:self.act.acquire_max_tries]]
        self.set_state(State.SEARCH)

    def attack(self):
        result = self.act.fight(self.target)
        if result in ("gone", "dead"):
            self.kills += 1
            self.last_name = self.target.name
            self.log.log("KILL", f"#{self.kills} {self.target.name}")
            self.set_state(State.LOOT)
        else:                        # "lost" / "timeout": move on without looting
            self.set_state(State.SEARCH)

    def loot(self):
        self.act.loot()
        self.set_state(State.SEARCH)

    # ---- helpers ----------------------------------------------------------

    def _recently_failed(self, target):
        """Skip spots where acquiring just failed (e.g. an unreachable mob), for a few seconds."""
        now = time.monotonic()
        self._failed = [(t, c) for t, c in self._failed if now - t < self.skip_seconds]
        return any(np.hypot(target.click[0] - c[0], target.click[1] - c[1]) < 40 for _, c in self._failed)
