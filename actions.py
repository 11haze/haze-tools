"""Input actions (Prompt 5): skills, clicks, target acquisition, potions, loot.

All delays are fixed values from config.yaml (timing). Every action is logged
to the session CSV. Nothing is sent unless the game window is in the foreground.
"""
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

import readers
import win_input
from detector import find_label_boxes, find_targets, relocate, target_box_name

# Sidestep directions, tried in this order (degrees; 0 = right, 90 = down).
SIDESTEP_ANGLES = [0, 180, 90, 270, 45, 225, 135, 315]


@dataclass
class Skill:
    key: str
    cooldown: float
    last_used: float = float("-inf")

    def ready(self, now):
        return now - self.last_used >= self.cooldown

    def remaining(self, now):
        return max(0.0, self.cooldown - (now - self.last_used))


class GameNotFocused(Exception):
    """Raised instead of sending input when the game isn't the foreground window."""


class Actions:
    def __init__(self, cfg, capture, log, dry_run=False):
        self.cfg = cfg
        self.capture = capture
        self.log = log
        self.dry_run = dry_run        # log actions without sending any input
        t = cfg.get("timing", {})
        self.key_hold = t.get("key_hold", 0.03)
        self.key_to_click = t.get("key_to_click", 0.05)
        self.click_hold = t.get("click_hold", 0.03)
        self.after_action = t.get("after_action", 0.10)
        self.acquire_timeout = t.get("acquire_timeout", 0.5)
        self.poll_interval = t.get("poll_interval", 0.05)
        self.potion_cooldown = t.get("potion_cooldown", 1.0)
        self.min_cast_interval = t.get("min_cast_interval", 0.3)
        self.click_interval = cfg.get("combat", {}).get("click_interval", 0.1)
        self._last_cast = float("-inf")
        self._last_click = float("-inf")
        self.acquire_max_tries = cfg.get("acquire_max_tries", 3)
        margin = float(cfg.get("cooldown_margin", 0.0))
        self.skills = [Skill(str(s["key"]), float(s["cooldown"]) + margin) for s in cfg["skills"]]
        self._rotation = 0
        self._last_potion = {"hp": float("-inf"), "mp": float("-inf")}
        self._hp_history = deque()    # (time, hp%) from check_potions
        s = cfg.get("sidestep", {})
        self.sidestep_on = s.get("enabled", False)
        self.sidestep_distance = s.get("distance", 0.25)
        self.sidestep_in_fight_after = s.get("in_fight_after", 1.0)
        self.sidestep_cooldown = s.get("cooldown", 2.0)
        self.sidestep_wait = s.get("wait_after", 0.4)
        self.last_sidestep = float("-inf")
        self._sidestep_dir = 0

    # ---- low-level input --------------------------------------------------

    def _ensure_focus(self):
        if self.dry_run:
            return
        if not self.capture.is_foreground():
            self.log.log("NO_FOCUS", "game is not the foreground window - input skipped")
            raise GameNotFocused()

    def press(self, key, reason=""):
        self._ensure_focus()
        self.log.log("KEY", f"{key}{' (' + reason + ')' if reason else ''}")
        if not self.dry_run:
            win_input.key_down(key)
            time.sleep(self.key_hold)
            win_input.key_up(key)

    def click(self, x, y, button="left", reason=""):
        """Click at client-area coordinates."""
        self._ensure_focus()
        sx, sy = self.capture.to_screen(x, y) if self.capture else (x, y)
        self.log.log(f"CLICK_{button.upper()}", f"client=({x},{y}) screen=({sx},{sy})"
                                                f"{' ' + reason if reason else ''}")
        if not self.dry_run:
            win_input.move_cursor(sx, sy)
            win_input.mouse_down(button)
            time.sleep(self.click_hold)
            win_input.mouse_up(button)

    def move_to(self, x, y):
        """Walk: left-click on open ground (used by PATROL in Prompt 6)."""
        self.click(x, y, "left", reason="move")
        time.sleep(self.after_action)

    # ---- skills -----------------------------------------------------------

    def next_skill(self, now=None):
        """Next ready skill in rotation order, or None if all are on cooldown."""
        now = time.monotonic() if now is None else now
        n = len(self.skills)
        for i in range(n):
            skill = self.skills[(self._rotation + i) % n]
            if skill.ready(now):
                return skill
        return None

    def seconds_until_ready(self, now=None):
        now = time.monotonic() if now is None else now
        return min(s.remaining(now) for s in self.skills)

    def cast_on(self, target, skill, fresh=False):
        """Press the skill key, then right-click the target's click point.
        Unless the caller just located the target (fresh=True), it is re-located
        on a new frame first, because mobs keep walking between scan and click.
        Casts are spaced at least timing.min_cast_interval apart.
        Returns the target as clicked."""
        wait = self._last_cast + self.min_cast_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_cast = time.monotonic()      # interval runs key press -> key press
        self.press(skill.key, reason="skill")
        time.sleep(self.key_to_click)
        if not self.dry_run and not fresh:
            moved = relocate(self.capture.grab(), target, self.cfg)
            if moved and moved.click != target.click:
                self.log.log("RELOCATE", f"{target.text!r} {target.click} -> {moved.click}")
                target = moved
        x, y = target.click
        self.click(x, y, "right", reason=f"target={target.text!r}")
        skill.last_used = time.monotonic()
        self._rotation = (self.skills.index(skill) + 1) % len(self.skills)
        self.log.log("CAST", f"skill {skill.key} on {target.text!r} at {target.click}")
        time.sleep(self.after_action)
        return target

    def _skill_for_acquire(self):
        """Skill to open a fight with. Spam mode doesn't track cooldowns, so it just
        takes the next key in order. Otherwise wait until one is ready - looped,
        because a sleep can end a hair before the cooldown is over."""
        if self.cfg.get("combat", {}).get("spam_casting", False):
            skill = self.skills[self._rotation]
            self._rotation = (self._rotation + 1) % len(self.skills)
            return skill
        while (skill := self.next_skill()) is None:
            time.sleep(max(self.seconds_until_ready(), 0.01))
        return skill

    def attack(self, target, fresh=False):
        """Cast the next ready skill on the target. Returns the target as clicked,
        or None if all skills are on cooldown."""
        skill = self.next_skill()
        if skill is None:
            return None
        return self.cast_on(target, skill, fresh=fresh)

    # ---- acquiring a target -----------------------------------------------

    def wait_for_target(self, timeout=None):
        """Poll the screen until the target frame appears. Returns (selected, last frame)."""
        deadline = time.monotonic() + (self.acquire_timeout if timeout is None else timeout)
        while True:
            frame = self.capture.grab()
            if readers.is_target_selected(frame, self.cfg):
                return True, frame
            if time.monotonic() >= deadline:
                return False, frame
            time.sleep(self.poll_interval)

    def _save_debug_frame(self, frame, tag):
        out = Path(self.cfg["logging"]["dir"]) / "debug_frames"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{tag}_{datetime.now():%H%M%S_%f}.png"
        cv2.imwrite(str(path), frame)
        return path

    def acquire(self, targets):
        """Cast on each target (nearest first) until the target frame confirms a selection.
        Returns the acquired target, or None if none of them could be selected."""
        for target in targets[:self.acquire_max_tries]:
            target = self.cast_on(target, self._skill_for_acquire())
            if self.dry_run:
                self.log.log("ACQUIRE_OK", f"{target.text!r} at {target.click} (dry run)")
                return target
            selected, frame = self.wait_for_target()
            if selected:
                self.log.log("ACQUIRE_OK", f"{target.text!r} at {target.click}")
                return target
            saved = self._save_debug_frame(frame, "acquire_fail")
            self.log.log("ACQUIRE_FAIL", f"{target.text!r} at {target.click} - no target frame "
                                         f"within {self.acquire_timeout}s, trying next (frame: {saved})")
        return None

    # ---- fighting ---------------------------------------------------------

    def fight(self, target):
        """Attack the acquired target with the skill rotation until it dies.
        Returns "dead" (HP read 0), "gone" (target frame closed), "lost" (neither
        its nameplate nor its name in the target box confirmed it for
        combat.nameplate_lost_timeout) or "timeout" (only when
        combat.attack_until_dead is false).
        With loot.space_while_fighting, SPACE is pressed every loot.space_interval
        seconds so drops are picked up as they land."""
        c = self.cfg.get("combat", {})
        until_dead = c.get("attack_until_dead", True)
        max_seconds = c.get("max_fight_seconds", 20)
        grace = c.get("target_lost_grace", 1.0)
        plate_timeout = c.get("nameplate_lost_timeout", 3.0)
        loot = self.cfg.get("loot", {})
        space_on = loot.get("space_while_fighting", False)
        space_every = loot.get("space_interval", 0.5)
        name_every = c.get("box_name_check_interval", 0.5)
        spam = c.get("spam_casting", False)
        start = time.monotonic()
        lost_since = None
        confirmed = start          # last time the nameplate or the box name confirmed the target
        plate_seen = start         # last time the nameplate itself was found
        last_name_check = float("-inf")
        last_space = float("-inf")

        while True:
            frame = self.capture.grab()
            self.check_potions(frame)
            now = time.monotonic()

            if space_on and now - last_space >= space_every:
                self.press("space", reason="loot while fighting")
                last_space = now

            box = readers.find_target_box(frame, self.cfg)
            if box:
                lost_since = None
                # The box shows the selected mob's name - it stays readable even when
                # the nameplate on screen is hidden behind another label.
                if now - last_name_check >= name_every:
                    last_name_check = now
                    if target_box_name(frame, box, self.cfg)[0] == target.name:
                        confirmed = now
                hp = readers.get_target_hp_percent(frame, self.cfg)
                if hp is not None and hp <= 0:
                    self.log.log("TARGET_DEAD", f"{target.text!r} HP 0 after {now - start:.1f}s")
                    return "dead"
            else:
                # Brief misreads (effects over the target box) shouldn't end the fight.
                lost_since = lost_since or now
                if now - lost_since >= grace:
                    self.log.log("TARGET_GONE", f"{target.text!r} target frame closed after {now - start:.1f}s")
                    return "gone"
                time.sleep(self.poll_interval)
                continue

            if not until_dead and now - start >= max_seconds:
                self.log.log("FIGHT_TIMEOUT", f"{target.text!r} still alive after {max_seconds}s")
                return "timeout"

            # The mob moves - follow its nameplate every loop (searches only around
            # its last position, no OCR, ~2ms).
            moved = relocate(frame, target, self.cfg)
            if moved:
                target, confirmed, plate_seen = moved, now, now
            elif (now - plate_seen >= self.sidestep_in_fight_after and now - confirmed < 0.6
                  and self.can_sidestep()):
                # The box says the mob is alive but its nameplate is hidden - usually
                # merged with your name or your pet's. Step aside, then find it again.
                if self.sidestep(frame, f"{target.text!r} nameplate hidden for "
                                        f"{now - plate_seen:.1f}s"):
                    found = [t for t in find_targets(self.capture.grab(), self.cfg)
                             if t.name == target.name]
                    if found:
                        target, plate_seen, confirmed = found[0], time.monotonic(), time.monotonic()
                        self.log.log("REFOUND", f"{target.text!r} at {target.click} after sidestep")
                continue
            elif now - confirmed >= plate_timeout:
                saved = "" if self.dry_run else f" (frame: {self._save_debug_frame(frame, 'nameplate_lost')})"
                self.log.log("TARGET_LOST", f"{target.text!r} not confirmed by nameplate or target box "
                                            f"for {plate_timeout}s, after {now - start:.1f}s{saved}")
                return "lost"
            if spam:
                self._spam_step(target)
                continue
            hit = self.attack(target, fresh=moved is not None)
            if hit:
                target = hit
            else:
                time.sleep(min(self.seconds_until_ready(), 0.2))

    def _spam_step(self, target):
        """Spam casting: ignore tracked cooldowns. Press the next skill key in order
        every timing.min_cast_interval (the game skips any still cooling down) and
        right-click the target at least every combat.click_interval."""
        now = time.monotonic()
        if now - self._last_cast >= self.min_cast_interval:
            skill = self.skills[self._rotation]
            self._rotation = (self._rotation + 1) % len(self.skills)
            self._last_cast = now
            self.press(skill.key, reason="skill (spam)")
            time.sleep(self.key_to_click)
            self._last_click = float("-inf")     # always click right after a skill key
        if time.monotonic() - self._last_click >= self.click_interval:
            self._last_click = time.monotonic()
            self.click(*target.click, "right", reason=f"target={target.text!r}")
        # Sleep until shortly before the next click or key is due; waking early leaves
        # time for the loop's screen reading (~35ms) so the click isn't late.
        due = min(self._last_click + self.click_interval, self._last_cast + self.min_cast_interval)
        time.sleep(max(0.0, due - time.monotonic() - 0.05))

    # ---- potions and loot -------------------------------------------------

    def use_potion(self, kind, reason=""):
        """kind: "hp" or "mp". Respects potion_cooldown so it isn't spammed."""
        now = time.monotonic()
        if now - self._last_potion[kind] < self.potion_cooldown:
            return False
        self.press(self.cfg["potions"][f"{kind}_key"], reason=f"{kind} potion {reason}".strip())
        self._last_potion[kind] = now
        return True

    def check_potions(self, frame):
        """Drink potions if HP or MP is below the config thresholds. Returns (hp, mp) readings."""
        th = self.cfg["thresholds"]
        hp = readers.get_hp_percent(frame, self.cfg)
        mp = readers.get_mp_percent(frame, self.cfg)
        if hp is not None and hp < th["hp_potion"]:
            self.use_potion("hp", reason=f"HP {hp:.0f}%")
        if mp is not None and mp < th["mp_potion"]:
            self.use_potion("mp", reason=f"MP {mp:.0f}%")
        if hp is not None:
            now = time.monotonic()
            self._hp_history.append((now, hp))
            while self._hp_history and now - self._hp_history[0][0] > 5.0:
                self._hp_history.popleft()
        return hp, mp

    def being_hit(self, window=3.0, min_drop=1.0):
        """True if HP dropped by at least min_drop percent within the last `window` seconds."""
        now = time.monotonic()
        recent = [hp for t, hp in self._hp_history if now - t <= window]
        return len(recent) >= 2 and max(recent) - recent[-1] >= min_drop

    # ---- sidestep -----------------------------------------------------------

    def sidestep(self, frame, reason):
        """Left-click open ground ~sidestep.distance of the screen away from the player,
        so nameplates that overlap yours or your pet's separate again. Directions
        rotate through SIDESTEP_ANGLES; spots in no-click zones, on nameplates or off
        screen are skipped. Returns True if a step was taken."""
        H, W = frame.shape[:2]
        px, py = self.cfg.get("detector", {}).get("player_position") or (W / 2, H / 2)
        zones = self.cfg.get("no_click_zones", {}).values()
        labels = [b for b, _ in find_label_boxes(frame)]

        def usable(x, y):
            if not (20 <= x < W - 20 and 20 <= y < H - 20):
                return False
            if any(zx <= x < zx + zw and zy <= y < zy + zh for zx, zy, zw, zh in zones):
                return False
            # Left-clicking a nameplate or the mob under it would select it, not walk.
            return not any(lx - 30 <= x <= lx + lw + 30 and ly - 30 <= y <= ly + lh + 70
                           for lx, ly, lw, lh in labels)

        for i in range(len(SIDESTEP_ANGLES)):
            angle = math.radians(SIDESTEP_ANGLES[(self._sidestep_dir + i) % len(SIDESTEP_ANGLES)])
            x = int(px + math.cos(angle) * self.sidestep_distance * W)
            y = int(py + math.sin(angle) * self.sidestep_distance * H)
            if usable(x, y):
                self._sidestep_dir = (self._sidestep_dir + i + 1) % len(SIDESTEP_ANGLES)
                self.log.log("SIDESTEP", f"{reason} -> walk to ({x},{y})")
                self.click(x, y, "left", reason="sidestep")
                self.last_sidestep = time.monotonic()
                time.sleep(self.sidestep_wait)
                return True
        self.log.log("SIDESTEP_SKIPPED", f"{reason} - no free ground in any direction")
        self.last_sidestep = time.monotonic()
        return False

    def can_sidestep(self):
        return self.sidestep_on and time.monotonic() - self.last_sidestep >= self.sidestep_cooldown

    def loot(self):
        """Pick up drops: wait for the pet, or press SPACE a few times."""
        loot = self.cfg["loot"]
        if loot["mode"] == "spacebar":
            for _ in range(loot.get("spacebar_presses", 3)):
                self.press("space", reason="loot")
                time.sleep(self.after_action)
        else:
            self.log.log("LOOT_WAIT", f"pet pickup {loot.get('pet_wait', 1.0)}s")
            time.sleep(loot.get("pet_wait", 1.0))
