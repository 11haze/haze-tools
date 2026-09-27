# RAN Online Farming Bot: Claude Code Prompt Sequence

Screen-reading farming bot for a friend's private server, built with the server admin's approval as part of an anti-cheat test. No anti-detection or evasion features by design.

Send each prompt to Claude Code one at a time. Test each step before moving on. Fill in anything in [brackets].

---

## Prep (before Prompt 1)

- Lock the client in windowed mode at 1017x772 (or note your actual resolution).
- Take 30 to 50 screenshots at your farm spot and put them in /screenshots:
  - mobs walking, mobs mid-fight, other players nearby, low HP moments
  - a few with a mob **targeted** (needed to calibrate the target frame)
  - a few with **red** nameplates and a few with **white** nameplates
- Note your farm mob names (e.g. "Hum"), skill slots, and rough cooldowns.

## Client notes

- Movement: left-click to move.
- Attack: press skill key (1-9), then RIGHT click the enemy.
- Loot: auto-picked by pet, or SPACEBAR.
- Mob nameplates: text on a small dark box, about 40-50 px above the mob. Text color changes with level difference (red when stronger, white once out-leveled).
- Player names: white text with icons, no dark box.
- No-click zones: chat (bottom-left), menu bar (bottom-right), minimap (top-right), skill bars (left vertical, top Q/W/E/A/S/D).

---

## Prompt 1: Project setup

```
I'm building a screen-reading farming bot for RAN Online, running on a friend's private server with his approval. He's using it to test his anti-cheat, so do NOT add any anti-detection, input humanization, or anti-cheat evasion features. It should behave like a plain, honest bot.

Set up a Python project with a venv using mss (screen capture), opencv-python, numpy, pynput (input), and pyyaml. Create a config.yaml holding: game window title "[window title]", resolution 1017x772 (windowed), skill slots [1,2,3] with cooldowns, target name whitelist (default: ["Hum"], editable), HP/MP thresholds, potion keys [Q/W], loot mode ("pet" or "spacebar"), and a list of no-click UI zones (chat, menu bar, minimap, skill bars). Add F12 as a global kill switch that stops everything instantly, plus a logger that writes every action with a timestamp to logs/session_[date].csv. Keep the code modular: capture.py, readers.py, detector.py, actions.py, brain.py, main.py.
```

## Prompt 2: Calibration tool

```
Write calibrate.py. It opens a screenshot from /screenshots and lets me click-drag to define screen regions: my HP bar, my MP bar, the target frame, the target's HP bar, and each no-click UI zone. Save the regions as coordinates in config.yaml. Also let me click a pixel to sample colors for the HP bar fill and the empty bar, and save those as HSV ranges.
```

## Prompt 3: State readers

```
In readers.py, using the regions from config.yaml, write functions for:
- get_hp_percent() and get_mp_percent(), measured by how much of each bar matches the fill color
- is_target_selected(), true when the target frame is visible
- get_target_hp_percent()
Test them against my saved screenshots and print the results so I can verify they're accurate.
```

## Prompt 4: Nameplate detector (the core piece)

```
In detector.py, find mobs by their nameplates. Mob names are text on a small solid dark rectangle floating above the mob. The text color changes with level difference (red for stronger mobs, white once I out-level them, possibly other colors), so do NOT filter by text color. Player names (including mine, near screen center) are white text with icons and no dark background box.
1. Find small dark rectangular label boxes, then confirm each contains bright text of any color (high contrast against the box, not a specific hue)
2. Exclude a zone around screen center (my character) and every no-click UI zone from config
3. OCR each label and keep only names in the target whitelist from config.yaml. This name check is the real filter, since player names can also be white. The whitelist must be editable without code changes
4. Return targets sorted by distance from screen center, each with a click point [45] px below the label center (the mob's body), rejecting any click point inside a no-click zone
Build debug_view.py that runs this on every image in /screenshots and saves annotated copies showing boxes, OCR text, and click points.
```

## Prompt 5: Action layer

```
In actions.py, write:
- cast_on(target, slot): press the skill key [1-9], move the mouse to the target's click point, and right-click
- a skill rotation that cycles through my configured skills and respects cooldowns
- acquire check: after casting, confirm is_target_selected() within [0.5]s; if not, try the next target
- use_potion() when HP or MP drops below the config thresholds
Log every action to the session CSV.
```

## Prompt 6: State machine

```
In brain.py, build a state machine: PATROL -> ACQUIRE -> ATTACK -> LOOT -> RECOVER -> back to PATROL.
- PATROL: click-to-move with LEFT click on open ground points in a loop around a fixed anchor. Never left-click on or near a detected nameplate, and never inside no-click zones. Stop patrolling as soon as the detector finds a target.
- ACQUIRE/ATTACK: press the skill key, then RIGHT click the target's click point. Repeat the rotation until the target frame disappears or target HP reads 0.
- LOOT: if loot mode is "pet", wait [1]s and continue. If "spacebar", press SPACE [2-3] times.
- RECOVER: use the potion keys until HP/MP are above thresholds.
Safety stops: pause if HP stays critical, if the screen hasn't changed for [N]s (stuck), or if a non-whitelisted name appears within [radius] px of my character. main.py runs the loop and shows a small overlay with the current state.
```

## Prompt 7: Test report for the server admin

```
Write report.py. It reads a session CSV and outputs a summary: session length, kills, actions per minute, the timing distribution between actions, and time spent in each state. The server admin will compare this against his server logs to see which bot behaviors his anti-cheat caught and which it missed.
```

---

## Test loop with the admin

1. Run a session.
2. Generate the report (Prompt 7).
3. Admin compares it with server logs: what got flagged, what didn't.
4. Admin tightens detection. Repeat.
