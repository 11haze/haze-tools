"""Global kill switch. Pressing the hotkey (default F12) logs it and hard-exits
the process immediately, so no further input can be sent."""
import os

from pynput import keyboard


def start_kill_switch(session_log, key_name="f12"):
    target = getattr(keyboard.Key, key_name.lower())

    def on_press(key):
        if key == target:
            try:
                session_log.log("KILL_SWITCH", f"{key_name.upper()} pressed")
                session_log.close()
            finally:
                os._exit(0)

    listener = keyboard.Listener(on_press=on_press, daemon=True)
    listener.start()
    return listener
