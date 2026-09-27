"""Keyboard and mouse input via the Windows SendInput API.

Keys are sent as hardware scan codes, which DirectInput games like RAN read
more reliably than virtual-key codes. The cursor is placed with SetCursorPos.
"""
import ctypes
import ctypes.wintypes as wt

# Own handle with explicit signatures: ctypes.windll.user32 is shared process-wide,
# and pynput changes the argtypes of some functions on it (e.g. VkKeyScanW).
user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.VkKeyScanW.argtypes = [wt.WCHAR]
user32.VkKeyScanW.restype = ctypes.c_short
user32.MapVirtualKeyW.argtypes = [wt.UINT, wt.UINT]
user32.MapVirtualKeyW.restype = wt.UINT
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wt.BOOL

# Windows rounds sleeps up to ~15.6ms by default; 1ms resolution keeps the fixed
# delays (and the 0.1s click interval) close to their configured values.
ctypes.WinDLL("winmm").timeBeginPeriod(1)

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE = 0x0002, 0x0008
MOUSE_FLAGS = {
    "left": (0x0002, 0x0004),
    "right": (0x0008, 0x0010),
}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wt.UINT


SPECIAL_SCANCODES = {
    "space": 0x39, "enter": 0x1C, "esc": 0x01, "tab": 0x0F, "backspace": 0x0E,
    "shift": 0x2A, "ctrl": 0x1D, "alt": 0x38,
    **{f"f{i}": 0x3A + i for i in range(1, 11)}, "f11": 0x57, "f12": 0x58,
}


def scancode(key):
    """Scan code for a key name: "1", "q", "space", "f1", ..."""
    key = str(key).lower()
    if key in SPECIAL_SCANCODES:
        return SPECIAL_SCANCODES[key]
    if len(key) != 1:
        raise ValueError(f"Unknown key {key!r}")
    vk = user32.VkKeyScanW(key) & 0xFF
    sc = user32.MapVirtualKeyW(vk, 0)   # MAPVK_VK_TO_VSC
    if not sc:
        raise ValueError(f"No scan code for key {key!r}")
    return sc


def _send(*inputs):
    arr = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        raise OSError(f"SendInput sent {sent}/{len(inputs)} events (error {ctypes.get_last_error()}). "
                      "If the game runs as administrator, run the bot as administrator too.")


def _key_event(sc, up):
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    return INPUT(type=INPUT_KEYBOARD, u=_INPUTUNION(ki=KEYBDINPUT(0, sc, flags, 0, 0)))


def _mouse_event(flags):
    return INPUT(type=INPUT_MOUSE, u=_INPUTUNION(mi=MOUSEINPUT(0, 0, 0, flags, 0, 0)))


def key_down(key):
    _send(_key_event(scancode(key), up=False))


def key_up(key):
    _send(_key_event(scancode(key), up=True))


def move_cursor(x, y):
    """Place the cursor at screen coordinates."""
    if not user32.SetCursorPos(int(x), int(y)):
        raise OSError(f"SetCursorPos({x}, {y}) failed (error {ctypes.get_last_error()}). "
                      "If the game runs as administrator, run the bot as administrator too.")


def mouse_down(button="left"):
    _send(_mouse_event(MOUSE_FLAGS[button][0]))


def mouse_up(button="left"):
    _send(_mouse_event(MOUSE_FLAGS[button][1]))
