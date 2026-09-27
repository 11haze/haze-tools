"""Screen capture of the game window's client area (Windows)."""
import ctypes
import ctypes.wintypes as wt

import cv2
import mss
import numpy as np

user32 = ctypes.windll.user32

# Without DPI awareness, window coordinates are scaled on high-DPI displays and
# captures/clicks land in the wrong place.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except (AttributeError, OSError):
    user32.SetProcessDPIAware()


def find_window(title_part):
    """Return the HWND of the first visible window whose title contains title_part."""
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def enum_proc(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if title_part.lower() in buf.value.lower():
                found.append(hwnd)
                return False
        return True

    user32.EnumWindows(enum_proc, 0)
    return found[0] if found else None


def client_rect(hwnd):
    """Client area as (left, top, width, height) in screen coordinates."""
    rect = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    origin = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return origin.x, origin.y, rect.right - rect.left, rect.bottom - rect.top


def is_admin():
    return bool(ctypes.windll.shell32.IsUserAnAdmin())


def window_is_elevated(hwnd):
    """True if the window's process runs as administrator (None if it can't be checked)."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32.OpenProcess.restype = wt.HANDLE
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    proc = kernel32.OpenProcess(0x1000, False, pid.value)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not proc:
        return None
    token = wt.HANDLE()
    try:
        if not advapi32.OpenProcessToken(proc, 0x0008, ctypes.byref(token)):   # TOKEN_QUERY
            return None
        elevated, size = wt.DWORD(), wt.DWORD()
        advapi32.GetTokenInformation(token, 20, ctypes.byref(elevated), 4, ctypes.byref(size))
        return bool(elevated.value)
    finally:
        if token:
            kernel32.CloseHandle(token)
        kernel32.CloseHandle(proc)


class GameCapture:
    def __init__(self, cfg):
        title = cfg["window"]["title"]
        self.hwnd = find_window(title)
        if not self.hwnd:
            raise RuntimeError(f"No visible window with title containing {title!r}")
        self.expected = (cfg["window"]["width"], cfg["window"]["height"])
        self.input_blocked = bool(window_is_elevated(self.hwnd)) and not is_admin()
        self._sct = mss.mss()
        self.refresh_rect()

    def activate(self):
        """Restore and bring the game window to the front. mss captures what is
        visible on screen, so any window covering the game ends up in the frame."""
        if user32.IsIconic(self.hwnd):
            user32.ShowWindow(self.hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(self.hwnd)
        return self.is_foreground()

    def is_foreground(self):
        return user32.GetForegroundWindow() == self.hwnd

    def refresh_rect(self):
        self.left, self.top, self.width, self.height = client_rect(self.hwnd)
        return (self.width, self.height) == self.expected

    def grab(self):
        """Current client area as a BGR numpy array."""
        box = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
        img = np.asarray(self._sct.grab(box))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def to_screen(self, x, y):
        """Convert client-area coordinates to screen coordinates for input."""
        return self.left + int(x), self.top + int(y)


def load_image(path):
    """Load a saved screenshot as BGR, for offline testing."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img
