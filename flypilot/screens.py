"""Which screen the windows open on: the monitors as WSLg's X server sees them, and moving windows there.

WSLg shows the Windows desktop to Linux as one X screen spanning every monitor (on a PC with a 4K screen
left of a 1080p one: 5760 x 2160). XRandR lists the monitors in it, and xdotool moves a window to a
spot. launch.py moves each of its windows (the 3D view, the dashboard, the Stops window, the camera)
to one monitor once, as it appears, so they don't land on the screen you're using for something else.
A window still takes the focus for a moment when it opens; flights with --headless open none.
"""
import ctypes
import ctypes.util
import shutil
import subprocess
import threading
from typing import NamedTuple


class Monitor(NamedTuple):
    x: int
    y: int
    width: int
    height: int
    primary: bool


class _XRRMonitorInfo(ctypes.Structure):
    _fields_ = [('name', ctypes.c_ulong), ('primary', ctypes.c_int), ('automatic', ctypes.c_int),
                ('noutput', ctypes.c_int), ('x', ctypes.c_int), ('y', ctypes.c_int),
                ('width', ctypes.c_int), ('height', ctypes.c_int), ('mwidth', ctypes.c_int),
                ('mheight', ctypes.c_int), ('outputs', ctypes.c_void_p)]


def monitors() -> list:
    """The X server's monitors (XRRGetMonitors), [] if it can't be asked."""
    try:
        x11 = ctypes.CDLL(ctypes.util.find_library('X11') or 'libX11.so.6')
        xrr = ctypes.CDLL(ctypes.util.find_library('Xrandr') or 'libXrandr.so.2')
    except OSError:
        return []
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    xrr.XRRGetMonitors.restype = ctypes.POINTER(_XRRMonitorInfo)
    xrr.XRRGetMonitors.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    xrr.XRRFreeMonitors.argtypes = [ctypes.POINTER(_XRRMonitorInfo)]
    display = x11.XOpenDisplay(None)
    if not display:
        return []
    try:
        n = ctypes.c_int(0)
        info = xrr.XRRGetMonitors(display, x11.XDefaultRootWindow(display), 1, ctypes.byref(n))
        if not info:
            return []
        out = [Monitor(info[k].x, info[k].y, info[k].width, info[k].height, bool(info[k].primary))
               for k in range(n.value)]
        xrr.XRRFreeMonitors(info)
        return out
    finally:
        x11.XCloseDisplay(display)


def _windows_main_origin() -> tuple | None:
    """Where Windows' main screen starts in the X screen: Windows puts its main screen's corner at (0, 0)
    and WSLg's X screen starts at the top-left-most monitor corner, so it is minus the smallest corner.
    (XRandR here doesn't say which monitor is the main one.) None if Windows can't be asked."""
    ps = shutil.which('powershell.exe') or '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    cmd = ('Add-Type -AssemblyName System.Windows.Forms; '
           "[System.Windows.Forms.Screen]::AllScreens | ForEach-Object { '{0} {1}' -f $_.Bounds.X, $_.Bounds.Y }")
    try:
        out = subprocess.run([ps, '-NoProfile', '-NonInteractive', '-Command', cmd], capture_output=True,
                             text=True, timeout=15).stdout.split()
        corners = [(int(out[k]), int(out[k + 1])) for k in range(0, len(out) - 1, 2)]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    if not corners:
        return None
    return -min(c[0] for c in corners), -min(c[1] for c in corners)


def pick(choice: str) -> Monitor | None:
    """The monitor for `choice`: 'auto' (the biggest one that isn't the main screen, if there is one),
    'main', or a number (1 = the main screen, 2.. the others left to right). None: leave windows be."""
    mons = monitors()
    if not mons or choice == 'off':
        return None
    main = next((m for m in mons if m.primary), None)
    if main is None:
        origin = _windows_main_origin()
        if origin is None:
            return None  # can't tell which is the main screen: leave the windows where they open
        main = min(mons, key=lambda m: abs(m.x - origin[0]) + abs(m.y - origin[1]))
    others = sorted((m for m in mons if m is not main), key=lambda m: m.x)
    if choice == 'main':
        return main
    if choice == 'auto':
        return max(others, key=lambda m: m.width * m.height) if others else None
    try:
        k = int(choice)
    except ValueError:
        return None
    ordered = [main] + others
    return ordered[k - 1] if 1 <= k <= len(ordered) else None


# Where each window goes on the chosen monitor, as fractions of its width and height (top-left corner).
LAYOUT = (('MuJoCo', 0.02, 0.03), ('Avoidance dashboard', 0.55, 0.03), ('Delivery stops', 0.55, 0.45),
          ('AI-deck camera', 0.02, 0.62))


class WindowPlacer(threading.Thread):
    """Moves each window named in LAYOUT to its spot on `monitor` once, when it appears (then leaves it
    alone: drag it wherever you like). Does nothing without xdotool."""

    def __init__(self, monitor: Monitor):
        super().__init__(daemon=True)
        self.monitor, self._done, self._stop = monitor, set(), threading.Event()

    @staticmethod
    def available() -> bool:
        return shutil.which('xdotool') is not None

    def stop(self):
        self._stop.set()

    def run(self):
        m = self.monitor
        while not self._stop.wait(0.3):
            for title, fx, fy in LAYOUT:
                try:
                    ids = subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', title], capture_output=True,
                                         text=True, timeout=2).stdout.split()
                except (OSError, subprocess.TimeoutExpired):
                    return
                for wid in ids:
                    if wid in self._done:
                        continue
                    x, y = m.x + int(fx * m.width), m.y + int(fy * m.height)
                    if subprocess.run(['xdotool', 'windowmove', wid, str(x), str(y)], capture_output=True,
                                      timeout=2).returncode == 0:
                        self._done.add(wid)


if __name__ == '__main__':  # bash py.sh screens.py: what launch.py would do with its windows
    for mon in monitors():
        print(mon)
    print("Windows' main screen starts at", _windows_main_origin())
    for c in ('auto', 'main', '1', '2'):
        print(f'--screen {c} ->', pick(c))
