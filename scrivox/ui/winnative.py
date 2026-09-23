"""Small platform helpers for a native-feeling window on Windows.

Everything here is best-effort: each helper swallows failures so an older
Windows build (or Linux/macOS during development) simply skips the nicety.
"""

import os
import subprocess
import sys

IS_WINDOWS = sys.platform == "win32"

# What a ctypes call into a missing/older Windows API can raise
_WIN_ERRORS = (AttributeError, OSError, ValueError, TypeError)

APP_USER_MODEL_ID = "KawaConnect.Scrivox"


def enable_dpi_awareness():
    """Opt in to DPI awareness. Must run BEFORE the first Tk() is created.

    Uses *system* awareness (1), not per-monitor: Tk 8.6 does not handle
    WM_DPICHANGED, so per-monitor mode would keep the original size when the
    window moves to a monitor with a different scale. System awareness still
    renders crisp text at the primary monitor's scale instead of a blurry
    bitmap-stretched UI.
    """
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except _WIN_ERRORS:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except _WIN_ERRORS:
            pass  # very old Windows, or awareness already set by the host


def set_app_user_model_id():
    """Give the taskbar button its own identity (and icon) instead of python's."""
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except _WIN_ERRORS:
        pass


def _user32():
    """user32 with argtypes declared, so a 64-bit HWND is never squeezed
    into a C int."""
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    u.GetParent.argtypes = [wintypes.HWND]
    u.GetParent.restype = wintypes.HWND
    u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, wintypes.UINT]
    u.SetWindowPos.restype = wintypes.BOOL
    return u


def _hwnd(win):
    win.update_idletasks()
    return _user32().GetParent(win.winfo_id()) or win.winfo_id()


def use_dark_title_bar(win):
    """Ask DWM for a dark title bar so it matches the dark window body.

    Attribute 20 is DWMWA_USE_IMMERSIVE_DARK_MODE on Windows 10 20H1+ and
    Windows 11; 19 is the pre-release value used by Windows 10 1809-1909.
    Windows 10 only repaints the frame when told to, hence SWP_FRAMECHANGED.
    """
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        from ctypes import wintypes
        hwnd = _hwnd(win)
        dwm = ctypes.windll.dwmapi.DwmSetWindowAttribute
        dwm.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        dwm.restype = ctypes.c_long
        value = ctypes.c_int(1)
        for attr in (20, 19):
            if dwm(hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                break
        SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER, SWP_NOACTIVATE, SWP_FRAMECHANGED = (
            0x1, 0x2, 0x4, 0x10, 0x20)
        _user32().SetWindowPos(hwnd, None, 0, 0, 0, 0,
                               SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE
                               | SWP_FRAMECHANGED)
    except _WIN_ERRORS:
        pass


def flash_taskbar(win):
    """Flash the taskbar button until the window is focused (like Explorer
    when a copy finishes). No-op when the window already has focus."""
    if not IS_WINDOWS:
        return
    try:
        if win.focus_displayof() is not None:
            return
    except (KeyError, RuntimeError):
        pass  # focus owner is a destroyed/foreign widget
    try:
        import ctypes
        from ctypes import wintypes

        class FLASHWINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("hwnd", wintypes.HWND),
                        ("dwFlags", wintypes.DWORD), ("uCount", wintypes.UINT),
                        ("dwTimeout", wintypes.DWORD)]

        FLASHW_ALL, FLASHW_TIMERNOFG = 0x3, 0xC
        info = FLASHWINFO(ctypes.sizeof(FLASHWINFO), _hwnd(win),
                          FLASHW_ALL | FLASHW_TIMERNOFG, 0, 0)
        flash = ctypes.windll.user32.FlashWindowEx
        flash.argtypes = [ctypes.POINTER(FLASHWINFO)]
        flash.restype = wintypes.BOOL
        flash(ctypes.byref(info))
    except _WIN_ERRORS:
        pass


def merge_path(current, machine, user, extra=()):
    """PATH for a refreshed environment: this process's PATH first, unchanged
    and in order, then any machine/user registry entries and extra folders it
    did not have yet. Keeping the process entries first matters: the frozen
    build's runtime hook prepends the bundled torch/lib folder so the bundled
    CUDA DLLs win over a system CUDA install. Duplicates are dropped (case-
    and trailing-slash-insensitive), order is kept."""
    seen = set()
    out = []
    for block in (current, machine, user, os.pathsep.join(extra)):
        for entry in (block or "").split(os.pathsep):
            entry = entry.strip().strip('"')
            if not entry:
                continue
            key = os.path.normcase(entry.rstrip("\\/"))
            if key in seen:
                continue
            seen.add(key)
            out.append(entry)
    return os.pathsep.join(out)


def _registry_path(root, subkey):
    import winreg
    try:
        with winreg.OpenKey(root, subkey) as key:
            value, kind = winreg.QueryValueEx(key, "Path")
    except OSError:
        return ""
    if kind == winreg.REG_EXPAND_SZ:
        value = winreg.ExpandEnvironmentStrings(value)
    return value


def refresh_path():
    """Re-read PATH from the registry into this process.

    A program installed while Scrivox is open (e.g. `winget install ffmpeg`)
    only updates the registry; running processes keep the PATH they started
    with, so "Check again" would never find it. winget's portable-app link
    folder is added too, since it may not be on PATH until the next sign-in.
    Returns True if PATH changed. No-op outside Windows.
    """
    if not IS_WINDOWS:
        return False
    try:
        import winreg
        machine = _registry_path(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
        user = _registry_path(winreg.HKEY_CURRENT_USER, "Environment")
    except (ImportError, OSError):
        return False
    extra = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        links = os.path.join(local, "Microsoft", "WinGet", "Links")
        if os.path.isdir(links):
            extra.append(links)
    old = os.environ.get("PATH", "")
    new = merge_path(old, machine, user, extra)
    if new != old:
        os.environ["PATH"] = new
        return True
    return False


def work_area(win):
    """Return (x, y, width, height) of the usable desktop (minus the taskbar)."""
    if IS_WINDOWS:
        try:
            import ctypes
            from ctypes import wintypes
            rect = wintypes.RECT()
            SPI_GETWORKAREA = 0x0030
            if ctypes.windll.user32.SystemParametersInfoW(
                    SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
                return (rect.left, rect.top,
                        rect.right - rect.left, rect.bottom - rect.top)
        except _WIN_ERRORS:
            pass
    # Elsewhere: whole screen minus a typical panel/taskbar allowance
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    return 0, 0, sw, max(sh - 48, 200)


def open_path(path):
    """Open a file or folder with its default application."""
    try:
        if IS_WINDOWS:
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except _WIN_ERRORS:
        return False


def reveal_in_folder(path):
    """Open the containing folder with `path` selected (Explorer /select)."""
    path = os.path.abspath(path)
    try:
        if IS_WINDOWS and os.path.exists(path):
            subprocess.Popen(["explorer", "/select,", path])
            return True
    except _WIN_ERRORS:
        pass
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    return open_path(folder)
