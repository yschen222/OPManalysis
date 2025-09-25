# ===== notifier.py =====
# Cross-platform "job done" notifier for long-running scripts.
# It tries (in order): desktop toast → native popup (Windows) → beep/voice → stdout fallback.
# Usage:
#   from notifier import run_with_notify, notify_done
#   run_with_notify(main_function, arg1, arg2, title="OPM Batch Task")
#   # or call notify_done("All finished ✅") at the end of your script.

from __future__ import annotations

import os
import sys
import time
import platform
import subprocess
from pathlib import Path
from typing import Any, Callable

__all__ = ["notify_done", "run_with_notify"]

# ----------------------------
# Low-level notifiers (private)
# ----------------------------

def _beep_fallback(times: int = 2, freq: int = 1000, dur_ms: int = 500) -> bool:
    """
    Try to make a sound as a last resort.
    Windows: winsound.Beep; macOS: 'say "job finished"' (can replace with '\a');
    Linux/other: terminal bell.
    Returns True if any method succeeded.
    """
    try:
        sysname = platform.system()
        if sysname == "Windows":
            import winsound  # type: ignore
            for _ in range(times):
                winsound.Beep(freq, dur_ms)
                time.sleep(0.15)
            return True
        elif sysname == "Darwin":
            # Text-to-speech; if you prefer terminal bell, use: sys.stdout.write('\\a')
            os.system('say "job finished"')
            return True
        else:
            # Terminal bell on POSIX
            sys.stdout.write("\a" * times)
            sys.stdout.flush()
            return True
    except Exception:
        pass
    return False


def _toast_windows(title: str, msg: str, duration: int = 5) -> bool:
    """
    Windows toast via win10toast (pip install win10toast).
    Returns True on success, False otherwise.
    """
    try:
        from win10toast import ToastNotifier  # type: ignore
        ToastNotifier().show_toast(title, msg, duration=duration, threaded=True)
        return True
    except Exception:
        return False


def _toast_plyer(title: str, msg: str) -> bool:
    """
    Cross-platform desktop notification via plyer (pip install plyer).
    Returns True on success, False otherwise.
    """
    try:
        from plyer import notification  # type: ignore
        notification.notify(title=title, message=msg, timeout=5)
        return True
    except Exception:
        return False


def _toast_linux(title: str, msg: str) -> bool:
    """
    Linux notifications via 'notify-send' (usually available on desktop distros).
    Returns True on success, False otherwise.
    """
    try:
        subprocess.Popen(["notify-send", title, msg])
        return True
    except Exception:
        return False


def _popup_windows(title: str, msg: str) -> bool:
    """
    Native Windows message box (no extra packages needed).
    Returns True on success, False otherwise.
    """
    try:
        import ctypes  # type: ignore
        ctypes.windll.user32.MessageBoxW(0, msg, title, 0x40)  # MB_ICONINFORMATION
        return True
    except Exception:
        return False


# ----------------------------
# Public API
# ----------------------------

def notify_done(msg: str = "Task completed ✅", title: str = "OPM Batch Task") -> None:
    """
    Fire a completion notification with graceful degradation.
    Priority:
      1) Desktop toast (win10toast/plyer/notify-send)
      2) Native popup (Windows only)
      3) Beep / voice (platform dependent)
      4) Stdout fallback
    """
    ok = False
    sysname = platform.system()

    # 1) Desktop notifications (preferred)
    if sysname == "Windows":
        ok = _toast_windows(title, msg) or _toast_plyer(title, msg)
    elif sysname == "Darwin":
        # Prefer plyer; fall back to AppleScript if plyer not available.
        ok = _toast_plyer(title, msg)
        if not ok:
            try:
                script = f'display notification "{msg}" with title "{title}"'
                subprocess.Popen(["osascript", "-e", script])
                ok = True
            except Exception:
                ok = False
    else:
        # Linux / other POSIX
        ok = _toast_plyer(title, msg) or _toast_linux(title, msg)

    # 2) Windows native popup (second chance)
    if not ok and sysname == "Windows":
        ok = _popup_windows(title, msg)

    # 3) Beep / voice
    if not ok:
        ok = _beep_fallback(times=2)

    # 4) Final stdout fallback
    if not ok:
        print(f"\n=== {title} ===\n{msg}\n")


def run_with_notify(
    func: Callable[..., Any],
    *args: Any,
    title: str = "OPM Batch Task",
    **kwargs: Any,
) -> Any:
    """
    Wrap a callable so it always sends a completion notification.
    On success: "completed" notification. On failure: "failed" notification with the exception message.
    Re-raises the exception so your upstream error handling still works.

    Example:
        result = run_with_notify(process_root, Path('path/to/root'), title="OPM Batch Task")
    """
    try:
        result = func(*args, **kwargs)
        notify_done("Finished successfully ✅", title=title)
        return result
    except Exception as e:
        notify_done(f"Failed ❌: {e}", title=title)
        raise
