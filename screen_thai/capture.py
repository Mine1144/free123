"""Map Qt logical display devices to MSS physical monitor coordinates."""
import ctypes
import sys


def windows_displays():
    from ctypes import wintypes

    class MonitorInfo(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                    ("szDevice", wintypes.WCHAR * 32)]

    displays = {}
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                                      ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MonitorInfo)]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    user32.EnumDisplayMonitors.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                                          callback_type, wintypes.LPARAM]
    user32.EnumDisplayMonitors.restype = wintypes.BOOL

    @callback_type
    def visit(handle, dc, rect, data):
        info = MonitorInfo()
        info.cbSize = ctypes.sizeof(info)
        if user32.GetMonitorInfoW(handle, ctypes.byref(info)):
            r = info.rcMonitor
            displays[info.szDevice.upper()] = dict(left=r.left, top=r.top,
                                                  width=r.right-r.left, height=r.bottom-r.top)
        return True

    if not user32.EnumDisplayMonitors(None, None, visit, 0):
        raise RuntimeError("อ่านข้อมูลจอภาพจาก Windows ไม่สำเร็จ")
    return displays


def monitor_for_screen(screen, monitors):
    if sys.platform == "win32":
        displays = windows_displays()
        geometry = screen.geometry()
        native_width = round(geometry.width() * screen.devicePixelRatio())
        native_height = round(geometry.height() * screen.devicePixelRatio())
        candidates = [m for m in displays.values()
                      if m["left"] == geometry.x() and m["top"] == geometry.y()
                      and abs(m["width"]-native_width) <= 2
                      and abs(m["height"]-native_height) <= 2]
        if len(candidates) == 1:
            return candidates[0]
        match = displays.get(screen.name().upper())
        if match and abs(match["width"]-native_width) <= 2 and abs(match["height"]-native_height) <= 2:
            return match
        # Never silently capture an unrelated monitor on a multi-monitor system.
        raise ValueError("จับคู่จอ Windows ไม่ได้ กรุณาตรวจ scaling แล้วเปิดโปรแกรมใหม่")
    # Development-only fallback on Linux. Windows is the supported deployment target.
    geometry = screen.geometry()
    for monitor in monitors:
        if monitor["left"] == geometry.x() and monitor["top"] == geometry.y():
            return dict(monitor)
    if len(monitors) == 1:
        return dict(monitors[0])
    raise ValueError("ไม่พบจอภาพที่เลือก")


def region_bounds(monitor, region):
    left = round(monitor["width"] * region.x())
    top = round(monitor["height"] * region.y())
    right = round(monitor["width"] * (region.x() + region.width()))
    bottom = round(monitor["height"] * (region.y() + region.height()))
    left, top = max(0, left), max(0, top)
    right, bottom = min(monitor["width"], right), min(monitor["height"], bottom)
    if right <= left or bottom <= top:
        raise ValueError("พื้นที่จับภาพว่าง")
    return {"left": monitor["left"] + left, "top": monitor["top"] + top,
            "width": right-left, "height": bottom-top}
