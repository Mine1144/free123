"""Windows hotkeys without keyboard hooks or administrator privileges."""
import ctypes
import sys
from PySide6.QtCore import QAbstractNativeEventFilter


class Hotkeys(QAbstractNativeEventFilter):
    def __init__(self, toggle, visibility, once):
        super().__init__()
        self.callbacks = {1: toggle, 2: visibility, 3: once}
        self.registered = []
        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            for key_id, vk in ((1, 0x54), (2, 0x48), (3, 0x53)):
                # Ctrl + Alt + T/H/S, no repeat.
                if user32.RegisterHotKey(None, key_id, 0x4003, vk):
                    self.registered.append(key_id)

    def nativeEventFilter(self, event_type, message):
        if sys.platform == "win32":
            from ctypes.wintypes import MSG
            msg = MSG.from_address(int(message))
            if msg.message == 0x0312 and msg.wParam in self.callbacks:
                self.callbacks[msg.wParam]()
                return True, 0
        return False, 0

    def close(self):
        if sys.platform == "win32":
            for key_id in self.registered:
                ctypes.windll.user32.UnregisterHotKey(None, key_id)
