import copy
import re
import time

import pynput
import pywintypes
import win32api
import win32con
import win32gui
import win32com.client
import pyperclip
import pythoncom

from pynput import keyboard


class WindowMgr:

    def __init__(self, title):
        self._handle = None
        self._hwnd_title = {}
        self.title = title

    def find_window(self, class_name, window_name=None):
        self._handle = win32gui.FindWindow(class_name, window_name)

    def _window_enum_callback(self, hwnd, mouse):
        if (win32gui.IsWindow(hwnd) and
                win32gui.IsWindowEnabled(hwnd) and
                win32gui.IsWindowVisible(hwnd)):
            self._hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})

        for hwnd, title in self._hwnd_title.items():
            if self.title == title:
                self._handle = hwnd
                break

    def find_window_wildcard(self):
        self._handle = None
        win32gui.EnumWindows(self._window_enum_callback, 0)

    def send_key_event(self):
        self._hwnd_title.clear()
        self.find_window_wildcard()
        pythoncom.CoInitialize()
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%')

        try:
            win32gui.SetForegroundWindow(self._handle)
            win32gui.SetActiveWindow(self._handle)
        except pywintypes.error as e:
            win32api.keybd_event(17, 0, 0, 0)
            win32api.keybd_event(121, 0, 0, 0)
            win32api.keybd_event(121, 0, win32con.KEYEVENTF_KEYUP, 0)  # 松开按键
            win32api.keybd_event(17, 0, win32con.KEYEVENTF_KEYUP, 0)

            time.sleep(0.1)
            self.find_window_wildcard()
            win32gui.SetForegroundWindow(self._handle)
            win32gui.SetActiveWindow(self._handle)

        time.sleep(0.1)
        win32api.keybd_event(17, 0, 0, 0)
        win32api.keybd_event(86, 0, 0, 0)
        win32api.keybd_event(86, 0, win32con.KEYEVENTF_KEYUP, 0)  # 松开按键
        win32api.keybd_event(17, 0, win32con.KEYEVENTF_KEYUP, 0)


class KeyBoard:
    def __init__(self, key_board_value, exec_func):
        self.key_board_value = key_board_value
        self.exec_func = exec_func
        self.count = 0

    def start_listen(self):
        def for_canonical(f):
            return lambda k: f(l.canonical(k))

        hotkey = keyboard.HotKey(
            keyboard.HotKey.parse(self.key_board_value), self.on_activate)

        with keyboard.Listener(
                on_press=for_canonical(hotkey.press),
                on_release=for_canonical(hotkey.release)) as l:
            l.join()

    def on_activate(self):
        time.sleep(0.2)
        win32api.keybd_event(17, 0, 0, 0)
        win32api.keybd_event(67, 0, 0, 0)
        win32api.keybd_event(67, 0, win32con.KEYEVENTF_KEYUP, 0)  # 松开按键
        win32api.keybd_event(17, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.1)
        str_prev = pyperclip.paste()
        str_next = re.sub(r"\s{1,2}", " ", str_prev)
        new_str = ""
        for idx, cha in enumerate(str_next):
            if cha == ' ' and idx > 0:
                if str_next[idx - 1] == '-':
                    continue
            new_str += cha

        try:
            print("<%s>: %s" % (self.count, new_str))
        except Exception as e:
            print(e)

        pyperclip.copy(new_str)
        self.count += 1

        if callable(self.exec_func):
            self.exec_func()


class Mouse:
    def __init__(self):
        self.x = None
        self.y = None
        self.count = 0

    def on_click(self, x, y, button, pressed):
        if button.name == "left":
            if not pressed:
                if (self.x, self.y) != (x, y):
                    win32api.keybd_event(17, 0, 0, 0)
                    win32api.keybd_event(67, 0, 0, 0)
                    win32api.keybd_event(67, 0, win32con.KEYEVENTF_KEYUP, 0)  # 松开按键
                    win32api.keybd_event(17, 0, win32con.KEYEVENTF_KEYUP, 0)

                    time.sleep(0.1)

                    str_prev = pyperclip.paste()
                    str_next = re.sub(r"\s{1,2}", " ", str_prev)
                    new_str = ""
                    for idx, cha in enumerate(str_next):
                        if cha == ' ' and idx > 0:
                            if str_next[idx - 1] == '-':
                                continue
                        new_str += cha

                    try:
                        print("<%s>: %s" % (self.count, new_str))
                    except Exception as e:
                        print(e)

                    pyperclip.copy(new_str)
                    self.count += 1

                    self.send_event()

            else:
                self.x, self.y = copy.deepcopy((x, y))

    def send_event(self):
        w.send_key_event()

    def start_listen(self):
        with pynput.mouse.Listener(
                on_click=self.on_click) as listener:
            listener.join()



if __name__ == "__main__":
    w = WindowMgr("HuaweiTranslateWindow")

    k = KeyBoard('<alt>+d', w.send_key_event)
    k.start_listen()

    # m = Mouse()
    # m.start_listen()
