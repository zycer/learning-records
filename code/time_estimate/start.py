import ctypes
import inspect
import time
from threading import Thread
import os
import win32api
import win32gui
import win32process
from utils.db_manager import DBManager
import listening_task
import win32con
import pyperclip


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


def show_menu():
    print()
    print('-' * 30)
    print("1：创建进程")
    print("2：已完成数据量")
    print("3：已匹配道路量")
    print("4：持久化数据")
    print("0：退出")
    print('-' * 30)


def window_enum_callback(hwnd, mouse):
    if (win32gui.IsWindow(hwnd) and
            win32gui.IsWindowEnabled(hwnd) and
            win32gui.IsWindowVisible(hwnd)):
        hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})


def start_cmd(process_num):
    for num in range(process_num):
        win32process.CreateProcess(None, 'cmd', None, None, True,
                                   win32process.CREATE_NEW_CONSOLE, None, None, win32process.STARTUPINFO())
        time.sleep(1)
        win32api.keybd_event(17, 0, 0, 0)
        win32api.keybd_event(86, 0, 0, 0)
        win32api.keybd_event(86, 0, win32con.KEYEVENTF_KEYUP, 0)  # 松开按键
        win32api.keybd_event(17, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(13, 0, 0, 0)
        win32api.keybd_event(13, 0, win32con.KEYEVENTF_KEYUP, 0)


def execute(flag):
    if flag == "1":
        pyperclip.copy("python main.py")
        print("1：自动创建")
        print("2：手动创建")
        flag2 = input("请输入：")
        if flag2 == "1":
            process_num = os.cpu_count()
        else:
            while True:
                try:
                    process_num = int(input("请输入创建进程数："))
                    break
                except Exception:
                    print("输入错误，请重新输入")
        start_cmd(process_num)
        print(f"创建完成<{process_num}>")

    elif flag == "2":
        result = db_handler.exec_sql("SELECT num, file_name from finish_flag")
        for res in result:
            print(f"{res[1]}: {res[0]}")
    elif flag == "3":
        result = db_handler.exec_sql("SELECT COUNT(*) from history_road_data")
        print("已匹配道路数量：", result[0][0])
    elif flag == "4":
        pyperclip.copy("python database_exec.py")
        start_cmd(1)

    elif flag == "0":
        win32gui.EnumWindows(window_enum_callback, 0)
        for key, value in hwnd_title.items():
            print(key, value)
            if "cmd.exe" in value:
                win32api.PostMessage(key, win32con.WM_CLOSE, 0, 0)
        stop_thread(listener)
        exit()


if __name__ == '__main__':
    hwnd_title = {}
    db_handler = DBManager()
    listener = Thread(target=listening_task.listening_task)
    listener.start()

    while True:
        show_menu()
        in_context = input("请输入：")
        execute(in_context)
