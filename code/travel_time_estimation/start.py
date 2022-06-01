from multiprocessing import Process
from threading import Thread
import os
import subprocess as sp
from subprocess import PIPE, STDOUT
import win32api
import win32process
from utils.db_manager import DBManager
import listening_task


def show_menu():
    print()
    print('-'*30)
    print("1：创建进程")
    print("2：已完成数据量")
    print("3：已匹配道路量")
    print("0：退出")
    print('-' * 30)


def execute(flag):
    if flag == "1":
        p = sp.Popen(["python", "main.py"], shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        p.poll()
        print(p.pid)
        # print(p.communicate()[0])
        # print(p.communicate()[1])
        # commands = "python main.py"
        # out, errs = p.communicate(commands.encode("gbk"))
        # print(out)
        # print(errs)
    elif flag == "2":
        result = db_handler.exec_sql("SELECT num, file_name from finish_flag")
        for res in result:
            print(f"{res[1]}: {res[0]}")
    elif flag == "3":
        result = db_handler.exec_sql("SELECT COUNT(*) from history_road_data")
        print("已匹配道路数量：", result[0][0])
    elif flag == "0":
        exit(-1)


if __name__ == '__main__':
    # db_handler = DBManager()
    # listener = Thread(target=listening_task.listening_task)
    # listener.start()
    #
    # while True:
    #     show_menu()
    #     in_context = input("请输入：")
    #     execute(in_context)

    # a = win32api.ShellExecute(0, 'open', r'C:\Windows\system32\cmd.exe', '', '..\\', 1)
    # print(a)
    a = win32process.CreateProcess(None,'cmd',None,None,True,
                                   win32process.CREATE_NEW_CONSOLE,None,None,win32process.STARTUPINFO())
    print(a)




