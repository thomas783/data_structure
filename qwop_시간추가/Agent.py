import win32api
import win32con
import time
import numpy as np
import pyautogui

VK_CODE = {'q': 0x51, 'w': 0x57, 'o': 0x4F, 'p': 0x50, 'R':0x20}

class Agent:

    def __init__(self):
        print("Agent Made")
        super().__init__()

    def n(self):
        return

    def q(self,t):
        win32api.keybd_event(0x51, 0, 0, 0)
        time.sleep(t)
        win32api.keybd_event(0x51, 0, win32con.KEYEVENTF_KEYUP, 0)

    def w(self,t):
        win32api.keybd_event(0x57, 0, 0, 0)
        time.sleep(t)
        win32api.keybd_event(0x57, 0, win32con.KEYEVENTF_KEYUP, 0)

    def o(self,t):
        win32api.keybd_event(0x4F, 0, 0, 0)
        time.sleep(t)
        win32api.keybd_event(0x4F, 0, win32con.KEYEVENTF_KEYUP, 0)

    def p(self,t):
        win32api.keybd_event(0x50, 0, 0, 0)
        time.sleep(t)
        win32api.keybd_event(0x50, 0, win32con.KEYEVENTF_KEYUP, 0)

    def a(self,t):
        pyautogui.keyDown('q')
        pyautogui.keyDown('p')
        time.sleep(t)
        pyautogui.keyUp('q')
        pyautogui.keyUp('p')

    def s(self,t):
        pyautogui.keyDown('w')
        pyautogui.keyDown('o')
        time.sleep(t)
        pyautogui.keyUp('w')
        pyautogui.keyUp('o')

    def space(self):
        win32api.keybd_event(0x20, 0, 0, 0)
        time.sleep(0.1)
        win32api.keybd_event(0x20, 0, win32con.KEYEVENTF_KEYUP, 0)

    def reload(self):
        win32api.keybd_event(0x52, 0, 0, 0)
        time.sleep(0.1)
        win32api.keybd_event(0x52, 0, win32con.KEYEVENTF_KEYUP, 0)
