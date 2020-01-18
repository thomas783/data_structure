from Agent import Agent
import numpy as np
import logging
import time
import pytesseract
import cv2
import re
from PIL import ImageGrab
from PIL import Image
from ctypes import windll
from numpy.lib.stride_tricks import as_strided
user32 = windll.user32
user32.SetProcessDPIAware()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

class Game:

    def __init__(self):
        self.agent = Agent()
        self.game_steps = 0

    def start(self):
        self.agent.space()
        self.game_steps = 0
        return self.execute_action('n',0)

    def reload(self):
        self.game_steps = 0
        self.agent.reload()
        self.agent.space()
        return self.execute_action('n',0)

    def execute_action(self, action, time):
        self.agent.space()
        self.game_steps += 1
        #self.agent.unpause()
        #'a','s','q','w','o','p','t
        if action == 'a':
            self.agent.a(time)
        elif action == 's':
            self.agent.s(time)
        elif action == 'q':
            self.agent.q(time)
        elif action == 'w':
            self.agent.w(time)
        elif action == 'o':
            self.agent.o(time)        
        else :
            self.agent.p(time)
        
        shot = ImageGrab.grab([505, 225, 1195, 1025])#850,900 -> 690:800
        img = np.array(shot)[:,:,0]
        img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
        shot = img
        #self.agent.pause()
        done = self.is_done(shot)
        score = 0.0
        if done:
            distance_score = self.get_score()[1]
            time_score = ((abs(distance_score)/self.game_steps)) # The higher the pace, the slowest it goes
            score = distance_score + time_score
            self.reload()
        return shot.astype(np.float).ravel(), score, done

    def is_done(self, shot):
        return self.get_score()[0]

    def get_score(self):
        raw = pytesseract.image_to_string(ImageGrab.grab([600, 40, 1200, 160]))
        current_score = ""
        for i in raw:
            if (i == 'm') | (i == 'e') | (i == 't') | (i == 'r') | (i == 'e') | (i == 's'):
                continue
            else:
                current_score = current_score + i
        try:
            float(current_score)
            current_score=float(current_score)
        except:
            current_score=1.8
        #print("score:",current_score)
        tmp = ImageGrab.grab([900,380,1000,400])
        r, g, b = tmp.getpixel((0, 0))
        game_over = False
        if r == 237 and g == 237 and b == 237:
            game_over = True
        return [game_over,current_score]
        