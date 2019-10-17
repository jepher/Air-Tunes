import cv2
import numpy as np
import math
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pygame
from pygame import mixer
import keyboard
import time
import tkinter as tk
from tkinter import messagebox
import threading

c = 0

cap = cv2.VideoCapture(0)

mixer.init()
model= load_model("dabs_weighting.model")
    
class MusicPlayer(object):
    """Plays Different sounds based on gesture given.

    Attributes:
        none so far
    """

    def __init__(self):
        """Return a MusicPlayer object."""
        self.sprinkle = mixer.Sound("res/sprinkle2.wav")
        self.scratch = mixer.Sound("res/scratch2.wav")
        self.drop = mixer.Sound("res/DROP_2.wav")
        self.clap = mixer.Sound("res/CLAP_1.wav")
        self.clap2 = mixer.Sound("res/CLAP_2.wav")
        self.kick = mixer.Sound("res/KICK_1.wav")
        self.glass = mixer.Sound("res/GLASS_1.wav")
        self.glass2 = mixer.Sound("res/GLASS_2.wav")
        self.hulk = mixer.Sound("res/hulk.wav")
        
    def PlaySound(self, sound_num):
        if sound_num == 0:
            self.clap.play()    
        if sound_num == 1:
            self.clap2.play()    
        if sound_num == 2:
            self.kick.play()     
        if sound_num == 3:
            self.kick.play()      
        if sound_num == 4:
            self.glass.play()     
        if sound_num == 5:
            self.glass2.play()     
        if sound_num == 6:
            self.drop.play()     
        if sound_num == 7:
            self.scratch.play()
        if sound_num == 8:
            self.sprinkle.play()
        if sound_num == 10:
            self.hulk.play()
            
mMusicPlayer = MusicPlayer()

def PlayLoop(sound):
    threading.Timer(1.846, PlayLoop, [sound]).start()
    mMusicPlayer.PlaySound(sound)
    print ("Playing...")
    
previousNum = 0;
newNumCount = 0;
PlayLoop(10)
while(1):
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    kernel = np.ones((3,3),np.uint8)
    
    roi=frame[0:900, 0:900]
    
    cv2.rectangle(frame,(0,0),(900,900),(0,255,0),0)    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    mask = cv2.erode(mask,kernel,iterations = 1)

    mask = cv2.GaussianBlur(mask,(5,5),200) 

    ret,thresh = cv2.threshold(mask, 70, 255, 0)

    cv2.imshow('mask',mask)


    mask = cv2.resize(mask, (80, 60))
    #mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    mask = mask.astype("float") / 255.0
    mask = img_to_array(mask)
    mask = np.expand_dims(mask, axis=0)
    
    (none, dabL, dabR, hitL, hitR, whipL, whipR, tPose, shoot, keke) = model.predict(mask)[0]
    values = [none, dabL, dabR, hitL, hitR, whipL, whipR, tPose, shoot, keke]
    currentNum = values.index(max(values))

    if(currentNum == previousNum):
        if(newNumCount < 7):
            newNumCount = newNumCount + 1
        print (str(newNumCount) + " occurences of " + str(currentNum))
    else:
        newNumCount = 0
        previousNum = currentNum
        print ("New number: " + str(currentNum) + ", " + str(previousNum))
        
    if(newNumCount == 3):
        print ("Play")
        if currentNum == 1:
            mMusicPlayer.PlaySound(0)
        if currentNum == 2:
            mMusicPlayer.PlaySound(1)
        if currentNum == 3:
            mMusicPlayer.PlaySound(2)
        if currentNum == 4:
            mMusicPlayer.PlaySound(3)
        if currentNum == 5:
            mMusicPlayer.PlaySound(4)
        if currentNum == 6:
            mMusicPlayer.PlaySound(5)
        if currentNum == 7:
            mMusicPlayer.PlaySound(6)
        if currentNum == 8:
            mMusicPlayer.PlaySound(7)
        if currentNum == 9:
            #break
            mMusicPlayer.PlaySound(0)
    #cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
       break
  
cv2.destroyAllWindows()
cap.release()    
    



