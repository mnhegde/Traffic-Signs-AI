from gtts import gTTS
from playsound import playsound
import os 

language = 'en'

def playString(string):
    global language
    filename = 'sounds/audio.mp3'
    audio = gTTS(text=string, lang=language, slow=False)
    audio.save(filename)
    playsound(filename)
    os.remove(filename)


  

