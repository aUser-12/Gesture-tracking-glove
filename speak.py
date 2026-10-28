import pyttsx3
import time


engine = pyttsx3.init()
engine.setProperty('rate', 250)
engine.setProperty('volume', 1.0)


voices = engine.getProperty('voices')
# for i, v in enumerate(voices):
#     print(f"{i}: {v.name}  ({v.id})")

# Pick one (e.g. British English)
engine.setProperty('voice', voices[1].id)




for line in ["You", "Will.", "Obey."]:
    engine.say(line)
    engine.runAndWait()
    time.sleep(0.3)
