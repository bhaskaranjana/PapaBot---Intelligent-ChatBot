import random
import json
import pickle
import numpy as np
import tkinter
from tkinter import *
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemet = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('papa_bot.h5')

def cleanupsent(sent):
    sent_words = nltk.word_tokenize(sent)
    sent_words = [lemet.lemmatize(word.lower()) for word in sent_words]
    return sent_words


def bag_o_words(sent):
    sent_words = cleanupsent(sent)
    bag = [0] * len(words)
    for s in sent_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return (np.array(bag))


def predclass(sent):
    bow = bag_o_words(sent)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    returnlist = []
    for r in results:
        returnlist.append({'intents': classes[r[0]], 'probability': str(r[1])})
    return returnlist


def hear_back(intents_list, intents_json):
    tag = intents_list[0]['intents']
    list_o_intents = intents_json['intents']
    for i in list_o_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def papa_sayback(quest):
    ints = predclass(quest)
    res = hear_back(ints, intents)
    return res

#while True: #obsolete on addition of GUI
#    quest = input("")
#    ints = predclass(quest)
#    res = hear_back(ints, intents)
#    print(res)

print("Papa is here son!")

# GUI CODE
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        res = papa_sayback(msg)
        ChatLog.insert(END, "Papa: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
base = Tk()
base.title("PapaBot - Papa is here son!")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", )
ChatLog.config(state=DISABLED)
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="                            Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
SendButton.place(x=128, y=401, height=90, width=265)
EntryBox.place(x=6, y=401, height=90)
base.mainloop()















