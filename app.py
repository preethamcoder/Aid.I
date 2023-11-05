import numpy as np
from flask import Flask, render_template, request
import time
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
import json
from keras.models import load_model
import random

model = load_model('model_500_new_negation.h5')
intents_fr_og = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
get_lemmas = WordNetLemmatizer()

def cleanup(s):
    s_words = nltk.word_tokenize(s)
    s_words = [get_lemmas.lemmatize(word.lower()) for word in s_words]
    print(s, s_words)
    return s_words

def bag_of_words(s, words, details):
    s_words = cleanup(s)
    bag = [0]*len(words)
    for sen in s_words:
        for each, w in enumerate(words):
            if w == sen:
                bag[each] = 1
                if details:
                    print(f"found in bag: {w}")
    return(np.array(bag))

def get_class(sentence, model):
    bow = bag_of_words(sentence, words, True)
    res = model.predict(np.array([bow]))[0]
    E_T = 0.25
    rs = [[each, r_e] for each, r_e in enumerate(res) if r_e>E_T]
    rs.sort(key=lambda x: x[1], reverse=True)
    ret = []
    for res in rs:
        ret.append({"intent": classes[res[0]], "probability": str(res[1])})
    return ret

def getjawaab(intents_fr, intents_json):
    print(intents_fr)
    # Okay, I don't know why this does not work
    time.sleep(1.1)
    # Just making this more human
    if not intents_fr:
        return "Not sure why you said what you said. Want to explain?"
    tag = intents_fr[0]['intent']
    intents_list = intents_json['intents']
    for each in intents_list:
        if(each['tag']== tag):
            res = random.choice(each['responses'])
            break
    return res

def cbot_response(message):
    intents_fr = get_class(message, model)
    res = getjawaab(intents_fr, intents_fr_og)
    return res

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_resp():
    uText = (request.args.get('msg')).lower()
    return cbot_response(uText)

if __name__ == "__main__":
    app.run()
