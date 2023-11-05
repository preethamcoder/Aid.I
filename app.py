from flask import Flask, render_template, request
import time
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

model = load_model('model_500_new_negation.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    print(sentence, sentence_words)
    return sentence_words

def bag_of_words(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for each,w in enumerate(words):
            if w == s: 
                bag[each] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return(np.array(bag))

def predict_class(sentence, model):
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    E_T = 0.25
    results = [[each,r] for each,r in enumerate(res) if r>E_T]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    print(ints)
    time.sleep(1.1)
    if not ints:
        return "Not sure why you said what you said. Want to explain?"
    tag = ints[0]['intent']
    intents_list = intents_json['intents']
    for each in intents_list:
        if(each['tag']== tag):
            res = random.choice(each['responses'])
            break
    return res

def cbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
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