import pandas as pd
"""
from string import punctuation 
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import os
from keras.models import load_model
from os.path import dirname, join, realpath
import joblib
import sklearn
"""
from fastapi.encoders import jsonable_encoder
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="API du modèle d'analyse de sentiment",
    description="Une API simple qui utilise un modèle NLP pour prédire le sentiment des commentaires du réseau social.",
    version="0.1",
)

origins = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


"""
model = load_model('models/sentimentsmodelnlp.h5')


with open(
join(dirname(realpath(__file__)), "models/sentiment_naives_bayes_classifier_v3.pkl"), "rb"
) as f:
    model1 = joblib.load(f)



def text_cleaner(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    # Return a list of words
    return text

@app.get("/predict-review")
def predict_sentiment(review: str):
    #A simple function that receive a review content and predict the sentiment of the content.
    #:param review:
    #:return: prediction, probabilities
    # clean the review
    cleaned_review = text_cleaner(review)
    
    # perform prediction
    prediction = model1.predict([cleaned_review])
    output = int(prediction[0])
    probas = model1.predict_proba([cleaned_review])
    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}
    
    # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result


def text_cleaning(data):
    data = pd.DataFrame([data], columns=['text'])

    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X)
    
    return X
"""

@app.get("/")
async def root():
    return {"message": "BONJOUR TOUT LE MONDE"}

@app.get("/publications")
async def all_publication():
    df_pub = pd.read_hdf('data/post.h5')
    pub = []
    for p in df_pub.iloc:
        pub.append(p)
    return pub

@app.get("/commentaires")
async def all_commentaire():
    df_com = pd.read_hdf('data/commentaire.h5')
    com = []
    for c in df_com.iloc:
        com.append(c)
    return com

"""
@app.get("/predire_sentiment")
async def predict_sentiment(commentaire: str):
    #Une fonction simple qui reçoit un commentaire et prédit le sentiment du commentaire.
    #:param commentaire :
    #:return : prédiction, probabilités

        # nettoyer les textes
    commentaire_nettoyer = text_cleaning(commentaire)
    prediction = model.predict([commentaire_nettoyer])
    prediction=prediction[0]
    #print(type(prediction[0]))
    #print(prediction)
    # dictionnaire de sorti
    
    if(prediction[0] > prediction[1]):
        return jsonable_encoder({
            "prediction": 'negative',
            "Probability": float(prediction[0])
            })
    else:
        return jsonable_encoder({
            "prediction": 'positive', 
            "Probability": float(prediction[1])
            })

"""
    
if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, debug=True, reload=True)