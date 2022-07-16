
import numpy as np
import pickle
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_path = "sentiment_model.h5"
tokenizer_path = "tokenizer.pkl"



def get_sentiment(sentence):

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f) # load tokenizer
        f.close()
    model = load_model(model_path) # load model
    max_len = 150

    sentence = simple_preprocess(sentence, deacc=True)
    porter_stemmer = PorterStemmer()
    sentence = [[porter_stemmer.stem(word) for word in sentence]]
    input_sequence = tokenizer.texts_to_sequences(sentence)
    input_padded = pad_sequences(input_sequence, maxlen=max_len)
    pred = np.round(model.predict(input_padded))
    if pred == 1:
        return "Positive"
    else:
        return "Negative"


# user_input = """California is a great state to live in, I just love it!!! 
#                 Not only it is very large, but there are many people who live in the city. 
#                 We can also find different types of jobs around the city."""
# print(get_sentiment(user_input))