
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

with open('data/senti_vars.pkl', 'rb') as f:
        l = pickle.load(f)

vocab_size, glove_size, embedding_matrix, tokenizer, sentiment_label = l


def create_model():
    model = Sequential() 
    model.add(Embedding(vocab_size, glove_size, input_length=200, weights=[embedding_matrix], trainable=False) )
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.4))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu')) 
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
    # print(model.summary())
    return model


def predict_sentiment(text, model):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])
    

model_path = "model/senti_model.h5"
model = create_model()
model.load_weights(model_path)
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 

test_sentence1 = "I like this book."
predict_sentiment(test_sentence1, model)

input_seq = "Start"
while input_seq != "q":
    input_seq = input('\nEnter \'q\' to Quit OR \nEnter A Review: ')
    if input_seq != 'q':
        predict_sentiment(input_seq, model)