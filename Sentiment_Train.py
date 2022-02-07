import pandas as pd
import numpy as np
import sys
import re
import os
import matplotlib.pyplot as plt
import nltk
import pickle
nltk.download('stopwords')
from nltk.corpus import stopwords
import contractions

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = "data"
train_data = pd.read_csv(os.path.join(path, 'train.csv'), names = ['Rating','Title','Review'],nrows=50000)
test_data = pd.read_csv(os.path.join(path, 'test.csv'), names = ['Rating','Title','Review'], nrows = 5000)

data = pd.concat([train_data, test_data])
data = data.reset_index(drop=True)

data=data.drop(columns=['Title'])

data.Rating = data.Rating.replace([1,2,3],0)
data.Rating = data.Rating.replace([4,5],1)


my_data=data[:]

sentiment_label = my_data.Rating.factorize()
sentiment_label

stops = stopwords.words('english')
def clean_matter(matter, remove_stopwords = True, stops = stops):
    # Convert words to lower case
    matter = str(matter)
    matter = matter.lower()
    
    # Replace contractions with their longer forms 
    matter = ' '.join([contractions.fix(word) for word in matter.split(" ")])    
    
    # Format words and remove unwanted characters
    matter = re.sub(r'https?:\/\/.*[\r\n]*', '', matter, flags=re.MULTILINE)
    matter = re.sub(r'\<a href', ' ', matter)
    matter = re.sub(r'&amp;', '', matter) 
    matter = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', matter)
    matter = re.sub(r'<br />', ' ', matter)
    matter = re.sub(r'\'', ' ', matter)
    
    # Optionally, remove stop words
    if remove_stopwords:
        matter = matter.split()
        matter = [w for w in matter if not w in stops]
        matter = " ".join(matter)

    return matter


review_texts = my_data.Review.apply(clean_matter)
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(review_texts)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(review_texts)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)


glove_size = 300

# # Method 1
# with open(os.path.join(path, 'glove.840B.300d.pkl'), 'rb') as fp:
#     glove = pickle.load(fp)

# Method 2
f = open(os.path.join(path, 'glove.840B.300d.txt'), encoding='utf-8')
glove = dict()
i = 1
for line in f:
    values = line.split(" ")
    if i < 5:
        print(values)
        i = i + 1
    glove[values[0]] = np.asarray(values[1:], dtype='float32')
f.close()


print(f'LOADED {len(glove)} WORD VECTORS.')

embedding_matrix = np.zeros((vocab_size, glove_size))
for word, i in tokenizer.word_index.items():
    embedding_vector = glove.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = Sequential() 
model.add(Embedding(vocab_size, glove_size, input_length=200, weights=[embedding_matrix], trainable=False) )
model.add(SpatialDropout1D(0.4))
model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.4))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model.summary())

checkpoint_filepath = 'model/senti_model.{epoch:02d}-{val_loss:.2f}.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath = checkpoint_filepath,
                                save_weights_only = True,
                                monitor = 'val_loss', 
                                mode = 'min',
                                save_best_only = True, 
                                save_freq = "epoch")

es = EarlyStopping( monitor = 'val_loss', 
                    mode = 'min', 
                    verbose = 1, 
                    patience = 1)
history = model.fit( padded_sequence, sentiment_label[0],
                    validation_split = 0.25,
                    epochs = 15,
                    batch_size=32,
                    callbacks = [es, model_checkpoint_callback])


l = [vocab_size, glove_size, embedding_matrix]
with open('data/senti_vars_new.pkl', 'rb') as f:
    pickle.dump(l, f)
