import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl

import nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
nltk.download('stopwords')
from nltk.corpus import stopwords
import contractions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam

from tensorflow.python.framework.ops import disable_eager_execution
tf.compat.v1.experimental.output_all_intermediates(True)
disable_eager_execution()
import pickle


from model import make_model




path = "data/"
train_data = pd.read_csv(os.path.join(path, 'train.csv'), names = ['Rating','Title','Review']) #,nrows=10000)
test_data = pd.read_csv(os.path.join(path, 'test.csv'), names = ['Rating','Title','Review'])# , nrows = 1000)


print("\n\n----- SHAPE OF ORIGINAL TRAIN & TEST DATA -----\n")
print(f"TRAIN DATA: {train_data.shape}")
print(f"TEST DATA: {test_data.shape}")

# print(train_data.head())

# Checking for null values
# print(train_data.isnull().sum())

# Since null values are very low as compared to the whole training dataset - we will drop those
train_data = train_data.dropna()
train_data.reset_index(inplace=True, drop=True)



# print(test_data.shape)

#checking for null values
# print(test_data.isnull().sum())

# Since null values are very low as compared to the whole training dataset - we will drop those
test_data = test_data.dropna()
test_data.reset_index(inplace=True, drop=True)


print("\n\n----- SHAPE OF TRAIN & TEST DATA AFTER DROPPING NULL VALUES-----\n")
print(f"TRAIN DATA: {train_data.shape}")
print(f"TEST DATA: {test_data.shape}")

print("\n\n----- PREVIEW OF TRAIN DATA -----\n")
for i in range(5):
    print("Review #",i+1)
    print(train_data.Review[i])
    print(train_data.Title[i])
    print()


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

train_data['Title'] = train_data['Title'].apply(lambda x: clean_matter(x, remove_stopwords = False))
train_data['Review'] = train_data['Review'].apply(lambda x: clean_matter(x, remove_stopwords = True))
train_data['Title'] = train_data['Title'].apply(lambda x : '_START_ '+ x + ' _END_')

test_data['Title'] = test_data['Title'].apply(lambda x: clean_matter(x, remove_stopwords = False))
test_data['Review'] = test_data['Review'].apply(lambda x: clean_matter(x, remove_stopwords = True))
test_data['Title'] = test_data['Title'].apply(lambda x : '_START_ '+ x + ' _END_')


# for i in range(2):
#     print('Title:', train_data['Title'][i],'Review:', train_data['Review'][i], sep='\n')
#     print()

# for i in range(2):
#     print('Title:', test_data['Title'][i],'Review:', test_data['Review'][i], sep='\n')
#     print()


Title_length = [len(x.split()) for x in train_data.Title]
Review_length = [len(x.split()) for x in train_data.Review]

Title_length_test = [len(x.split()) for x in test_data.Title]
Review_length_test = [len(x.split()) for x in test_data.Review]

print("\n\n----- SHAPE OF TRAIN & TEST DATA AFTER CLEANING-----\n")
print(f"TRAIN DATA: {train_data.shape}")
print(f"TEST DATA: {test_data.shape}")



print("\n\n----- LOADING GLOVE 840B 300D WORD EMBEDDINGS-----\n")
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


words_source_ALL = []
for i in train_data['Review'] :
    words_source_ALL.extend(i.split(' '))
for i in test_data['Review'] :
    words_source_ALL.extend(i.split(' '))

print("Total Number of Words: ", len(words_source_ALL))

words_source_ALL = set(words_source_ALL)
print("Unique Words", len(words_source_ALL))


inter_words = set(glove.keys()).intersection(words_source_ALL)
print("No of Words common in Glove and Corpus: {} = {}% ".format(len(inter_words), np.round((float(len(inter_words))/len(words_source_ALL))
*100)))

words_corpus_source_ALL = {}
words_glove = set(glove.keys())
for i in words_source_ALL:
    if i in words_glove:
        words_corpus_source_ALL[i] = glove[i]
print("Word 2 Vec Length", len(words_corpus_source_ALL))


def num(text):
  words = [w for w in text.split() if not w in inter_words]
  return len(words)

train_data['unique'] = train_data['Review'].apply(num)


train_data = train_data[train_data['unique'] < 4]
train_data.reset_index(inplace=True, drop=True)


max_length_x = max(Review_length + Review_length_test)
max_length_y = max(Title_length + Title_length_test)

# test_data.Review =  pd.Series(test_data.Review, dtype="string")
# test_data.Title =  pd.Series(test_data.Title, dtype="string")

# train_data.Review =  pd.Series(train_data.Review, dtype="string")
# train_data.Title =  pd.Series(train_data.Title, dtype="string")


list_sentences = train_data.Review.tolist() + train_data.Title.tolist() + test_data.Review.tolist() + test_data.Title.tolist()

all_sentences = train_data.Review.tolist() + train_data.Title.tolist() + test_data.Review.tolist() + test_data.Title.tolist()

x_t = Tokenizer()
x_t.fit_on_texts(all_sentences)
x_vocab_size = len(x_t.word_index) + 1

encoded_xtrain = x_t.texts_to_sequences(train_data['Review'])
encoded_xtest = x_t.texts_to_sequences(test_data['Review'])

padded_xtrain = pad_sequences(encoded_xtrain, maxlen=max_length_x, padding='post')
padded_xtest = pad_sequences(encoded_xtest, maxlen=max_length_x, padding='post')


all_y_sentences = train_data.Title.tolist() + test_data.Title.tolist()

y_t = Tokenizer()
y_t.fit_on_texts(all_y_sentences)
y_vocab_size = len(y_t.word_index) + 1

encoded_ytrain = y_t.texts_to_sequences(train_data['Title'])
encoded_ytest = y_t.texts_to_sequences(test_data['Title'])

padded_ytrain = pad_sequences(encoded_ytrain, maxlen=max_length_y, padding='post')
padded_ytest = pad_sequences(encoded_ytest, maxlen=max_length_y, padding='post')


print(f'\n\n----- LOADING WORD VECTORS:{len(glove)}')
print()

   
embedding_matrix = np.zeros((x_vocab_size, glove_size))
for word, i in x_t.word_index.items():
    embedding_vector = glove.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Store the important variables for the
vars = [max_length_x, max_length_y, x_vocab_size, y_vocab_size, glove_size, embedding_matrix, y_vocab_size, x_t, y_t]
with open(os.path.join(path, 'vars_new.pkl'), 'wb') as f:
    pickle.dump(vars, f)



model = make_model(max_length_x, x_vocab_size, glove_size, embedding_matrix, y_vocab_size, req = 'Train_Model')
print("\n\n----- MODEL SUMMARY-----\n")
print(model.summary())


model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', experimental_run_tf_function=False)
checkpoint_filepath = os.path.join('model', 'in_progress_model.{epoch:02d}-{val_loss:.2f}.h5')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True, save_freq = "epoch")
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)


history=model.fit([padded_xtrain,padded_ytrain[:,:-1]], padded_ytrain.reshape(padded_ytrain.shape[0],padded_ytrain.shape[1], 1)[:,1:] ,
                      epochs=10,
                      batch_size=128, 
                      validation_split=0.1, 
                      callbacks=[es, model_checkpoint_callback])