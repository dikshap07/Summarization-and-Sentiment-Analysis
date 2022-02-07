from model import make_model
import pickle
import re
import numpy as np
import contractions
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

def decode_sequence(input_seq, max_length_x, target_word_index, reverse_target_word_index, embedding_matrix, x_vocab_size, glove_size, y_vocab_size):
    encoder_model, decoder_model = make_model(max_length_x, x_vocab_size, glove_size, embedding_matrix, y_vocab_size, req = 'Decoder_Model')

    print(input_seq.shape)
    input_seq= input_seq.reshape(1,max_length_x)
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = target_word_index['start']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        if(sampled_token!='end'):
            decoded_sentence += ' '+sampled_token
 
        if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_length_y-1)):
                stop_condition = True

        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        e_h, e_c = h, c

    return decoded_sentence



def clean_matter(matter, x_t, remove_stopwords = True):
    stops = stopwords.words('english')
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
    
    # print(matter)
    matter = x_t.texts_to_sequences(matter)
    matter = list(sum(matter, []))

    matter = pad_sequences([matter], maxlen=max_length_x, padding='post')
    return matter


if __name__ == "__main__":
    with open('data/vars.pkl', 'rb') as f:
        l = pickle.load(f)

    max_length_x, max_length_y, x_vocab_size, y_vocab_size, glove_size, embedding_matrix, y_vocab_size, x_t, y_t = l

    reverse_target_word_index = y_t.index_word 
    reverse_source_word_index = x_t.index_word 
    target_word_index = y_t.word_index

    input_seq = "Start"
    while input_seq != "q":
        input_seq = input('\nEnter \'q\' to Quit OR \nEnter A Review: ')
        if input_seq != 'q':
            cleaned_input = clean_matter(input_seq, x_t)
            summary = decode_sequence(cleaned_input, max_length_x, target_word_index, reverse_target_word_index, embedding_matrix, x_vocab_size, glove_size, y_vocab_size)
            print("\nPredicted summary:", summary)


'''
Sample Input:

book definitely interesting read however bought purpose helping microeconomics class book teaching basic economics opinion piece buyer beware looking educational informational technical book economics one fit bill 

'''