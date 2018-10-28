import numpy as np 
import pandas as pd 

import os
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Embedding, Input, MaxPooling1D, Conv1D, Dropout
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD
from sklearn.metrics import roc_auc_score

import keras.backend as K
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical


# some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.02
BATCH_SIZE = 128
EPOCHS = 5


# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('/floyd/input/dataset/glove.6B/glove.6B.100d.txt'),encoding='utf-8') as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))



# prepare text samples and their labels
print('Loading in comments...')


#sentences = ["I passed an exam that I was absolutely certain that I had failed.","Every time I imagine that someone I love or I could contact a serious illness, even death.","I was the reason behind the break-up of my friend's relationship with his girlfriend.  She finished with him.","Mad at my dad.","I am extremely angry now."]
#sentences = ["anger","happy","sad","fear","guilty"]
# convert the sentences (strings) into integers
def preprocess(sentences):
	# convert the sentences (strings) into integers
	tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
	tokenizer.fit_on_texts(sentences)
	sequences = tokenizer.texts_to_sequences(sentences)



	# get word -> integer mapping
	word2idx = tokenizer.word_index
	print('Found %s unique tokens.' % len(word2idx))


	# pad sequences so that we get a N x T matrix
	database = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
	print('Shape of data tensor:', database.shape)


	# Predicting values

	print("Loading saved model")
	model = model_from_json(open("/floyd/input/dataset/hate_speech_lstm_model.json","r").read())
	model.load_weights('/floyd/input/dataset/hate_speech_lstm_model_weight.h5')
	print("Saved model loaded!!!")

	pred = model.predict(database)
	print(pred)

	print("Output values:")
	print("Not    :  "+str(round(1-pred[0][0]*100,2))+"%")
	print("Racist/Sexist   :  "+str(round((pred[0][0])*100,2))+"%")
	


sentences = ["you are a female. this is not your job, you black sheep!!","I am very happy today!"]
preprocess(sentences)

"""
while True:
  sentences = []
  print("Enter the text:")
  ques = input()
  sentences.append(ques)
  preprocess(sentences)
  print("Enter (y/n) to continue:")
  choice = input()
  if choice=='n':
    break
"""
