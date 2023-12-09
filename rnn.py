import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
dataset = pd.read_csv("./training.1600000.processed.noemoticon.csv")
print(dataset.head())

dataset.columns = ['sentiment','id','date','query_string','user','text']
sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()
print(sentences[1])
print(labels[1])
training_size = int(len(sentences) * 0.8)
training_sentences = sentences[0: training_size]
testing_senteces = sentences[: training_size]
training_labels = labels[0: training_size]
testing_labels = labels[: training_size]

# Put labels into list to use later:

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
vocab_size = 1000
embedding_dim = 16
max_length = 280
trunc_type='post'
padding_type='post'
oov_tok = ""
     

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_senteces)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(embedding_dim,
                         return_sequences=False)
))
model.add(tf.keras.layers.Dense(6, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.legacy.Adam(0.01),
              metrics=['accuracy'])

callbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)
num_epochs=10
modelo = model.fit(training_padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final),
          callbacks=[callbacks])
plt.plot(modelo.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(modelo.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
     

sub_tweets = ['Trust @VictorBautiista on who should be the next president. #Trump2016 https://t.co/mzWxx74kZn', 
                'Big companies NEED to be split up - too much power  ""Bell Telephone"" was too big &amp; was split up 2 many smaller ones. I hope #Trump can do it',
                'THE DEMOCRAT PARTY  is EVIL.  Have you ever seen the Republican party pay people to destroy, fight, block voters and be paid?  NO. EVIL!!!', 
                '@FrankLuntz  You do NOT KNOW who is voting!!  #Trump is getting lots of democrats and Ind.  He is going to win FLA!! you sd not speculate!', 
                'To all my fellow Americans! We know what we have with Hillary. Let\'s make Trump the next POTUS. Let\'s make this miracle happen for the USA!',
                'Americans game on!  Don\'t sit on the sidelines. GET OUT VOTE!  Do your part to save our country. Vote for Trump', 
                '@seanhannity I want to Vote for Trump But to scared to legally blind i am 53  i hope she gos to jail hope you see my tweets',
                'Clinton/Trump is the WrestleMania 32 of presidential elections', 
                '@oreillyfactor Obama called Trump PRESIDENT before he stopped himself FIVE']


# Create the sequences
padding_type='post'
sample_sequences = tokenizer.texts_to_sequences(sub_tweets)
tweets_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)           

classes = model.predict(tweets_padded)
