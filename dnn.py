
import os 
import random as rnd
import re
import string
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import trax
import nltk
import trax.fastmath.numpy as np
from trax import fastmath

# import trax.layers
from trax import layers as tl
nltk.download('twitter_samples')
nltk.download('stopwords')
from nltk.corpus import stopwords, twitter_samples 
from nltk.tokenize import TweetTokenizer
stopwords_english = stopwords.words('english')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def load_tweets():
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')  
    return all_positive_tweets, all_negative_tweets
    
def process_tweet(tweet):

    # remove stock market tickers like $GE
    tweet = re.sub(r'$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    ### START CODE HERE ###
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and # remove stopwords
            word not in string.punctuation): # remove punctuation
            #tweets_clean.append(word)
            stem_word = stemmer.stem(word) # stemming word
            tweets_clean.append(stem_word)
    return tweets_clean


all_positive_tweets, all_negative_tweets = load_tweets()

train_pos  = all_positive_tweets[:4000]
train_neg  = all_negative_tweets[:4000] 

# Combine training data into one set
train_x = train_pos + train_neg 

Vocab = {'__PAD__': 0, '____': 1, '__UNK__': 2} 

for tweet in train_x: 
    processed_tweet = process_tweet(tweet)
    for word in processed_tweet:
        if word not in Vocab: 
            Vocab[word] = len(Vocab)



def classifier(vocab_size=10000, embedding_dim=256, output_dim=4, mode='train'):
    # create embedding layer
    embed_layer = tl.Embedding(
        vocab_size=vocab_size, # Size of the vocabulary
        d_feature=embedding_dim)  # Embedding dimension
    
    mean_layer = tl.Mean(axis=1)
    
    # Create a dense layer, one unit for each output
    dense_output_layer = tl.Dense(n_units = output_dim)

    log_softmax_layer = tl.LogSoftmax()
    
    # Use tl.Serial combinator
    model = tl.Serial(
      embed_layer, # embedding layer
      mean_layer, # mean layer
      dense_output_layer, # dense output layer 
      log_softmax_layer # log softmax layer
    )
    
    # return the model of type
    return model

model = classifier()

def tweet_to_tensor(tweet, vocab_dict, unk_token='__UNK__', verbose=False):
   
    
    word_l = process_tweet(tweet)
    
    if verbose:
        print("List of words from the processed tweet:")
        print(word_l)
        
    tensor_l = []
    
    unk_ID = vocab_dict[unk_token]
    
    if verbose:
        print(f"The unique integer ID for the unk_token is {unk_ID}")
        
    for word in word_l:
        
        word_ID = vocab_dict[word] if word in vocab_dict else unk_ID
        tensor_l.append(word_ID) 
    
    return tensor_l


weights, state = model.init_from_file("./Sentiment_NN_model.pkl")


def predict(sentence):
    inputs = np.array(tweet_to_tensor(sentence, vocab_dict=Vocab))
    
    inputs = inputs[None, :]  
    
    preds_probs = model(inputs) # log softmax result
    
    # Turn probabilities into categories
    preds = int(preds_probs[0, 1] > preds_probs[0, 0])
    
    sentiment = "negative"
    if preds == 1:
        sentiment = 'positive'

    return preds, sentiment



sub_tweets = ['Trust @VictorBautiista on who should be the next president. #Trump2016 https://t.co/mzWxx74kZn', 
                'Big companies NEED to be split up - too much power  ""Bell Telephone"" was too big &amp; was split up 2 many smaller ones. I hope #Trump can do it',
                'THE DEMOCRAT PARTY  is EVIL.  Have you ever seen the Republican party pay people to destroy, fight, block voters and be paid?  NO. EVIL!!!', 
                '@FrankLuntz  You do NOT KNOW who is voting!!  #Trump is getting lots of democrats and Ind.  He is going to win FLA!! you sd not speculate!', 
                'To all my fellow Americans! We know what we have with Hillary. Let\'s make Trump the next POTUS. Let\'s make this miracle happen for the USA!',
                'Americans game on!  Don\'t sit on the sidelines. GET OUT VOTE!  Do your part to save our country. Vote for Trump', 
                '@seanhannity I want to Vote for Trump But to scared to legally blind i am 53  i hope she gos to jail hope you see my tweets',
                'Clinton/Trump is the WrestleMania 32 of presidential elections', 
                '@oreillyfactor Obama called Trump PRESIDENT before he stopped himself FIVE']


def classify_csv(df): 
    for i in range(sub_tweets):
        print(f'{sub_tweets[i]} is {predict(sub_tweets[i])}')

