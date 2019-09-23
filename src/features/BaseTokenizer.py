#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 15:13:40 2019

@author: ishanshrivastava
"""

import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re, string, unicodedata
import nltk
import inflect
from bs4 import BeautifulSoup
from nltk.stem import LancasterStemmer, WordNetLemmatizer
#import contractions
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)
    
#def replace_contractions(text):
#    """Replace contractions in string of text"""
#    return contractions.fix(text)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_dot(words):
    """Remove punctuation from list of tokenized words"""
    ds = re.compile(r'\b(\w+[.]\w+)')
    new_words = []
    for word in words: 
        m=ds.match(word)
        if m:
            new_word = m.sub('.',' ')
            if new_word != '':
                new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words: 
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = [] 
    num = re.compile(r'^\d*\.?\d*$')            
    for word in words:
        if re.match(num, word):
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    stop = stopwords.words('english') + [' ']
    new_words = []
    for word in words:
        if word not in stop:
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    #words=replace_contractions(words)
    words =replace_numbers(words)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    #words = remove_dot(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    return words

"""
    Function to call
"""

def BaseTokenizerSingleSent(sent):
    sent=sent.replace('/', ' ')
    sent=sent.replace('-', ' ')
    sent = remove_between_square_brackets(sent)
    tokens = nltk.word_tokenize(sent)
    tokens = normalize(tokens)
    return tokens

# Input SHOULD be 2 sentences, but to preserve reverse-compatibility
# accepting either 1 or 2 sentences.
def BaseTokenizer(sent1, sent2=None):
	if sent2 == None:
		return BaseTokenizerSingleSent(sent1)
	else:
		return [BaseTokenizerSingleSent(sent1), BaseTokenizerSingleSent(sent2)]
