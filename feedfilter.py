# Feed filter
# Takes a sentence/tweet and classifies the entries

import sys
import codecs
import nltk

def read(tweets,classifier):
    sys.stdout = codecs.getwriter(sys.stdout.encoding)(sys.stdout,errors='replace')
    words=tweets
    for word in words:
            print word
            print 'Guess: '+ str(classifier.classify(word))
            cl=raw_input('Enter category: ')
            filtered_words=[]
            for w in words:
                if w not in nltk.corpus.stopwords.words('english'):
                    filtered_words.append(w)
            classifier.train(w,cl)
