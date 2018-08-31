#Example from https://www.twilio.com/blog/2017/09/sentiment-analysis-python-messy-data-nltk.html

import nltk
nltk.download('punkt')
def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

print(format_sentence("The cat is very cute"))
