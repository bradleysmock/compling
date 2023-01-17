import nltk
import random
from tokenizer import tokenize
from lemmatizer import get_lemma

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


def clean_data(filename):
    text_data = []
    with open(filename) as f:
        for line in f:
            tokens = prepare_text_for_lda(line)
            if random.random() > .99:
                print(tokens)
                text_data.append(tokens)

    return text_data
