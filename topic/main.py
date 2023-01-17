import gensim
import pyLDAvis
from gensim import corpora
import pickle
import pyLDAvis.gensim_models
from clean import clean_data

# # load data
text_data = clean_data('dataset.csv')

# create dictionary
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

# analyze
NUM_TOPICS = 5
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
lda_model.save('model.gensim')
topics = lda_model.print_topics(num_words=4)
for topic in topics:
    print(topic)

# visualize
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model.gensim')
lda_display = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, sort_topics=False)
# pyLDAvis.display(lda_display)
pyLDAvis.save_html(lda_display, 'output.html')
