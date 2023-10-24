from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pandas import DataFrame
import pandas as pd
from normaliza import preprocess_dataframe

def topics(dataframe):

    print("\n\n")
    documents = dataframe['unique_normal']
    print(documents)


    tfidf_vectorized = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True)
    features = tfidf_vectorized.fit_transform(documents)
    model_lexicon = tfidf_vectorized.get_feature_names_out()
    feature_vectors = features.toarray()
    lda = LatentDirichletAllocation(n_components=3, max_iter=10000, random_state=0)
    dt_matrix = lda.fit_transform(feature_vectors)
    topic_features = DataFrame(dt_matrix, columns=["topic 1", "topic 2", "topic 3"])
    print("\n\n")
    print(topic_features)
    print("\n\n")
    vocab = model_lexicon
    topic_matrix = lda.components_
    tt_matrix = lda.components_
    for topic_weights in tt_matrix:
        topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]
        topic = sorted(topic, key=lambda x: -x[1])
        topic = [item for item in topic if item[1] > 0.6]
        
        print(topic)
        print()
    
    return

data = pd.read_csv('https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv')
data = preprocess_dataframe(data)

# Chame a função vectorize_text com o DataFrame pré-processado
topics(data)