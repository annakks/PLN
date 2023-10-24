from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
import pandas as pd
from normaliza import preprocess_dataframe

def vectorize_text(dataframe):

    #Selecio documento    
    documents = dataframe['unique_normal'].sample(n=2)
    
    print("\n\n")
    print('Documentos usados')
    print (documents)

    #bag of words
    counter_vectorize = CountVectorizer(min_df=0., max_df= 1.)
    features = counter_vectorize.fit_transform(documents)
    model_lexicon = counter_vectorize.get_feature_names_out()
    feature_vectors = features.toarray()
    print("\n\n")
    print ('modelo lexico:')
    print (model_lexicon)
    print("\n\n")
    print ('feature vectors')
    print (feature_vectors)
    print("\n\n")

    feature_dfBW = DataFrame(feature_vectors,columns= model_lexicon).transpose()
    print("Vectorized corpus document models: ")
    print("\n\n")
    print(feature_dfBW)

    tfidf_vectorized = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True)
    features = tfidf_vectorized.fit_transform(documents)
    model_lexicon = tfidf_vectorized.get_feature_names_out()
    feature_vectors = features.toarray()
    print("\n\n")
    print('feature_vectors: ')
    print(feature_vectors)
    print("\n\n")
    
    feature_dfV = DataFrame(feature_vectors,columns= model_lexicon).transpose()
    print("Vectorized corpus document models: ")
    print(feature_dfV)
    print("\n\n")

   

    return 


data = pd.read_csv('https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv')
data = preprocess_dataframe(data)

# Chame a função vectorize_text com o DataFrame pré-processado
vectorize_text(data)





