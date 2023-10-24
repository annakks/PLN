from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot as plt
from normaliza import preprocess_dataframe

def dendogram_cluster(dataframe):

    print("\n\n")
    documents = dataframe['unique_normal']
    print(documents)

    #Building text corpus dataframe to visualize document clusters
    corpus_df = DataFrame({"Document": documents})
    tfidf_vectorized = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True)
    features = tfidf_vectorized.fit_transform(documents)
    model_lexicon = tfidf_vectorized.get_feature_names_out()
    feature_vectors = features.toarray()
    print(model_lexicon)
    print(feature_vectors)
    print("Vectorized corpus document models: ")
    feature_df = DataFrame(feature_vectors,columns= model_lexicon).transpose()
    print(feature_df)

    # Similarity matrix computation
    similarity_matrix = cosine_similarity(feature_vectors)
    Z = linkage(similarity_matrix, 'ward')
    print( pd.DataFrame(Z,columns=['Document 1\Cluster 1', 'Document\Cluster 2',
                'Distance', 'Cluster Size']) )
    dendrogram(Z)
    plt.title("Dendrogram for hierarchical clustering")
    plt.xlabel("document index")
    plt.ylabel("cluster distances")
    plt.grid()
    plt.show()


    # Building text corpus dataframe to visualize document clusters
    tfidf_vectorizer = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True)
    features = tfidf_vectorizer.fit_transform(documents)
    feature_vectors = features.toarray()
    similarity_matrix = cosine_similarity(feature_vectors)
    Z = linkage(similarity_matrix, 'ward')
    max_dist = 1.5
    cluster_labels = fcluster(Z, max_dist, criterion='distance')

    # Adicione a coluna 'Cluster Label' ao DataFrame original (df_pre)
    dataframe['Cluster Label'] = cluster_labels

    # Agora df_pre contém a coluna 'Cluster Label
    print("\n\n")
    print (dataframe)
    
    return 

data = pd.read_csv('https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv')
data = preprocess_dataframe(data)

# Chame a função vectorize_text com o DataFrame pré-processado
dendogram_cluster(data)