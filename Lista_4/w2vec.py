import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec
from nltk.tokenize import word_tokenize
from sklearn.neural_network import MLPClassifier
import numpy as np
from numpy import average
from sklearn.metrics import confusion_matrix

nltk.download('stopwords')
stop_words = stopwords.words('portuguese')
stop_words.remove('não')

def remover_stopwords(texto):
    texto = texto.split(' ')
    for word in texto:
        if word in stop_words:
            texto.remove(word)
    
    return ' '.join(texto)

csv_url = 'https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/master/B2W-Reviews01.csv'
df = pd.read_csv(csv_url)

# Filtrar os comentários com 'review_text' não vazia ou NaN
filtered_df = df.dropna(subset=['review_text'])
filtered_df = filtered_df[filtered_df['review_text'].str.strip() != '']

# Ordenar o DataFrame por data (coluna 'review_creation_date')
filtered_df['submission_date'] = pd.to_datetime(filtered_df['submission_date'])
filtered_df = filtered_df.sort_values(by='submission_date', ascending=False)

recent_comments = filtered_df

df_pre = recent_comments[['review_text', 'overall_rating']]

df_pre['review_text_normalized'] = [re.sub(r'[^a-zA-ZãõÃÕáéíóúâêîôûàèìòùäëïöüçÁÉÍÓÚÂÊÎÔÛÀÈÌÒÙÄËÏÖÜÇ$]', ' ', word).lower() for word in df_pre['review_text']]
df_pre['review_text_normalized'] = df_pre['review_text_normalized'].apply(remover_stopwords)
df_pre['review_text_normalized'] = df_pre['review_text_normalized'].apply(word_tokenize)

normalized_sentences = []
sentences = []
positive = []
negative = []
neutral = []

def selecionarFrases(rating, frase_normalizada, frase):
    if len(positive) < 15 and rating >= 4 :
        positive.append(frase_normalizada)
        normalized_sentences.append(frase_normalizada)
        sentences.append(frase)
    if len(negative) < 15 and rating <= 2:
        negative.append(frase_normalizada)
        normalized_sentences.append(frase_normalizada)
        sentences.append(frase)
    if len(neutral) < 15 and rating == 3:
        neutral.append(frase_normalizada)
        normalized_sentences.append(frase_normalizada)
        sentences.append(frase)

df_pre.apply(lambda row: selecionarFrases(row['overall_rating'], row['review_text_normalized'], row['review_text']), axis=1)

feature_size = 5  # size of vector representation
window_context = 2
min_word_count = 1
sample = 1e-3
w2vec_model = word2vec.Word2Vec(df_pre['review_text_normalized'], vector_size= feature_size,
                                window=window_context, min_count= min_word_count,
                                sample=sample, epochs = 50)

classified_reviews= [
        {'corpus': "Ótimo produto e entrega Super rapida chegou em menos de dois dias", 'review_type': 'positive', 'feature_vector': [-0.13853185,  0.4105362,   0.79071844,  0.27744952,  0.07487416]},
        {'corpus': "Chegou bem antes do prazo Tudo otimo recomendo demais", 'review_type': 'positive', 'feature_vector': [ 0.07570543, -0.34266919,  0.52896768,  0.27744952, -0.30617532]},
        {'corpus': "Ótimo produto Excelente qualidade Já usei e recomendo este produto", 'review_type': 'positive', 'feature_vector': [-0.13853185,  0.4105362,   0.79071844,  0.27744952,  0.07487416]},
        {'corpus': "Produto de boa qualidade e recebi em perfeito estado entrega super rápida", 'review_type': 'positive', 'feature_vector': [-0.42395949,  0.27744952,  0.7360329,  -0.30617532, -0.13853185]},
        {'corpus': "Chegou no prazo correto Aparelho ótimo tudo conforme descrito no anúncio", 'review_type': 'positive', 'feature_vector': [ 0.07570543, -0.34266919,  0.52896768,  0.27744952, -0.30617532]},

        {'corpus': "Não gostei do produto Qualidade duvidosa Não recomendo", 'review_type': 'negative', 'feature_vector': [ 0.044854,    0.27744952,  0.52896768,  0.27744952, -0.10858855]},
        {'corpus': "Ainda não recebi o produto e nem uma posição ou retorno de quando receberei", 'review_type': 'negative', 'feature_vector': [ 0.4105362,   0.03417454,  0.7360329,  -0.02887908,  0.03417454]},
        {'corpus': "Comprei e me arrependi Produto de baixa qualidade veio com uma peça trincada Não recomendo", 'review_type': 'negative', 'feature_vector': [ 0.27744952,  0.79071844,  0.07487416, -0.42395949, -0.34266919]},
        {'corpus': "Até hoje não recebi o produto e nem o dinheiro não recomendo essa americanas para ninguém", 'review_type': 'negative', 'feature_vector': [-0.13853185,  1.05806339,  0.07570543,  0.27744952, -0.4690485 ]},
        {'corpus': "produto não é bom", 'review_type': 'negative', 'feature_vector': [ 0.07487416, -0.42395949,  0.27744952,  0.7360329,  -0.30617532]},

        {'corpus': "muito simples pelo preço isso que da compra espontaneamente", 'review_type': 'neutral', 'feature_vector': [ 0.79071844, -0.30617532,  0.4105362,  -0.13853185,  0.27744952]},
        {'corpus': "Bom Conforme combinado chegou antes do prazo Tudo certo", 'review_type': 'neutral', 'feature_vector': [0.27744952, 0.79071844, 0.27744952, 0.03417454, 0.6538589 ]},
        {'corpus': "estou satisfeito com o produto bom e barato o meu não o entrou aguá", 'review_type': 'neutral', 'feature_vector': [-0.34266919, -0.10858855, -0.13853185,  0.27744952, -0.30617532]},
        {'corpus': "A qualidade do produto é equivalente ao preço ou seja, é bom para o valor cobrado", 'review_type': 'neutral', 'feature_vector': [-0.08223853, -0.30617532, -0.02887908,  0.49274963,  0.4105362 ]},
        {'corpus': "Nao gostei do material do produto O tamanho é ótimo chegou muito rápido do não gostei do material", 'review_type': 'neutral', 'feature_vector': [-0.02887908,  0.27744952,  0.52896768,  0.27744952, -0.10858855]}
]

j = 0
df_pre['sentiment_text'] = 'neutral'
for documento in normalized_sentences:
    word_embedding = [] 
    for word in documento:
        if word in w2vec_model.wv:
            word_embedding.append(w2vec_model.wv[word])

    word_embedding = np.array(word_embedding)

    doc_embedding = np.zeros(feature_size)
    for i in range(min(feature_size, len(word_embedding))):
        doc_embedding[i] = average(word_embedding[i])


    unclassified_review = {
        'corpus': documento,
        'review_type': '',
        'feature_vector': doc_embedding  
        }

    X = [] # feature vectorsS
    y = [] # feature classes
    for review in classified_reviews:
        X.append(review['feature_vector'])
        y.append(review['review_type'])

    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 3), random_state=1, max_iter=1000)



    classifier.fit(X, y)

    sentimento = classifier.predict([unclassified_review['feature_vector']])
    print(sentences[j])
    print(sentimento)
    print('==================================================================')
    df_pre['sentiment_text'][j] = sentimento[0]
    j = j + 1

def calcular_matriz_confusao(df):
    
    rating_mapping = {5: 'positive', 4: 'positive', 3: 'neutral', 2: 'negative', 1: 'negative'}
    df['overall_rating'] = df['overall_rating'].map(rating_mapping)

    print (df)
    
    # Calculando a matriz de confusão
    matriz_confusao = confusion_matrix(df['overall_rating'].values, df['sentiment_text'].values)

    print(matriz_confusao)

    # Calculando a acurácia
    accuracy = (matriz_confusao[0,0]+matriz_confusao[1,1]+matriz_confusao[2,2])/45

    # Calculando a precisão para a classe positiva
    precision_positivo = (matriz_confusao[0,0])/(matriz_confusao[0,0]+matriz_confusao[1,0]+matriz_confusao[2,0])

    precision_neutral = (matriz_confusao[0,1])/(matriz_confusao[0,1]+matriz_confusao[1,1]+matriz_confusao[2,1])

    precision_negativo = (matriz_confusao[0,2])/(matriz_confusao[0,2]+matriz_confusao[1,2]+matriz_confusao[2,2])

    
    return matriz_confusao, accuracy, precision_positivo, precision_negativo

matriz_confusao, accuracy, precision_positivo, precision_negativo = calcular_matriz_confusao(df_pre)

print("\n \n")
print("Matriz de Confusão:")

print("        Positivo |   Neutro |  Negativo ")
print("Positivo   ", matriz_confusao[0,0],"|       ", matriz_confusao[0,1],"|    ", matriz_confusao[0,2]) 
print("Neutro      ", matriz_confusao[1,0],"|       ", matriz_confusao[1,1],"|    ", matriz_confusao[1,2])
print("Negativo   ", matriz_confusao[2,0],"|       ", matriz_confusao[2,1],"|    ", matriz_confusao[2,2])

print("\n \n")
print("Acurácia:", accuracy*100, " %" )

print("\n \n")
print("Precisão (Positivo):", precision_positivo*100, " %")
print("\n \n")

print("Precisão (Negativo):", precision_negativo*100, " %")
print("\n \n")