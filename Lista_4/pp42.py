import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
from numpy import average
from gensim.models import word2vec
from nltk.tokenize import word_tokenize

# Use o modelo word2vec já treinado
feature_size = 5  # tamanho da representação vetorial (mesmo valor usado durante o treinamento)
window_context = 2
min_word_count = 1
sample = 1e-3

# Utilize a mesma base de dados utilizada para treinamento
text_corpus = "Os melhores produtos " \
"são vendidos por uma loja que se preocupa "\
"com a experiência do cliente. Muitos adoram " \
"produtos que chegam no prazo ou que satisfazem "\
"suas expectativas. Pessoas gostam de produtos "\
"vendidos com tamanhos corretos, adequados "\
"ou que chegam no prazo. Pessoas adoram mercadorias "\
"que atendem as suas expectativas. "

stop_words = ['a', 'as', 'e', 'o', 'os', 'da', 'de', 'do', 'um', 'uma']

normalized_sentences = []
word_tokens = [word_tokenize(sentence) for sentence in text_corpus.split('.')]
for sentence in word_tokens:
    normalized_sentences.append([re.sub(r'[^A-Za-zÀ-Ýà-ý]','', word).lower() for word in sentence if re.sub(r'[^A-Za-zÀ-Ýà-ý]','', word).lower() != ''])

for word in stop_words:
    for sentence in normalized_sentences:
        if word in sentence:
            print(word)
            print(sentence)
            sentence.remove(word)

# Treine o modelo word2vec
w2vec_model = word2vec.Word2Vec(normalized_sentences, vector_size=feature_size,
                                window=window_context, min_count=min_word_count,
                                sample=sample, epochs=50)
print(normalized_sentences)
print("\n\n")

# Função para substituir palavras por sinônimos com base no cosseno
def substituir_por_sinonimos(frase, modelo, threshold=0.6):
    print("\n\nFrase original:")
    print(frase)
    
    tokens = [re.sub(r'[^A-Za-zÀ-Ýà-ý]','', word).lower() for word in word_tokenize(frase)]
    
    print("\nTokens:")
    print(tokens)
    
    novos_tokens = []

    for token in tokens:
        if token in modelo.wv:
            similaridade_max = threshold
            sinonimo = token
            
            for palavra in modelo.wv.index_to_key:
                if palavra != token:
                    v1 = modelo.wv[token]
                    v2 = modelo.wv[palavra]
                    similaridade = dot(v1, v2) / (norm(v1) * norm(v2)) # calculando o cosseno 
                    
                    if similaridade > similaridade_max:
                        similaridade_max = similaridade
                        sinonimo = palavra
            
            if similaridade_max >= threshold:
                novos_tokens.append(sinonimo)
                print(f"Substituindo '{token}' por '{sinonimo}'")
            else:
                novos_tokens.append(token)

    nova_frase = ' '.join(novos_tokens)
    
    print("\nFrase com substituição de sinônimos:")
    print(nova_frase)
    print("\n\n")

# Exemplo de uso
frase_usuario = input("Digite uma frase: ")
substituir_por_sinonimos(frase_usuario, w2vec_model)