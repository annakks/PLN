import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')

def remover_caracteres(texto):
    texto = texto.replace(',', '')  # Remove vírgulas
    texto = ''.join(e for e in texto if not e.isdigit())  # Remove números
    texto = ''.join(e for e in texto if e.isalnum() or e.isspace())  # Remove caracteres especiais
    return texto

# tokenização
def tokenize_unique(text):
    tokens = word_tokenize(text)
    unique_tokens = list(set(tokens)) 
    return ' '.join(unique_tokens) 

# pré-processamento
def preprocess_dataframe(dataframe):
    
    # Filtrar os comentários com 'review_text' não vazia ou NaN 
    filtered_df = dataframe.dropna(subset=['review_text'])
    filtered_df = filtered_df[filtered_df['review_text'].str.strip() != '']
    # Ordenar o DataFrame por data (coluna 'review_creation_date')
    filtered_df['submission_date'] = pd.to_datetime(filtered_df['submission_date'])
    filtered_df = filtered_df.sort_values(by='submission_date', ascending=False)
    # Selecionar os 10 comentários mais recentes
    recent_comments = filtered_df.sample(n=10)

    df_pre = recent_comments[['review_text']]
    dataframe=df_pre

    
    # remoção de stopwords
    comments = []
    stop_words = set(['a', 'ao', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'assim', 'até', 'com', 'como',
                      'da', 'de', 'deles', 'dela', 'desde', 'deve','disso', 'do', 'dos', 'e', 'ela', 'elas', 'ele', 'eles', 'em', 
                      'então', 'esse', 'esta', 'está', 'eu', 'isso', 'meu', 'neste', 'no', 'o', 'os', 'ou', 'outro', 'para', 
                      'pode', 'por', 'pois', 'porquê', 'quando', 'que', 'quem', 'quer', 'se', 'ser', 'sua', 'são', 'também', 'tem', 
                      'um', 'uma', 'você', 'vocês', 'vós'])
    dataframe['review_text'] = dataframe['review_text'].astype(str)

    for words in dataframe['review_text']:
        tokens = nltk.word_tokenize(words) # Somente palavras
        lower_case = [l.lower() for l in tokens] # Tudo em minúsculo
        filtered_result = [l for l in lower_case if l not in stop_words] # Remove as palavras stopwords
        comments.append(' '.join(filtered_result))

    # pré-processados na coluna 'review_text'
    dataframe['review_text'] = comments

    # aplicar a remoção de números, caracteres especiais e vírgulas
    dataframe['review_text'] = dataframe['review_text'].apply(remover_caracteres)

    # acrescenta a coluna com a normalização
    dataframe['unique_normal'] = dataframe['review_text'].apply(tokenize_unique)
    

    return dataframe