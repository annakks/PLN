import nltk
import spacy
import numpy as np
import pandas as pd
import copy as cp
import joblib
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.corpus import mac_morpho as mm
nltk.download('punkt')
nltk.download('popular')
nltk.download('wordnet')

import spacy
import spacy.cli
from spacy import displacy
from spacy.lang.pt.examples import sentences
import en_core_web_sm

spacy.cli.download("pt_core_news_sm")
en_core_web_sm = en_core_web_sm.load()

print ( "PP.1.1 Baseando-se no código-fonte fornecido pelo professor, exemplifique o carregamento da biblioteca NLTK, em Python e efetue a tokenização de um texto em português pertencente a alguma obra literária de domínio público. Utilize um texto de pelo menos 2000 caracteres. Mostre o funcionamento do seu programa e descreva ao menos 5 POS tags.")

pp1 = "As armas e os Barões assinalados Que da Ocidental praia Lusitana Por mares nunca de antes navegados Passaram ainda além da Taprobana, Em perigos e guerras esforçados Mais do que prometia a força humana, E entre gente remota edificaram Novo Reino, que tanto sublimaram; E também as memórias gloriosas Daqueles Reis que foram dilatando A Fé, o Império, e as terras viciosas De África e de Ásia andaram devastando, E aqueles que por obras valerosas Se vão da lei da Morte libertando, Cantando espalharei por toda parte, Se a tanto me ajudar o engenho e arte. Cessem do sábio Grego e do Troiano As navegações grandes que fizeram; Cale-se de Alexandro e de Trajano A fama das vitórias que tiveram; Que eu canto o peito ilustre Lusitano, A quem Neptuno e Marte obedeceram. Cesse tudo o que a Musa antiga canta, Que outro valor mais alto se alevanta. E vós, Tágides minhas, pois criado Tendes em mi um novo engenho ardente, Se sempre em verso humilde celebrado Foi de mi vosso rio alegremente, Dai-me agora um som alto e sublimado, Um estilo grandíloco e corrente, Por que de vossas águas Febo ordene Que não tenham enveja às de Hipocrene. Dai-me üa fúria grande e sonorosa, E não de agreste avena ou frauta ruda, Mas de tuba canora e belicosa, Que o peito acende e a cor ao gesto muda; Dai-me igual canto aos feitos da famosa Gente vossa, que a Marte tanto ajuda; Que se espalhe e se cante no universo, Se tão sublime preço cabe em verso. "
words = pp1.split()
bag_of_words = cp.deepcopy(words)
np.random.shuffle(bag_of_words)
# Bag of words:
print("Palavras: ", bag_of_words)

portuguese_tagger = joblib.load('/content/POS_tagger_brill.pkl')
pos_tags = portuguese_tagger.tag(nltk.word_tokenize(pp1))
print ("Tags POS com NLTK")
print(pos_tags)
pos_tags_df = pd.DataFrame(pos_tags).T
print(pos_tags_df)

print("---------------------------")

print("Tags POS com Spacy")
## https://spacy.io/models/pt
model_spacy = en_core_web_sm
pos_tags_2 = [ (word, word.pos_) for word in model_spacy(pp1)]
pos_tags_2_df = pd.DataFrame(pos_tags_2).T
print(pos_tags_2)
print(pos_tags_2_df)

print("_____________________________________________________________")

print ( "PP.1.2. Exemplifique a stemização e a lematização de um texto, em língua portuguesa. Ilustre um caso onde textos diferentes conduzem a uma mesma saída através do stemming ou lemmatization. Considere como saída um vetor ordenado contendo lemas e stems.")

text1 = "As crianças brincam felizes no parque"
text2 = "A criança brinca feliz no parque."

stemmer = PorterStemmer()
words1 = word_tokenize(text1)
words2 = word_tokenize(text2)

stems1 = [stemmer.stem(word1) for word1 in words1]
stems2 = [stemmer.stem(word2) for word2 in words2]
print("Stemização:")
print("Stems texto 1: ", stems1)
print("Stems texto 2: ", stems2)

print("---------------------------")

nlp = spacy.load('pt_core_news_sm')

doc1 = nlp(text1)
doc2 = nlp(text2)

lemmas1 = [token.lemma_ for token in doc1]
lemmas2 = [token.lemma_ for token in doc2]
print("Lematização:")
print("Lemmas texto 1: ", lemmas1)
print("Lemmas texto 2: ", lemmas2)

print("_____________________________________________________________")

print ( "PP.1.3. Repita PP.1.1. considerando a língua inglesa.")

pp3 = "Call me Ishmael. Some years ago - never mind how long precisely - having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking my hat off to the world - then, I account it high time to get to the sea as soon as I can."
words3 = pp3.split()
bag_of_words3 = cp.deepcopy(words3)
np.random.shuffle(bag_of_words3)
# Bag of words:
print("Palavras: ", bag_of_words3)


ingles_tagger = joblib.load('/content/POS_tagger_brill.pkl')
pos_tags3 = ingles_tagger.tag(nltk.word_tokenize(pp3))
print ("Tags POS com NLTK")
print(pos_tags3)
pos_tags_df3 = pd.DataFrame(pos_tags3).T
print(pos_tags_df3)

print("---------------------------")

print("Tags POS com Spacy")
## https://spacy.io/models/pt
model_spacy3 = en_core_web_sm
pos_tags_3 = [ (word, word.pos_) for word in model_spacy(pp1)]
pos_tags_3_df = pd.DataFrame(pos_tags_3).T
print(pos_tags_3)
print(pos_tags_3_df)

print("_____________________________________________________________")

print ( "PP.1.4. Repita PP.1.2. considerando a língua inglesa.")

text3 = "you are walking" # (Você esta caminhando)
text4 = "you are walking" # (Vocês estão caminhando)

stemmer = PorterStemmer()
words3 = word_tokenize(text1)
words4 = word_tokenize(text2)

stems3 = [stemmer.stem(word3) for word3 in words3]
stems4 = [stemmer.stem(word4) for word4 in words4]
print("Stemização:")
print("Stems texto 3: ", stems3)
print("Stems texto 4: ", stems4)

print("---------------------------")

nlp = spacy.load('pt_core_news_sm')

doc3 = nlp(text3)
doc4 = nlp(text4)

lemmas3 = [token.lemma_ for token in doc3]
lemmas4 = [token.lemma_ for token in doc4]
print("Lematização:")
print("Lemmas texto 3: ", lemmas3)
print("Lemmas texto 4: ", lemmas4)

