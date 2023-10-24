<p align = "center">
*ATIVIDADE 03 PLN - 6°D.S.M* 
<p align = "center">
Ana Carolina, Bruno Pisciotta, Daniel Luciano e Vinicius Buarque


TERMINOLOGIA E CONCEITOS 

TC.3.1. Qual tipo de problema pode surgir na montagem de um modelo do po bag of words caso etapas tais como a remoção de caracteres especiais (ex. sinais de pontuação) e conversão para minúsculas/maiúsculas não sejam efetuadas? Ilustre isso para as seguintes frases que fazem parte de um mesmo corpus de texto (i.e.: são usadas para a montagem do léxico do modelo): 
Frase 1: Eu quero tomar água! 
Frase 2: eu, prefiro tomar café. 
Na sua resposta mostre como ficaria o léxico do modelo e os bag of words correspondentes.

```bash


No caso onde a remoção de caracteres especiais (como sinais de pontuação)
e a conversão para minúsculas/maiúsculas não forem efetuadas,
ao adotar um modelo do Bag of Words (BoW),
um problema que pode surgir é a contagem inadequada de palavras.
Porque o modelo trataria palavras com letras maiúsculas e minúsculas,
bem como palavras com e sem pontuação, como diferentes palavras.

Léxico:

Eu, eu, quero, tomar, água, prefiro, café. 

ex: ("Eu" e "eu" -> Palavras diferentes)

 Bag of Words: 

Frase 1: [1, 1, 1, 1, 0, 0, 0]
Frase 2: [0, 0, 1, 0, 1, 1, 1]

Usando as frases como exemplo

As palavras "Eu" e "eu" são tratadas como palavras distintas devido à diferença de maiúsculas/minúsculas.
Isso pode levar a representações diferentes e menos eficazes do texto,
já que as palavras comuns são tratadas como únicas e a contagem é afetada pela forma como as palavras são escritas.

Pesquisa:
https://medium.com/turing-talks/introdu%C3%A7%C3%A3o-a-bag-of-words-e-tf-idf-43a128151ce9
````

TC.3.2. Qual a relação entre as etapas de pré-processamento de texto e a redução de dimensionalidade quando se lida com extração de características? Ilustre isso considerando 2 exemplos que façam uso de stemização e/ou lemazação. 

```bash

Stemização:
A stemização é o processo de reduzir as palavras à sua forma de base ou raiz (stem).
Reduz as palavras às suas formas mais simples, removendo afixos e sufixos.
Exemplo:
Palavras: "correr", "corria", "correria"
Após a stemização: "corr"

Lemmatização:
A lematização é semelhante à stemização, mas leva em consideração a morfologia e o contexto das palavras.
Reduz as palavras à sua forma de lema, que é a forma de dicionário ou base.
Exemplo:
Palavras: "correr", "corria", "correria"
Após a lematização: "correr"

Conclusão:
Tanto a stemização quanto a lematização reduzem a dimensionalidade, pois mapeiam várias formas de palavras para uma forma base.
Isso é benéfico para a extração de características, pois reduz o número de características únicas,
tornando os dados mais gerenciáveis e economizando recursos computacionais.
A redução de dimensionalidade é especialmente importante quando se aplica técnicas de modelagem de tópicos,
classificação de texto e agrupamento, pois ajuda a lidar com a esparsidade dos dados e a melhorar o desempenho do modelo.

Pesquisa:
https://www.alura.com.br/artigos/lemmatization-vs-stemming-quando-usar-cada-uma?utm_term=&utm_campaign=%5BSearch%5D+%5BPerformance%5D+-+Dynamic+Search+Ads+-+Artigos+e+Conte%C3%BAdos&utm_source=adwords&utm_medium=ppc&hsa_acc=7964138385&hsa_cam=11384329873&hsa_grp=111087461203&hsa_ad=662261334153&hsa_src=g&hsa_tgt=dsa-843358956400&hsa_kw=&hsa_mt=&hsa_net=adwords&hsa_ver=3&gclid=CjwKCAjwyY6pBhA9EiwAMzmfwdldK2BOsvGBiWkxGp4okhzIQ8HVxpDhlKa6MSab9yeQExsT2uTzdhoCKEYQAvD_BwE
```

TC.3.3. Descreva como ficaria o léxico do modelo e o vetor de características n-gram para n=1, 2 e 3 para os seguintes documentos pertencentes ao mesmo corpus: 

Frase 1: Eu não gostei do produto e o produto parece ruim. 

Frase 2: O produto parece bom. 

Frase 3: O produto parece ruim. 

```bash

N-grams são sequências contíguas de n elementos (geralmente palavras ou caracteres) em um texto.
Os n-grams capturam informações sobre a ordem e a proximidade das palavras em um texto.
Por exemplo, em bigrams, "Eu não" e "não gostei" representam combinações específicas de palavras que podem ter significados distintos.

Léxico: 
['gostei', 'produto', 'parece', 'ruim']

Unigram (n=1):
Os unigrams são simplesmente as palavras individuais na frase.
Frase 1: [1, 2, 1, 1] 
Frase 2: [0, 1, 1, 0]
Frase 3: [0, 1, 1, 1]


Bigram (n=2): 
Os bigrams são pares de palavras juntas na frase. Eles são criados deslizando uma janela de tamanho 2 pelas palavras da frase.
Frase 1: [0, 1, 1, 1, 1, 1, 0, 0, 0]
Frase 2: [0, 0, 1, 0, 1, 0, 0, 0, 0]
Frase 3: [0, 0, 1, 0, 1, 0, 0, 0, 1]
(Contagens de 'gostei do', 'do produto', 'produto e', 'e o', 'o produto', 'produto parece', 'parece ruim').

Trigram (n=3):
Os trigrams são conjuntos de três palavras juntas na frase. Eles são criados deslizando uma janela de tamanho 3 pelas palavras da frase.
Frase 1: [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0] 
Frase 2: [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
Frase 3: [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
(Contagens de 'gostei do produto', 'do produto e', 'produto e o', 'e o produto', 'o produto parece', 'produto parece ruim').

Pesquisa:
https://www.insightlab.ufc.br/pln-processamento-de-linguagem-natural-para-iniciantes/

```

TC.3.4. Para o exercício TC.3.3, considerando um modelo bag of words, com n=1, mostre como se calcula o valor da transformação TFIDF para as palavras produto e ruim, ambas na frase 1. Utilize as expressões fornecidas no material das aulas de PLN. 

```bash

Frequência do Termo (TF):
A frequência do termo mede a frequência com que uma palavra específica aparece em um documento. 
TF = 
(Número de vezes que a palavra aparece no documento) 
/  
(Número total de palavras no documento)


Frequência Inversa do Documento (IDF):
A frequência inversa do documento mede a raridade de uma palavra em todo o corpus de documentos. 
IDF = 
log(Número total de documentos no corpus 
/ 
Número de documentos que contêm a palavra)

Calcular o TF-IDF:
TF-IDF = TF * IDF

Frase: "Eu não gostei do produto e o produto parece ruim."
Palavras: “produto, ruim”

TF (Frequência do Termo):


Frase 1: "Eu não gostei do produto e o produto parece ruim."
Frase 2: "O produto parece bom."
Frase 3: "O produto parece ruim."

*TF (Frequência do Termo):*

Para “produto” na Frase 1:
TF (“produto”) = (2 / 9) ≈ 0.2222

Para "produto" na Frase 2:
TF ("produto") = (1 / 5) = 0.2

Para "produto" na Frase 3:
TF ("produto") = (1 / 5) = 0.2

Para "ruim" na Frase 1:
TF ("ruim") = (1 / 9) ≈ 0.1111

Para "ruim" na Frase 2:
TF ("ruim") = (0 /5) = 0 (a palavra "ruim" não está presente na Frase 2)

Para "ruim" na Frase 3:
TF ("ruim") = (1 / 5) = 0.2

*IDF (Frequência inversa do documento):*

IDF ("produto") = log(3 / 3) = 0 (a palavra "produto" está presente em todos os documentos)

IDF ("ruim") = log(3 / 2) ≈ 0.1761 (a palavra "ruim" está presente em dois dos três documentos)

*TF-IDF:*

TF-IDF para "produto" na Frase 1: (0.2222) * (0) = 0
TF-IDF para "produto" na Frase 2: (0.2) * (0) = 0
TF-IDF para "produto" na Frase 3: (0.2) * (0) = 0

TF-IDF para "ruim" na Frase 1: (0.1111) * (0.1761) ≈ 0.01959
TF-IDF para "ruim" na Frase 2: 0 (a palavra "ruim" não está presente na Frase 2)
TF-IDF para "ruim" na Frase 3: (0.2) * (0.1761) ≈ 0.03522


Portanto, o valor do TF-IDF na palavra "produto" é igual a 0. Nesse caso, a palavra produto têm um IDF de 0 porque estão presentes em todos os documentos do corpus, já a palavra "ruim" aparece em 2 dos documentos com um TF-IDF de 0.01959 e 0.03522, sendo assim ele tem uma maior importancia que a palavra "produto".

```

PRÁTICA DE PROGRAMAÇÃO 


PP.3.1. Baseando-se nos exemplos fornecidos pelo professor, que fazem uso da biblioteca scikitlearn, ilustre a obtenção de um vetor do po bag of words com transformação do po TFIDF para dois documentos que representem reviews de produtos em um site de e-commerce. Execute todas as etapas de pré-processamento necessárias para normalizar os dados. 

<a href="https://github.com/annakks/PLN/blob/main/Lista_3/pp31.py">Ir para o Código</a>

PP.3.2. Considerando um corpus de texto contendo revisões de produtos, selecione algumas revisões que possam ser caracterizadas como positivas, negativas ou neutras. Treine cada um dos modelos seguintes, tendo como base o código-fonte fornecido pelo professor, para que sejam capazes de classificar uma determinada revisão informada pelo usuário, diferente daquela que foi utilizada no treinamento do modelo. Salve os dados do seu modelo treinado em um arquivo pickle, recarregue e demonstre a sua utilização para nova classificação. a) Mullayer perceptron; b) K-Nearest Neighbors

<a href="https://github.com/annakks/PLN/blob/main/Lista_3/pp32.py">Ir para o Código</a>

PP.3.3. Demonstre a técnica de agrupamento hierárquico de documentos similares utilizando alguns dados de reviews de produtos. Ilustre e explique o dendrograma em especial no que se refere aos pontos de corte para as distâncias. Faça uso de dados de reviews de produtos e não se esqueça de normalizar os dados antes de efetuar a montagem dos vetores de características. 

<a href="https://github.com/annakks/PLN/blob/main/Lista_3/pp33.py">Ir para o Código</a>

PP.3.4. Demonstre a modelagem de tópicos, com LDA, utilizando alguns documentos representativos de revisões de produtos. Efetue todas as etapas de pré-processamento adequadas antes de efetuar a modelagem. 

<a href="https://github.com/annakks/PLN/blob/main/Lista_3/pp34.py">Ir para o Código</a>



