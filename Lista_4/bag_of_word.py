from sklearn.neural_network import MLPClassifier
import numpy as np

classified_reviews= [
        {'corpus': "Ótimo produto e entrega Super rapida chegou em menos de dois dias", 'review_type': 'positive', 'feature_vector': []},
        {'corpus': "Chegou bem antes do prazo Tudo otimo recomendo demais", 'review_type': 'positive', 'feature_vector': [ ]},
        {'corpus': "Ótimo produto Excelente qualidade Já usei e recomendo este produto", 'review_type': 'positive', 'feature_vector': []},
        {'corpus': "Produto de boa qualidade e recebi em perfeito estado entrega super rápida", 'review_type': 'positive', 'feature_vector': []},
        {'corpus': "Chegou no prazo correto Aparelho ótimo tudo conforme descrito no anúncio", 'review_type': 'positive', 'feature_vector': [ ]},

        {'corpus': "Não gostei do produto Qualidade duvidosa Não recomendo", 'review_type': 'negative', 'feature_vector': [ ]},
        {'corpus': "Ainda não recebi o produto e nem uma posição ou retorno de quando receberei", 'review_type': 'negative', 'feature_vector': []},
        {'corpus': "Comprei e me arrependi Produto de baixa qualidade veio com uma peça trincada Não recomendo", 'review_type': 'negative', 'feature_vector': [ ]},
        {'corpus': "Até hoje não recebi o produto e nem o dinheiro não recomendo essa americanas para ninguém", 'review_type': 'negative', 'feature_vector': []},
        {'corpus': "produto não é bom", 'review_type': 'negative', 'feature_vector': []},

        {'corpus': "muito simples pelo preço isso que da compra espontaneamente", 'review_type': 'neutral', 'feature_vector': [ ]},
        {'corpus': "Bom Conforme combinado chegou antes do prazo Tudo certo", 'review_type': 'neutral', 'feature_vector': []},
        {'corpus': "estou satisfeito com o produto bom e barato o meu não o entrou aguá", 'review_type': 'neutral', 'feature_vector': []},
        {'corpus': "A qualidade do produto é equivalente ao preço ou seja, é bom para o valor cobrado", 'review_type': 'neutral', 'feature_vector': []},
        {'corpus': "Nao gostei do material do produto O tamanho é ótimo chegou muito rápido do não gostei do material", 'review_type': 'neutral', 'feature_vector': []}
]
unclassified_review = {
    'corpus': 'parece ser razoável nunca mais compro novamente',
    'review_type': '',
    'feature_vector': []
}

base_model_lexicon = ['a', 'ajuda', 'ajudar', 'confusão', 'bom', 'mau', 
                 'atendimento', 'como', 'confuso', 
                 'consigo', 'de', 'desejo', 'encerrar', 
                 'estou', 'favor', 'gostaria', 'há', 'mais',
                   'me', 'não', 'obter', 'opção', 'outra', 'poderia', 
                   'por', 'qual', 'sei', 'é', 'essa']

# words used as inputs shall be those resulted from tokenization proccess and other
# preprocessing steps.
def build_model_lexicon(words, model_lexicon):
    for word in words:
        if word not in model_lexicon:
            model_lexicon.append(word)
    model_lexicon.sort()

def build_feature_vector(words, model_lexicon):
    bag_of_words_count = np.zeros(len(model_lexicon))
    for pos in range(len(model_lexicon)):
        for word in words:
            if word == model_lexicon[pos]:
                bag_of_words_count[pos] += 1
    return bag_of_words_count


# Here we build the model lexicon
for classified_review in classified_reviews:
    build_model_lexicon(classified_review['corpus'].split(), base_model_lexicon)
build_model_lexicon(unclassified_review['corpus'].split(), base_model_lexicon)

# Now we extract the feature vector considering the model
for classified_review in classified_reviews:
    classified_review['feature_vector'] = build_feature_vector(classified_review['corpus'].split(), base_model_lexicon)

unclassified_review['feature_vector'] = build_feature_vector(unclassified_review['corpus'].split(), base_model_lexicon)

X = [] # feature vectors
y = [] # feature classes
for review in classified_reviews:
    X.append(review['feature_vector'])
    y.append(review['review_type'])

classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 3), random_state=1)



classifier.fit(X, y)

print(classifier.predict([unclassified_review['feature_vector']]))
print(classifier.predict_proba([unclassified_review['feature_vector']]))
print(classifier.classes_)
