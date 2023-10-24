from pickleshare import pickle
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

base_model_lexicon = ['a', 'ajuda', 'ajudar', 'confusão', 'bom', 'mau',
                    'atendimento', 'como', 'confuso',
                    'consigo', 'de', 'desejo', 'encerrar',
                    'estou', 'favor', 'gostaria', 'há', 'mais',
                    'me', 'não', 'obter', 'opção', 'outra', 'poderia',
                    'por', 'qual', 'sei', 'é', 'essa']

def build_feature_vector(words, model_lexicon):
        bag_of_words_count = np.zeros(len(model_lexicon))
        for pos in range(len(model_lexicon)):
            for word in words:
                if word == model_lexicon[pos]:
                    bag_of_words_count[pos] += 1
        return bag_of_words_count

def build_model_lexicon(words, model_lexicon):
        for word in words:
            if word not in model_lexicon:
                model_lexicon.append(word)
        model_lexicon.sort()



def mlp():

    classified_reviews= [
        {'corpus': "adorei o produto", 'review_type': 'positive', 'feature_vector': []},
        {'corpus': "amei o produto ", 'review_type': 'positive', 'feature_vector': []},
        {'corpus': "achei o produto excelente", 'review_type': 'positive', 'feature_vector': []},
        {'corpus': "com certeza comprarei novamente ", 'review_type': 'positive', 'feature_vector': []},
        {'corpus': "não adorei o produto", 'review_type': 'negative', 'feature_vector': []},
        {'corpus': "detestei o produto ", 'review_type': 'negative', 'feature_vector': []},
        {'corpus': "produto horrível ", 'review_type': 'negative', 'feature_vector': []},
        {'corpus': "nunca mais compro novamente ", 'review_type': 'negative', 'feature_vector': []},
        {'corpus': "pior produto que comprei ", 'review_type': 'negative', 'feature_vector': []},
        {'corpus': "produto satisfatório", 'review_type': 'neutral', 'feature_vector': []},
        {'corpus': "produto atende as necessidades", 'review_type': 'neutral', 'feature_vector': []},
        {'corpus': "produto ok", 'review_type': 'neutral', 'feature_vector': []},
        {'corpus': "produto razoável", 'review_type': 'neutral', 'feature_vector': []}
    ]


    unclassified_review = {
        'corpus': 'Não adorei o produto, não estava tudo ok',
        'review_type': '',
        'feature_vector': []
    }

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

    return classifier.classes_


def knn():

    classified_reviews= [
        {'corpus': "amei o produto", 'review_type': 'positive', 'feature_vector': []},
        {'corpus': "não adorei o produto ", 'review_type': 'negative', 'feature_vector': []},
        {'corpus': "produto satisfatório", 'review_type': 'neutral', 'feature_vector': []}
    ]

    unclassified_review = {
        'corpus': 'Adorei o produto, estava tudo ok',
        'review_type': '',
        'feature_vector': []
    }

    # Here we build the model lexicon
    for classified_review in classified_reviews:
        build_model_lexicon(classified_review['corpus'].split(), base_model_lexicon)
    build_model_lexicon(unclassified_review['corpus'].split(), base_model_lexicon)

    # Now we extract the feature vector considering the model
    for classified_review in classified_reviews:
        classified_review['feature_vector'] = build_feature_vector(classified_review['corpus'].split(), base_model_lexicon)

    unclassified_review['feature_vector'] = build_feature_vector(unclassified_review['corpus'].split(), base_model_lexicon)

    # Now we perform the classification:
    dot_product_values = []
    for classified_review in classified_reviews:
        dot_product_values.append({"class":classified_review['review_type'],
                                "score": np.dot(unclassified_review['feature_vector'], classified_review['feature_vector'])})
    dot_product_sorted = sorted(dot_product_values, key=lambda d: d['score'], reverse=True)
    print(dot_product_sorted)
    unclassified_review['review_type'] = dot_product_sorted[0]['class']
    print("Review classified as: ", unclassified_review['review_type'])

    return


with open('modelo_mlp.pkl', 'wb') as arquivo_mlp:
    pickle.dump(mlp(), arquivo_mlp)

with open('modelo_mlp.pkl', 'rb') as arquivo_mlp:
    loaded_classifier_mlp = pickle.load(arquivo_mlp)

print("\n\n")
print ("leu: ", loaded_classifier_mlp)

new_unclassified_review_mlp = {
    'corpus': 'Adorei o produto, estava tudo ok',
    'review_type': '',
    'feature_vector': []
}

new_unclassified_review_mlp['feature_vector'] = build_feature_vector(new_unclassified_review_mlp['corpus'].split(), base_model_lexicon)

# Fazer previsão com o modelo carregado
predicted_class = loaded_classifier_mlp.predict([new_unclassified_review_mlp['feature_vector']])
print("Nova revisão classificada como:", predicted_class[0])


modelo_knn = knn()

with open('modelo_knn.pkl', 'wb') as arquivo_knn:
    pickle.dump(modelo_knn, arquivo_knn)

with open('modelo_knn.pkl', 'rb') as arquivo_knn:
    loaded_classifier_knn = pickle.load(arquivo_knn)

print("\n\n")
print (loaded_classifier_knn)

new_unclassified_review_knn = {
    'corpus': 'Adorei o produto, estava tudo ok',
    'review_type': '',
    'feature_vector': []
}

new_unclassified_review_knn['feature_vector'] = build_feature_vector(new_unclassified_review_knn['corpus'].split(), base_model_lexicon)

new_unclassified_review_knn['review_type'] = loaded_classifier_knn

print("Nova revisão classificada como:", new_unclassified_review_knn['review_type'])