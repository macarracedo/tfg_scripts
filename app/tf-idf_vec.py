"""
 tf-idf_vec.py
 Este script coge submissions de la BD, lleva a cabo un preprocesado del texto
 parametrizable. Esto es, se podrán modificar mediante argumentos los "filtros" que se le aplican al texto
 (stopwords, numeros, puntuación, apóstrofes, minúsculas, lematización o stemización, etc.).
 Seguidamente guardará el dataframe preprocesado en un pickle.
"""
import argparse
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocessing_pipeline
import numpy as np
from sklearn.feature_selection import chi2

path = Path.cwd()

parser = argparse.ArgumentParser(description='Takes submissions from database,'
                                             '\n preprocess its body and'
                                             '\n saves them to a pickle file.')

parser.add_argument("-min_gram", "--min_gram", type=str, help="Minimum grammar in tfidf")
parser.add_argument("-max_gram", "--max_gram", type=str, help="Maximum grammar in tfidf")

parser.add_argument('--output-filename', type=str, help='Name of the output file')

args = parser.parse_args()

# python app\tf-idf_vec.py -min_gram 1 -max_gram 3
preprocessing_route = f"{str(path)}/data/preprocessed_texts_df.pkl"
min_gram = int(args.min_gram)
max_gram = int(args.max_gram)

prep_texts = pd.read_pickle(preprocessing_route)
# flair_texts es un dataframe con las columnas [flair, combined, result]

flairs = [flair for flair in prep_texts['flair']]
texts = [texts for texts in prep_texts['result']]

tf_idf_vect = TfidfVectorizer(ngram_range=(min_gram, max_gram))
X_train_tf_idf = tf_idf_vect.fit_transform(texts).toarray()
terms = tf_idf_vect.get_feature_names()

# this is the matrix ! #
for i, flair in enumerate(flairs):
    # first flair and tfidf vector (all vectors have the same size)
    """
    the flair is the result of the prediction (y_pred) and 
    the vector corresponding to the tfidf will be the features 
    that help predict (x_pred) when performing ML methods
    """
    print(f"'{flair}': {X_train_tf_idf[i]}")

# separator
prep_texts['id'] = prep_texts['flair'].factorize()[0]
flair_category = prep_texts[['flair', 'id']].drop_duplicates().sort_values('id')
print(flair_category)

# Creo un diccionario de etiquetas
category_labels = dict(flair_category.values)
print(category_labels)

# Extracting the features by fitting the Vectorizer on Combined Data
feat = tf_idf_vect.fit_transform(texts).toarray()
labels = prep_texts['id']  # Series containing all the post labels
print(feat.shape)

# chisq2 statistical test
N = 5  # Number of examples to be listed
for f, i in sorted(category_labels.items()):
    chi2_feat = chi2(feat, labels == i)
    indices = np.argsort(chi2_feat[0])
    feat_names = np.array(tf_idf_vect.get_feature_names_out())[indices]
    unigrams = [w for w in feat_names if len(w.split(' ')) == 1]
    bigrams = [w for w in feat_names if len(w.split(' ')) == 2]
    trigrams = [w for w in feat_names if len(w.split(' ')) == 3]
    print("\nFlair '{}':".format(f))
    print("Most correlated unigrams:\n\t. {}".format('\n\t. '.join(unigrams[-N:])))
    print("Most correlated bigrams:\n\t. {}".format('\n\t. '.join(bigrams[-N:])))
    print("Most correlated trigrams:\n\t. {}".format('\n\t. '.join(trigrams[-N:])))

# separator

"""
tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(combined_texts)

    # get the first vector out (for the first document)
    first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

    # place tf-idf values in a pandas data frame
    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(),
                      index=tfidf_vectorizer.get_feature_names(),
                      columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)

"""
