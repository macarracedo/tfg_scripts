"""
 tf-idf_vec.py
 Este script coge submissions de la BD, lleva a cabo un preprocesado del texto
 parametrizable. Esto es, se podrán modificar mediante argumentos los "filtros" que se le aplican al texto
 (stopwords, numeros, puntuación, apóstrofes, minúsculas, lematización o stemización, etc.).
 Seguidamente guardará el dataframe preprocesado en un pickle.
"""
import gc       # garbage-collector
import argparse
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocessing_pipeline
import numpy as np
from sklearn.feature_selection import chi2

if __name__ == '__main__':

    p = Path.cwd()
    print(f'\nCurrent path: {str(p)}')

    parser = argparse.ArgumentParser(description='Takes submissions from database,'
                                                 '\n preprocess its body and'
                                                 '\n saves them to a pickle file.')

    parser.add_argument("-min_gram", "--min_gram", type=str, help="Minimum grammar in tfidf", default=1)
    parser.add_argument("-max_gram", "--max_gram", type=str, help="Maximum grammar in tfidf", default=2)

    parser.add_argument('--input_filename', '-if', type=str, help='Name of the input file', default='prep_df')
    parser.add_argument('--output_filename', '-of', type=str, help='Name of the output file', default='tfidf_df')

    args = parser.parse_args()
    print(str(args))

    # python app\tf-idf_vec.py -min_gram 1 -max_gram 3
    dataset_path = f"{str(p)}/data/prep_datasets"
    tfidf_matrix_path = f"{str(p)}/data/tfidf_matrices"
    input_filename = args.input_filename
    output_filename = args.output_filename

    min_gram = int(args.min_gram)
    max_gram = int(args.max_gram)

    prep_df = pd.read_csv(f'{dataset_path}/{input_filename}.csv')
    print(f'Loaded Dataframe: \n{prep_df}')
    # prep_df es un dataframe con las columnas [flair, result]

    print("Flair Count: \n" + str(prep_df['flair'].value_counts()))

    # texts = [texts for texts in prep_df['result'].values.astype('U')]
    texts = prep_df['result'].values.astype('U')
    flairs = prep_df['flair']

    del prep_df

    tfidf_vect = TfidfVectorizer(input="content",
                                 encoding="utf-8",
                                 decode_error="strict",
                                 strip_accents=None,
                                 lowercase=True,
                                 preprocessor=None,
                                 tokenizer=None,
                                 analyzer="word",
                                 stop_words=None,
                                 token_pattern=r"(?u)\b\w\w+\b",
                                 ngram_range=(min_gram, max_gram),
                                 max_df=1.0,
                                 min_df=1,
                                 max_features=None,
                                 vocabulary=None,
                                 binary=False,
                                 dtype=np.float64,
                                 norm="l2",
                                 use_idf=True,
                                 smooth_idf=True,
                                 sublinear_tf=False)


    X_tfidf = tfidf_vect.fit_transform(texts).toarray()

    del texts

    gc.collect()

    tfidf_df = pd.DataFrame()
    tfidf_df['flair'] = flairs

    # tfidf_df['tfidf_corpus'] = X_tfidf.toarray()                      # too big to handle
    # tfidf_df = tfidf_df.assign( tfidf_corpus = X_tfidf.toarray())     # too big to handle
    tfidf_df['tfidf_corpus'] = X_tfidf.tolist()

    """
    Awful efficiency. Do not use. I'm leaving it here so I don't run into this solution again.
    for i, x in enumerate(flairs):
        # print(f'i:{i} | x:{x} \t| X_tfidf.toarray()[i]: {X_tfidf.toarray()[i]}')
        tfidf_df.insert(i,                              # index
                        'tfidf_corpus',                 # column name
                        X_tfidf.toarray()[i],           # data
                        True)                           # allow duplicates
    """

    print(f"\nResulting Dataframe: \n{tfidf_df}")
    print(f"\nResulting Dataframe with flairs: \n{tfidf_df}")

    tfidf_df.to_csv(f'{tfidf_matrix_path}/{output_filename}.csv')

    # separator
    """
    # Sustituyo flairs por su homólogo numérico
    prep_df['id'] = prep_df['flair'].factorize()[0]
    flair_category = prep_df[['flair', 'id']].drop_duplicates().sort_values('id')
    print(f'Flair Category: \n{flair_category}')

    # Creo un diccionario de etiquetas
    category_labels = dict(flair_category.values)
    print(f'Category Labels: \n{category_labels}')

    # Extracting the features by fitting the Vectorizer on Combined Data
    labels = prep_df['id']  # Series containing all the post labels but id
    print(X_tfidf.shape)

    # chisq2 statistical test
    N = 5  # Number of examples to be listed
    for f, i in sorted(category_labels.items()):
        chi2_feat = chi2(X_tfidf, labels == i)
        indices = np.argsort(chi2_feat[0])
        feat_names = np.array(tfidf_vect.get_feature_names_out())[indices]
        for i in range(min_gram,max_gram+1):
            n_gram = [w for w in feat_names if len(w.split(' ')) == i]
            print("\nFlair '{}':".format(f))
            print("Most correlated n_grams({}):\n\t. {}".format(i,'\n\t. '.join(n_gram[-N:])))"""

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
