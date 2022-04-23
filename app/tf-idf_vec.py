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

path = Path.cwd()

parser = argparse.ArgumentParser(description='Takes submissions from database,'
                                             '\n preprocess its body and'
                                             '\n saves them to a pickle file.')
parser.add_argument("-o", "--option", type=str, help="Select an option in the script")
parser.add_argument("-min_gram", "--min_gram", type=str, help="Minimum grammar in tfidf")
parser.add_argument("-max_gram", "--max_gram", type=str, help="Maximum grammar in tfidf")

parser.add_argument('--output-filename', type=str, help='Name of the output file')
parser.add_argument('--extra_spaces', nargs='?', metavar='', const=False, type=bool, default=True,
                    help='Cleantext removes extra spaces')
parser.add_argument('--lowercase', nargs='?', const=False, type=bool, default=True, help='Cleantext sets lowercase')
parser.add_argument('--numbers', nargs='?', const=False, type=bool, default=True, help='Cleantext removes numbers')
parser.add_argument('--punct', nargs='?', const=False, type=bool, default=True, help='Cleantext removes punct')
parser.add_argument('--stopwords', nargs='?', const=False, type=bool, default=True, help='Cleantext removes stopwords')

group_args = parser.add_mutually_exclusive_group()
group_args.add_argument('-l', '--lemmatization', nargs='?', const=True, type=bool, default=False,
                        help='Reduce las palablas a su lema')
group_args.add_argument('-s', '--stemming', nargs='?', const=True, type=bool, default=False,
                        help='Reduce las palabras a su raiz')

args = parser.parse_args()

if args.option == "preprocessing_texts":
    # python app\tf-idf_vec.py -o preprocessing_texts

    filename = f"{str(path)}/data/prep_tf-idf.p"
    filename = "prep_text.p" if args.output_filename is None else args.output_filename
    prep_text = pd.read_pickle(f'{path}/data/{filename}')
    preprocessing_export_route = f"{str(path)}/data/texts_preprocessing.pkl"

    combined_texts = prep_text['combined']
    flairs = prep_text["flair"]
    flair_texts = []

    clean_texts = preprocessing_pipeline(combined_texts, cleantext=True, lemmatization=False, stemming=False,
                                         stpwrds=True)

    for i, flair in enumerate(flairs):
        flair_texts.append([ flair, clean_texts[i] ])

    pickle.dump(flair_texts, open(preprocessing_export_route, 'wb'))


if args.option == "tfidf":
    # python app\tf-idf_vec.py -o tfidf -min_gram 1 -max_gram 3

    preprocessing_route = f"{str(path)}/data/texts_preprocessing.pkl"
    min_gram = int(args.min_gram)
    max_gram = int(args.max_gram)

    flair_texts = pickle.load(open(preprocessing_route, 'rb'))

    flairs = [flair[0] for flair in flair_texts]
    texts = [texts[1] for texts in flair_texts]

    tf_idf_vect = TfidfVectorizer(ngram_range = (min_gram, max_gram))
    X_train_tf_idf = tf_idf_vect.fit_transform(texts).toarray()
    terms = tf_idf_vect.get_feature_names()

    # this is the matrix ! #
    for i, flair in enumerate(flairs):
        #first flair and tfidf vector (all vectors have the same size)
        """
        the flair is the result of the prediction (y_pred) and 
        the vector corresponding to the tfidf will be the features 
        that help predict (x_pred) when performing ML methods
        """
        print(f"{flair}: {X_train_tf_idf[i]}")

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