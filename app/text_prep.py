"""
 text_prep.py
 Este script coge submissions de la BD, lleva a cabo un preprocesado del texto
 parametrizable. Esto es, se podrán modificar mediante argumentos los "filtros" que se le aplican al texto
 (stopwords, numeros, puntuación, apóstrofes, minúsculas, lematización o stemización, etc.).
 Seguidamente guardará el dataframe preprocesado en un pickle.
"""

# Parser cmd-line options and arguments
import argparse
# File manage
from pathlib import Path

# Data Manipulation
import pandas as pd
# Text preprocessing
from cleantext import clean
from nltk.stem import PorterStemmer, WordNetLemmatizer
# Natural Language Processing
from nltk.tokenize import word_tokenize
from sqlalchemy import select

# DataBase access
import db
from models import Submission
from preprocessing import preprocessing_pipeline


p = Path.cwd()
print(str(p))

filename = f"{str(p)}/data/prep_tf-idf.p"
data_path = f"{str(p)}/data"

parser = argparse.ArgumentParser(description='Takes submissions from database,'
                                             '\n preprocess its body and'
                                             '\n saves them to a pickle file.')
parser.add_argument('--output-filename', type=str, help='Name of the output file', default='preprocessed_texts_df')
parser.add_argument('--clean_text', nargs='?', const=False, type=bool, default=True, help="Don't use clean_text library")
parser.add_argument('--tokenize', nargs='?', const=True, type=bool, default=False, help='Tokenize result column')
parser.add_argument('--stopwords', nargs='?', const=False, type=bool, default=True, help="Don't remove stopwords")


group_args = parser.add_mutually_exclusive_group()
group_args.add_argument('-l', '--lemmatization', nargs='?', const=True, type=bool, default=False,
                        help='Reduce las palablas a su lema')
group_args.add_argument('-s', '--stemming', nargs='?', const=True, type=bool, default=False,
                        help='Reduce las palabras a su raiz')

args = parser.parse_args()
print(args)

filename = "preprocessed_texts_df" if args.output_filename is None else args.output_filename

# Recojo submissions de la base de datos a un dataframe
db.Base.metadata.create_all(db.engine)
query = select(Submission.link_flair_text, Submission.title, Submission.selftext).where(
    Submission.link_flair_text is not None)
result = db.session.execute(query).fetchall()

features = [
    'flair',
    'title',
    'body'
]
prep_df = pd.DataFrame(result, columns=features)

# Creo una columna combinando titulo y cuerpo, para aportar más información
prep_df['combined'] = prep_df['title']

for i in range(len(prep_df)):
    if type(prep_df.loc[i]['body']) != float:
        prep_df['combined'][i] = prep_df['combined'][i] + ' ' + prep_df['body'][i]
# Elimino columnas de Titulo y Cuerpo
prep_df.drop('title', inplace=True, axis=1)
prep_df.drop('body', inplace=True, axis=1)

# Uso el método del archivo clean_text de preprocessing y ponemos flairs en minúsculas.
prep_df['result'] = preprocessing_pipeline(prep_df['combined'], cleantext=args.clean_text, lemmatization=args.lemmatization, stemming=args.stemming,
                                         stpwrds=args.stopwords)

prep_df['flair'] = prep_df['flair'].apply(str.lower)

# Uso NLTK para tokenización.
if args.tokenize:
    prep_df['result'].apply(word_tokenize)

# Visualización y guardado del resultado
print(f"Result DataFrame: \n{prep_df}")

# Finalmente guardo el dataframe preprocesado en un pickle.
prep_df.to_pickle(f'{data_path}/{filename}.pkl')
prep_df.to_excel(f'{data_path}/{filename}.xlsx', sheet_name="pagina1")
print(f'saved preprocessed_texts_df in {data_path}/{filename}')