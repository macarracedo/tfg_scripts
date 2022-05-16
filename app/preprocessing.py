"""
 preprocessing.py
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
from preprocessing import preprocessing_pipeline
# Natural Language Processing
from nltk.tokenize import word_tokenize
from sqlalchemy import select
# DataBase access
from db import database_connect
from keeper import get_submissions
from models import Submission

if __name__ == '__main__':

    por_ciento = lambda x: x/100

    session = database_connect()

    p = Path.cwd()
    print(f'\nCurrent path: {str(p)}')

    parser = argparse.ArgumentParser(description='Takes submissions from database,'
                                                 '\n preprocess its body and'
                                                 '\n saves them to a pickle file.')
    parser.add_argument('--output_filename', '-of', type=str, help='Name of the output file', default='prep_df')
    parser.add_argument('--clean_text', '-ct', nargs='?', const=False, type=bool, default=True, help="Don't use clean_text library")
    parser.add_argument('--tokenize', '-t', nargs='?', const=True, type=bool, default=False, help='Tokenize result column')
    parser.add_argument('--stopwords', '-s', nargs='?', const=False, type=bool, default=True, help="Don't remove stopwords")
    parser.add_argument('--empty_flair', '-ef', nargs='?', const=False, type=bool, default=True, help="Don't remove flair '' ")
    parser.add_argument('--occurrence', '-o', nargs='?', const=False, type=bool, default=True, help="Don't remove texts w/flair that has very low occurrences")

    group_args = parser.add_mutually_exclusive_group()
    group_args.add_argument('-L', '--lemmatization', nargs='?', const=True, type=bool, default=False,
                            help='Reduce las palablas a su lema')
    group_args.add_argument('-S', '--stemming', nargs='?', const=True, type=bool, default=False,
                            help='Reduce las palabras a su raiz')
    args = parser.parse_args()
    print(str(args))

    MINIMUM_OCURRENCES = 1      # Percentage

    output_path = f"{str(p)}/data/prep_datasets"
    output_filename = args.output_filename

    # Recojo submissions de la base de datos a un dataframe
    """db.Base.metadata.create_all(db.engine)
    query = select(Submission.link_flair_text, Submission.title, Submission.selftext).where(
        Submission.link_flair_text is not None)

    result = db.session.execute(query).fetchall()"""

    result = get_submissions(session)
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

    # Elimino muestras con flair vacío
    if args.empty_flair:
        prep_df.drop(prep_df[prep_df['flair'].map(len) == 0].index, inplace=True, axis=0)
        # El índice de las filas no se actualiza tras eliminar filas,
        # pero al guardar en csv e importarlo sí lo hace en el nuevo df

    # Aprovecho para eliminar aquí flair con una presencia menor al 1%
    removed_flairs = []
    if args.occurrence:
        min_rep = int(por_ciento(MINIMUM_OCURRENCES) * len(prep_df.index))
        for index, value in prep_df['flair'].value_counts().items():
            if value < min_rep:
                removed_flairs.append(index)
                prep_df.drop(prep_df[prep_df['flair'] == index].index, inplace=True, axis=0)

    # Uso el método del archivo clean_text de preprocessing
    prep_df['result'] = preprocessing_pipeline(prep_df['combined'], cleantext=args.clean_text,
                                               lemmatization=args.lemmatization, stemming=args.stemming,
                                               stpwrds=args.stopwords)

    # Pongo flairs en minúsculas.
    prep_df['flair'] = prep_df['flair'].apply(str.lower)

    # Uso NLTK para tokenización.
    if args.tokenize:
        prep_df['result'] = prep_df['result'].apply(word_tokenize)

    # Elimino columnas innecesarias
    prep_df.drop('title', inplace=True, axis=1)
    prep_df.drop('body', inplace=True, axis=1)
    prep_df.drop('combined', inplace=True, axis=1)

    # Visualización del resultado
    print(f"\nRemoved Flairs {removed_flairs}\n"
          f"\nFlair count: \n{str(prep_df['flair'].value_counts())}\n"
          f"\nResulting Dataframe: \n{prep_df}")

    # Finalmente guardo el dataframe preprocesado
    prep_df.to_csv(f'{output_path}/{output_filename}.csv', index=False)
    print(f'\nSaved resulting dataframe in {output_path}/{output_filename}.csv')