"""
 model_test.py
 Este script separa un dataset en X_train, X_test, Y_train, Y_test.
"""

# DataBase access
# Natural Language Processing
import argparse
import pickle
# Data Manipulation
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
# Machine Learning
# Performance Evaluation and Support
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from ml_classifiers import getClassifier, kernels, getGridSearchCV
# next should be removed when all models finished
from ml_classifiers import *


if __name__ == '__main__':
    p = Path.cwd()

    parser = argparse.ArgumentParser(description='Not implemented yet')

    parser.add_argument("-o", "--option", type=str, help="Select an option in the script")

    parser.add_argument('--input_filename', '-if', type=str, help='Name of the input file', default='tfidf_df')

    args = parser.parse_args()

    # python app\model_test.py -o [svm, nb, rf, lr]
    tfidf_matrix_path = f"{str(p)}/data/tfidf_matrices"
    dataset_path = f"{str(p)}/data/prep_datasets"
    models_path = f"{str(p)}/data/models"

    input_filename = args.input_filename

    # recoger la matriz tfidf
    tfidf_df = pd.read_csv(f'{tfidf_matrix_path}/{input_filename}.csv')

    # recoger la matriz tfidf
    # prep_df = pd.read_csv(f'{dataset_path}/prep_df.csv')

    print(f'Loaded Dataframe: \n{tfidf_df}')

    # Convierto los flairs a un identificador numérico en una nueva columna
    tfidf_df['flair_id'] = tfidf_df['flair'].factorize()[0]

    # Las 3 siguientes lineas muestran el identificador numérico asociado al flair y la cantidad de cada flair que tenemos en el dataset
    flair_ids = tfidf_df[['flair', 'flair_id']].drop_duplicates().sort_values('flair_id')
    flair_ids['value_counts'] = tfidf_df['flair'].value_counts(sort=False).tolist()
    print(f'Flair IDs and Counts: \n{flair_ids}')


    tfidf_corpus = []
    for x in np.array(tfidf_df['tfidf_corpus']):
        documento = []
        palabra = []
        doc_temp = x[1:-1].split(',')
        for y in doc_temp:
            palabra = float(y)
            documento.append(palabra)
        tfidf_corpus.append(np.array(documento))


    # Splitting 20% of the data into train test split
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(tfidf_corpus, tfidf_df['flair_id'],
                                                        test_size=0.3,
                                                        random_state=2022)
    model = None

    if args.option == 'svm':
        # python .\app\model_test.py -o svm

        for i in range(4):
            svclassifier = getClassifier(i)
            svclassifier.fit(X_train_tfidf, y_train)  # Make prediction
            y_pred = svclassifier.predict(X_test_tfidf)  # Evaluate our model
            print("Evaluation:", kernels[i], "kernel")
            print(classification_report(y_test, y_pred))

        grid = getGridSearchCV()
        grid.fit(X_train_tfidf, y_train)

        # print best parameter after tuning
        print(f"Best params: {grid.best_params_}")
        # print how our model looks after hyper-parameter tuning
        print(f"Best estimator: {grid.best_estimator_}")

        grid_predictions = grid.predict(X_test_tfidf)
        print(f"Confusion matrix:\n{confusion_matrix(y_test, grid_predictions)}")
        print(f"Classification report:\n{classification_report(y_test, grid_predictions)}")

    elif args.option == 'rf':
        # python .\app\model_test.py -o rf

        """model = RandomForestClassifier().fit(X_train_tfidf, y_train)
        # guardar model en archivo
        with open(f'{models_path}/rf.pickle', 'wb') as handle:
            pickle.dump(model, handle)
            print(f'\nSaved current model in {models_path}/rf.pickle')

        y_predictions = model.predict(X_test_tfidf)
        print(classification_report(y_test, y_predictions))"""

        rf_random = getRandomizedSearchCV()
        rf_random.fit(X_train_tfidf, y_train)

        # print best parameter after tuning
        print(f"Best params: {rf_random.best_params_}")
        # print how our model looks after hyper-parameter tuning
        print(f"Best estimator: {rf_random.best_estimator_}")

        base_model = RandomForestRegressor(n_estimators=10, random_state=2022)
        base_model.fit(X_train_tfidf, y_train)
        base_accuracy = evaluateRFmodel(base_model, X_test_tfidf, y_test)

        best_random = rf_random.best_estimator_
        random_accuracy = evaluateRFmodel(best_random, X_test_tfidf, y_test)

        print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

        # print(random_forest_classifier(X_train_tfidf, X_test_tfidf, y_train, y_test))

    elif args.option == 'nb':
        # python .\app\model_test.py -o nb

        print(nb_classifier(X_train_tfidf, X_test_tfidf, y_train, y_test))

    elif args.option == 'lr':
        # python .\app\model_test.py -o lr

        print(log_reg_classifier(X_train_tfidf, X_test_tfidf, y_train, y_test))

    elif args.option == 'lda':
        # python .\app\model_test.py -o lda

        model = SVC().fit(X_train_tfidf, y_train)

        y_predictions = model.predict(X_test_tfidf)
        print(classification_report(y_test, y_predictions))

        fine_tuning(X_train_tfidf, y_train)

    # guardar model en archivo
    with open(f'{models_path}/svm.pickle', 'wb') as handle:
        pickle.dump(model, handle)
        print(f'\nSaved current model in {models_path}/svm.pickle')

    """
    # Naive Bayes Classifier
    def nb_classifier(X_train, X_test, y_train, y_test):
        nb_fit = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('model', MultinomialNB()),
                           ])
        nb_fit.fit(X_train, y_train)  # Fitting the data to the trianing data
    
        # Making Predictions on the test data
        y_pred = nb_fit.predict(X_test)
        acc = accuracy_score(y_pred=y_pred, y_true=y_test)
        f_score_micro = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
        f_score_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
        precision_micro = precision_score(y_pred=y_pred, y_true=y_test, average='micro')
        precision_macro = precision_score(y_pred=y_pred, y_true=y_test, average='macro')
        recall_micro = recall_score(y_pred=y_pred, y_true=y_test, average='micro')
        recall_macro = recall_score(y_pred=y_pred, y_true=y_test, average='macro')
    
        print(f"Model Accuracy: {acc}, "
              f"\nF_Score_micro: {f_score_micro}, F_Score_macro: {f_score_macro},"
              f"\nPrecision_micro: {precision_micro}, Precision_macro: {precision_macro},"
              f"\nRecall_micro: {recall_micro}, Recall_macro: {recall_macro}")
    
    
    # Random Forest Classifier
    def random_forest(X_train, X_test, y_train, y_test):
        forest = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('model', RandomForestClassifier()),
                           ])
        forest.fit(X_train, y_train)  # Fitting the data to the trianing data
    
        # Making Predictions on the test data
        y_pred = forest.predict(X_test)
        acc = accuracy_score(y_pred=y_pred, y_true=y_test)
        f_score_micro = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
        f_score_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
        precision_micro = precision_score(y_pred=y_pred, y_true=y_test, average='micro')
        precision_macro = precision_score(y_pred=y_pred, y_true=y_test, average='macro')
        recall_micro = recall_score(y_pred=y_pred, y_true=y_test, average='micro')
        recall_macro = recall_score(y_pred=y_pred, y_true=y_test, average='macro')
    
        print(f"Model Accuracy: {acc}, "
              f"\nF_Score_micro: {f_score_micro}, F_Score_macro: {f_score_macro},"
              f"\nPrecision_micro: {precision_micro}, Precision_macro: {precision_macro},"
              f"\nRecall_micro: {recall_micro}, Recall_macro: {recall_macro}")
    
    
    # Support Vector Machines Classifier
    def svc(X_train, X_test, y_train, y_test):
        svc_fit = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('model', SVC()),
                            ])
        svc_fit.fit(X_train, y_train)  # Fitting the data to the trianing data
    
        # Making Predictions on the test data
        y_pred = svc_fit.predict(X_test)
        acc = accuracy_score(y_pred=y_pred, y_true=y_test)
        f_score_micro = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
        f_score_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
        precision_micro = precision_score(y_pred=y_pred, y_true=y_test, average='micro')
        precision_macro = precision_score(y_pred=y_pred, y_true=y_test, average='macro')
        recall_micro = recall_score(y_pred=y_pred, y_true=y_test, average='micro')
        recall_macro = recall_score(y_pred=y_pred, y_true=y_test, average='macro')
    
        print(f"Model Accuracy: {acc}, "
              f"\nF_Score_micro: {f_score_micro}, F_Score_macro: {f_score_macro},"
              f"\nPrecision_micro: {precision_micro}, Precision_macro: {precision_macro},"
              f"\nRecall_micro: {recall_micro}, Recall_macro: {recall_macro}")
    
    
    # Logistic Regression Classifier
    def log_reg(X_train, X_test, y_train, y_test):
        logreg = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('model', LogisticRegression()),
                           ])
        logreg.fit(X_train, y_train)  # Fitting the data to the trianing data
    
        # Making Predictions on the test data
        y_pred = logreg.predict(X_test)
        acc = accuracy_score(y_pred=y_pred, y_true=y_test)
        f_score_micro = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
        f_score_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
        precision_micro = precision_score(y_pred=y_pred, y_true=y_test, average='micro')
        precision_macro = precision_score(y_pred=y_pred, y_true=y_test, average='macro')
        recall_micro = recall_score(y_pred=y_pred, y_true=y_test, average='micro')
        recall_macro = recall_score(y_pred=y_pred, y_true=y_test, average='macro')
    
        print(f"Model Accuracy: {acc}, "
              f"\nF_Score_micro: {f_score_micro}, F_Score_macro: {f_score_macro},"
              f"\nPrecision_micro: {precision_micro}, Precision_macro: {precision_macro},"
              f"\nRecall_micro: {recall_micro}, Recall_macro: {recall_macro}")
    
    
    print("\nEvaluate Naive Bayes Classifier")
    nb_classifier(X_train, X_test, y_train, y_test)
    
    print("\nEvaluate Random Forest Classifier")
    random_forest(X_train, X_test, y_train, y_test)
    
    print("\nEvaluate Logistic Regression Model")
    log_reg(X_train, X_test, y_train, y_test)
    
    print("\nEvaluate SVC Model")
    svc(X_train, X_test, y_train, y_test)
    """
