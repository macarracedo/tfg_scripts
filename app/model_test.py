"""
 model_test.py
 Este script separa un dataset en X_train, X_test, Y_train, Y_test.
"""

# DataBase access
# Natural Language Processing
import argparse

import nltk
# Data Manipulation
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
# Machine Learning
# Performance Evaluation and Support
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import argparse
from sklearn.metrics import classification_report
from pathlib import Path
from ml_classifiers import nb_classifier, svc_classifier, log_reg_classifier, random_forest_classifier

if __name__ == '__main__':
    p = Path.cwd()

    parser = argparse.ArgumentParser(description='Not implemented yet')

    parser.add_argument("-o", "--option", type=str, help="Select an option in the script")

    parser.add_argument('--input_filename', '-if', type=str, help='Name of the input file', default='tfidf_df')

    args = parser.parse_args()

    # python app\model_test.py -o [svm, nb, rf, lr]
    tfidf_matrix_path = f"{str(p)}/data/tfidf_matrices"
    dataset_path = f"{str(p)}/data/prep_datasets"

    input_filename = args.input_filename

    # recoger la matriz tfidf
    tfidf_df = pd.read_csv(f'{tfidf_matrix_path}/{input_filename}.csv')
    prep_df = pd.read_csv(f'{dataset_path}/prep_df.csv')

    print(f'Loaded Dataframe: \n{tfidf_df}')

    tfidf_df['flair_id'] = tfidf_df['flair'].factorize()[0]
    flair_ids = tfidf_df[['flair', 'flair_id']].drop_duplicates().sort_values('flair_id')
    flair_ids['value_counts'] = tfidf_df['flair'].value_counts(sort=False).tolist()
    print(f'Flair IDs and Counts: \n{flair_ids}')

    prep_df['flair_id'] = prep_df['flair'].factorize()[0]


    # Splitting 20% of the data into train test split
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(tfidf_df['tfidf_corpus'], tfidf_df['flair_id'],
                                                        test_size=0.3,
                                                        random_state=2022)

    """X_train_tfidf_2, X_test_tfidf_2, y_train_2, y_test_2 = train_test_split(prep_df['result'], prep_df['flair_id'],
                                                                    test_size=0.3,
                                                                    random_state=2022)"""


    # Creating an instance of the TFID Vectorizer

    tfidf_vect = TfidfVectorizer(ngram_range=(1,1))
    
    # tranforma el corpus en ngramas de nuevo, si guardo en pickle lo anterior no lo hago, sino que cargo ese pickle
   
    """X_train_tfidf_2 = tfidf_vect.fit_transform(X_train_tfidf_2)
    X_test_tfidf_2 = tfidf_vect.transform(X_test_tfidf_2)"""

    """print(f'X_train_tfidf.get_type(): {X_train_tfidf.get_type()}' \ # type= Series
          f'X_train_tfidf: \n{X_train_tfidf}')"""

    print(f'X_train_tfidf._to_numpy(): \n{X_train_tfidf.to_numpy()}')

    """print(f'X_train_tfidf_2._get_dtype(): {X_train_tfidf_2._get_dtype()}' \
          f'X_train_tfidf_2: \n{X_train_tfidf_2}')"""


    if args.option == 'svm':
        # python .\app\model_test.py -o svm

        model = SVC(C=10, gamma=0.01, kernel="rbf").fit(X_train_tfidf, y_train)
        # guardar model en archivo
        y_predictions = model.predict(X_test_tfidf)
        print(classification_report(y_test, y_predictions))

    elif args.option == 'nb':
        # python .\app\model_test.py -o nb

        print(nb_classifier(X_train_tfidf, X_test_tfidf, y_train, y_test))
    elif args.option == 'rf':
        # python .\app\model_test.py -o rf

        print(random_forest_classifier(X_train_tfidf, X_test_tfidf, y_train, y_test))
    elif args.option == 'lr':
        # python .\app\model_test.py -o lr

        print(log_reg_classifier(X_train_tfidf, X_test_tfidf, y_train, y_test))


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
