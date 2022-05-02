# DataBase access
# Natural Language Processing
import argparse

import nltk
# Data Manipulation
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
# Statistical libraries
from sklearn.feature_selection import chi2
# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
# Performance Evaluation and Support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import argparse
import pickle
from pathlib import Path

path = Path.cwd()

parser = argparse.ArgumentParser(description='Takes submissions from database,'
                                             '\n preprocess its body and'
                                             '\n saves them to a pickle file.')
parser.add_argument("-o", "--option", type=str, help="Select an option in the script")

parser.add_argument('--output-filename', type=str, help='Name of the output file')

args = parser.parse_args()

preprocessing_route = f"{str(path)}/data/preprocessed_texts_df.pkl"
prep_texts = pd.read_pickle(preprocessing_route)
# flair_texts es un dataframe con las columnas [flair, combined, result]


# Splitting 20% of the data into train test split

X_train, X_test, y_train, y_test = train_test_split(prep_texts['result'], prep_texts['flair'],
                                                    test_size=0.2,
                                                    random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Creating an instance of the TFID transformer
count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(X_train)

# Creating an instance of the TFID transformer
tfidf_trans = TfidfTransformer()
X_train_tfidf = tfidf_trans.fit_transform(X_train_counts)


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
