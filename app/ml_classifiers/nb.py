# Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import get_predictions

def nb_classifier(X_train, X_test, y_train, y_test):
    nb_fit = Pipeline([('model', MultinomialNB()),
                       ])
    nb_fit.fit(X_train, y_train)  # Fitting the data to the trianing data

    # Making Predictions on the test data
    y_pred = nb_fit.predict(X_test)

    return get_predictions(y_pred, y_test)

