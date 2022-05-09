# Logistic Regression Classifier

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .utils import get_predictions

def log_reg_classifier(X_train, X_test, y_train, y_test):
    logreg = Pipeline([('model', LogisticRegression()),
                       ])
    logreg.fit(X_train, y_train)  # Fitting the data to the trianing data

    # Making Predictions on the test data
    y_pred = logreg.predict(X_test)

    return get_predictions(y_pred, y_test)