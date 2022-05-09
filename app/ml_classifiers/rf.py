# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import get_predictions

def random_forest_classifier(X_train, X_test, y_train, y_test):
    forest = Pipeline([('model', RandomForestClassifier()),
                       ])
    forest.fit(X_train, y_train)  # Fitting the data to the trianing data

    # Making Predictions on the test data
    y_pred = forest.predict(X_test)

    return get_predictions(y_pred, y_test)