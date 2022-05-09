# Support Vector Machines Classifier

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from .utils import get_predictions
from sklearn.metrics import classification_report


def svc_classifier(X_train, X_test, y_train, y_test):
    svc_fit = Pipeline([('tf-idf_vect', TfidfVectorizer(ngram_range=(1,1))),
                        ('model', SVC(C=10, gamma=0.01, kernel="rbf")),
                        ])
    svc_fit.fit(X_train, y_train)  # Fitting the data to the trianing data

    # Making Predictions on the test data
    y_pred = svc_fit.predict(X_test)

    return get_predictions(y_pred, y_test)

def svc(X_train_tfidf, X_test_tfidf, y_train, y_test, save=True):

    model = SVC(C=10, gamma=0.01, kernel="rbf").fit(X_train_tfidf, y_train)

    y_predictions = model.predict(X_test_tfidf)

    return classification_report(y_test, y_predictions)
