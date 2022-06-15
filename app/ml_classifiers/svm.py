# Support Vector Machines Classifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from .utils import get_predictions
from sklearn.metrics import classification_report

kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

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

#A function which returns the corresponding SVC model
def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return SVC(kernel='linear', gamma="auto")

def getGridSearchCV():
    return GridSearchCV(SVC(),param_grid,refit=True,verbose=3)