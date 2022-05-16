"""
utils.py
functions used by all classifiers
"""
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pickle

def save_model(model, output_filename='model', output_path=''):
    toret = True
    try:
        with open(f'{output_path}/{output_filename}.pkl', 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        toret = False

    return toret

def get_predictions(y_pred, y_test):

    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    f_score_micro = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
    f_score_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    precision_micro = precision_score(y_pred=y_pred, y_true=y_test, average='micro')
    precision_macro = precision_score(y_pred=y_pred, y_true=y_test, average='macro')
    recall_micro = recall_score(y_pred=y_pred, y_true=y_test, average='micro')
    recall_macro = recall_score(y_pred=y_pred, y_true=y_test, average='macro')
    recall_weighted = recall_score(y_pred=y_pred, y_true=y_test, average='weighted')

    return f"Model Accuracy: {acc}, " \
           f"\nF_Score_micro: {f_score_micro}, F_Score_macro: {f_score_macro}," \
           f"\nPrecision_micro: {precision_micro}, Precision_macro: {precision_macro}," \
           f"\nRecall_micro: {recall_micro}, Recall_macro: {recall_macro}, Recall_Weighted: {recall_weighted}"