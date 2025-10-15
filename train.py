import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.inspection import permutation_importance

def train_eval_model(model, X_train, y_train, X_test, y_test,
                     feature_selection=True,
                     feature_importance=True):
    """
    Train and evaluate a classification model.

        Returns: model, tpr, fpr, optimal_treshold, performance_dict
    """
    # fit using sequential feature selector
    # default: 5 fold cross-validation
    if feature_selection:
        sfs = SequentialFeatureSelector(model, n_jobs=-1, tol=0.08, direction='backward', scoring='roc_auc')
        sfs.fit(X_train, y_train)

    # refit the model using only the selected features
    if feature_selection:
        X_train = X_train[X_train.columns[sfs.get_support()]]
        X_test = X_test[X_train.columns]  # ensure test set has the same features
    model.fit(X_train, y_train)

    # get feature and importances
    if feature_importance:
        result = permutation_importance(
            model, X_test, y_test, n_repeats=4, random_state=42, n_jobs=-1,
            scoring='roc_auc'
        )
        sorted_importances_idx = result.importances_mean.argsort()
        importances = pd.DataFrame(
            result.importances[sorted_importances_idx].T,
            columns=X_train.columns[sorted_importances_idx],
        )
    else:
        importances = None
    
    # get train threshold, that gives best F1 score
    y_train_prob = model.predict_proba(X_train)[:,1]
    fpr_train, tpr_train, thresholds = roc_curve(y_train, y_train_prob)
    optimal_idx = np.argmax(tpr_train - fpr_train)
    optimal_threshold = thresholds[optimal_idx]

    # evaluate on test set
    y_test_prob = model.predict_proba(X_test)[:,1]
    y_test_pred = (y_test_prob >= optimal_threshold).astype(int)

    # get precision, recall, f1-score
    report = classification_report(y_test, y_test_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_test_pred, normalize='true')
    # round the values in confusion matrix to 2 decimal places
    conf_matrix = np.round(conf_matrix, 2)
    auc = roc_auc_score(y_test, y_test_prob)
    performance_dict = {
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1-score': report['macro avg']['f1-score'],
        'auc': auc,
        'confusion_matrix': conf_matrix
    }

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)

    return model, tpr, fpr, optimal_threshold, performance_dict, importances