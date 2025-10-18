import numpy as np
import pandas as pd
from sklearn.metrics import auc

def plot_model_results(idx, axes, model_name, tpr, fpr, performance_dict, importances):
    """
    Plot ROC curve, confusion matrix and print performance metrics.

        Parameters:
            idx: index of the model (for subplotting)
            axes: matplotlib axes array
            model_name: name of the model
            tpr: true positive rate
            fpr: false positive rate
            performance_dict: dictionary with precision, recall, f1-score, auc, confusion_matrix
    """

    # plot ROC curve
    axes[idx, 0].plot(fpr, tpr, label=f'{model_name} (AUC = {performance_dict["auc"]:.2f})')
    axes[idx, 0].plot([0, 1], [0, 1], 'k--')
    axes[idx, 0].set_xlabel('False Positive Rate')
    axes[idx, 0].set_ylabel('True Positive Rate')
    axes[idx, 0].set_title(f'ROC Curve - {model_name}')
    axes[idx, 0].legend(loc='lower right')

    # plot confusion matrix
    axes[idx, 1].imshow(performance_dict['confusion_matrix'], cmap='Blues')
    axes[idx, 1].set_xticks([0, 1])
    axes[idx, 1].set_yticks([0, 1])
    axes[idx, 1].set_xticklabels(['Predicted 0', 'Predicted 1'])
    axes[idx, 1].set_yticklabels(['Actual 0', 'Actual 1'])
    axes[idx, 1].set_title(f'Confusion Matrix - {model_name}: Prec / Rec / F1: {performance_dict["precision"]:.2f} / {performance_dict["recall"]:.2f} / {performance_dict["f1-score"]:.2f}')
    
    # plot bar plot of feature importances
    importances_mean = importances.mean().sort_values(ascending=False)
    importances_std = importances.std().sort_values(ascending=False)
    axes[idx, 2].bar(importances_mean.index, importances_mean.values, color='skyblue')
    # rotate x_labels by 90 degrees
    axes[idx, 2].tick_params(axis='x', rotation=90)
    axes[idx, 2].errorbar(importances_mean.index, importances_mean.values, yerr=importances_std.values, fmt='o', color='black', capsize=5)
    axes[idx, 2].set_title(f'Feature Importances - {model_name}')
    axes[idx, 2].set_ylabel('Permutation Importance (roc_auc)')
    axes[idx, 2].set_xlabel('Features')

    # Add text annotations to confusion matrix
    for i in range(2):
        for j in range(2):
            axes[idx, 1].text(j, i, performance_dict['confusion_matrix'][i, j], 
                             ha='center', va='center', color='white' if performance_dict['confusion_matrix'][i, j] > performance_dict['confusion_matrix'].max()/2 else 'black')

    return axes

def plot_repeated_model_results(idx, axes, model_name, tpr_list, fpr_list, performance_dict, importances_list):
    """
    Plot ROC curve, confusion matrix and print performance metrics.

        Parameters:
            idx: index of the model (for subplotting)
            axes: matplotlib axes array
            model_name: name of the model
            tpr: true positive rate
            fpr: false positive rate
            performance_dict: dictionary with precision, recall, f1-score, auc, confusion_matrix
    """

# plot ROC curve
    interp_len = 100
    for i in range(5):
        # interpolate 
        tpr_interp = np.interp(np.linspace(0, 1, interp_len), fpr_list[i], tpr_list[i])
        fpr_interp = np.linspace(0, 1, interp_len)
        # ROC AUC 
        roc_auc = auc(fpr_interp, tpr_interp)
        axes[idx, 0].plot(fpr_interp, tpr_interp, label=f'run {i+1} (AUC {roc_auc})', alpha=0.7, color='gray')
    axes[idx, 0].plot([0, 1], [0, 1], 'k--')
    axes[idx, 0].set_xlabel('False Positive Rate')
    axes[idx, 0].set_ylabel('True Positive Rate')
    axes[idx, 0].set_title(f'ROC Curve - {model_name}')
    axes[idx, 0].legend(loc='lower right')

    # plot bar plot of feature importances
    means = []
    for importances in importances_list:
        importances_mean = importances.mean().sort_values(ascending=False)

        means.append(importances_mean)
    importances_mean = pd.concat(means, axis=1).mean(axis=1).sort_values(ascending=False)
    importances_std = pd.concat(means, axis=1).std(axis=1).sort_values(ascending=False)
    axes[idx, 1].bar(importances_mean.index, importances_mean.values, color='skyblue')

    # rotate x_labels by 90 degrees
    axes[idx, 1].tick_params(axis='x', rotation=90)
    axes[idx, 1].errorbar(importances_mean.index, importances_mean.values, yerr=importances_std.values, fmt='o', color='black', capsize=5)
    axes[idx, 1].set_title(f'Feature Importances - {model_name}')
    axes[idx, 1].set_ylabel('Permutation Importance (roc_auc) as mean over 5 runs')
    axes[idx, 1].set_xlabel('Features')

    return axes