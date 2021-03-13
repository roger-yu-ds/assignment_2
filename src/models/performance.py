from sklearn.metrics import roc_auc_score
from typing import Dict
import pandas as pd


def convert_cr_to_dataframe(report_dict: Dict) -> pd.DataFrame:
    """
    Converts the dictionary format of the Classification Report (CR) to a
    dataframe for easy of sorting
    :param report_dict: The dictionary returned by
    sklearn.metrics.classification_report.
    :return: Returns a dataframe of the same information.
    """
    beer_style = list(report_dict.keys())
    beer_style.remove('accuracy')
    beer_style.remove('macro avg')
    beer_style.remove('weighted avg')

    precision = []
    recall = []
    f1 = []
    support = []

    for key, value in report_dict.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            precision.append(value['precision'])
            recall.append(value['recall'])
            f1.append(value['f1-score'])
            support.append(value['support'])

    result = pd.DataFrame({'beer_style': beer_style,
                           'precision': precision,
                           'recall': recall,
                           'f1': f1,
                           'support': support})

    return result


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    # From https://stackoverflow.com/a/52750599
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

    # marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    # using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
    roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict
