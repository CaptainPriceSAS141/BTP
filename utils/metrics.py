from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

def calculate_metrics(labels, preds):

    precision = precision_score(
        labels,
        preds,
        average='weighted'
    )

    recall = recall_score(
        labels,
        preds,
        average='weighted'
    )

    f1 = f1_score(
        labels,
        preds,
        average='weighted'
    )

    return precision, recall, f1