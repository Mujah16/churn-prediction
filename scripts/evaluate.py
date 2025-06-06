import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, dataloader):
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            outputs = model(x_batch)
            preds = (outputs > 0.5).int()
            predictions.extend(preds.view(-1).tolist())
            targets.extend(y_batch.view(-1).tolist())

    print("\nClassification Report:")
    print(classification_report(targets, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(targets, predictions))
