import sys
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, confusion_matrix, f1_score, roc_auc_score
import torch
import evaluate


def calculate_cm_metrics(cm):
    TN, FP = cm[0]
    FN, TP = cm[1]
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return sensitivity, specificity, precision, accuracy

def calculate_tpr_fpr(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    return tpr, fpr

def plot_and_save_auc_curve(filepath, y_true, y_pred):
    display = RocCurveDisplay.from_predictions(y_true, y_pred)
    display.plot()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved ROC curve to {filepath}.")

def compute_metrics(eval_preds):
    print("COMPUTINGGGG")
    roc_auc = evaluate.load("roc_auc")
    f1 = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)  # Convert logits to class predictions

    # Compute ROC AUC (only valid for multi-class problems with "ovo" setting)
    try:
        auc_score = roc_auc.compute(predictions=predictions, references=labels, multi_class="ovo")["roc_auc"]
    except ValueError:
        auc_score = None  # Handle potential errors with single-class inputs

    # Compute F1 Score
    f1_score = f1.compute(predictions=predictions, references=labels, average="macro")["f1"]

    # Compute Confusion Matrix (manually since `evaluate` doesn't provide one)
    cm = confusion_matrix(labels, predictions).tolist()

    # Print for debugging
    sys.stdout.flush()
    print(f"\n🔹 AUC: {auc_score}")
    print(f"🔹 F1-Score: {f1_score}")
    print(f"🔹 Confusion Matrix: {cm}\n")

    return {"AUC": auc_score, "F1": f1_score, "Confusion Matrix": cm}

def report(auc, cm, f1_score):
    sensitivity, specificity, precision, accuracy = calculate_cm_metrics(cm)
    print("| Model   | Sensitivity | Specificity |  AUC  | Accuracy | Precision | F1 Score |")
    print("|---------|-------------|-------------|-------|----------|-----------|----------|")
    print(f"| LSTM    |    {sensitivity:.2f}     |    {specificity:.2f}     | {auc:.2f}  |   {accuracy:.2f}   |   {precision:.2f}    |   {f1_score:.2f}   |")