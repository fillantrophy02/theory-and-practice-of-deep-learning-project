import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    RocCurveDisplay, confusion_matrix, f1_score,
    roc_auc_score, precision_score, recall_score,
    accuracy_score, precision_recall_curve
)

def calculate_cm_metrics(cm):
    TN, FP = cm[0]
    FN, TP = cm[1]
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return sensitivity, specificity, precision, accuracy

def calculate_tpr_fpr(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    return tpr, fpr

def plot_roc_curve(y_true, y_prob, save_path="roc_curve.png"):
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.savefig(save_path)
    plt.close("all")
    print(f" Saved ROC curve to {save_path}")

def plot_pr_curve(y_true, y_prob, save_path="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close("all")
    print(f" Saved PR curve to {save_path}")

def evaluate_and_report(y_true, y_prob, model_name="Model", threshold=0.5):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sensitivity, specificity, precision, accuracy = calculate_cm_metrics(cm)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print("| Model   | Sensitivity | Specificity |  AUC  | Accuracy | Precision | F1 Score |")
    print("|---------|-------------|-------------|-------|----------|-----------|----------|")
    print(f"| {model_name:<7} |    {sensitivity:.2f}     |    {specificity:.2f}     | {auc:.2f}  |   {accuracy:.2f}   |   {precision:.2f}    |   {f1:.2f}   |")

    return {
        "F1": f1,
        "Precision": precision,
        "Recall": sensitivity,
        "Accuracy": accuracy,
        "AUC": auc,
        "ConfusionMatrix": cm.tolist()
    }

def threshold_sweep(y_true, y_prob, thresholds=np.arange(0.1, 0.91, 0.1)):
    print("\n🔍 Threshold Sweep Analysis")
    print("Threshold | F1    | Precision | Recall")
    print("----------|-------|-----------|--------")

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        print(f"   {thresh:.2f}   | {f1:.4f} |  {prec:.4f}  | {rec:.4f}")
