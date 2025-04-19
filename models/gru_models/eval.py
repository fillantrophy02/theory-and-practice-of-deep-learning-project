import torch
import numpy as np
import pandas as pd
import os
from config_custom.config_gru import CONFIG
from models.gru_models.metrics import (
    evaluate_and_report, plot_roc_curve, plot_pr_curve, threshold_sweep
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, dataloader, epoch=0, is_seq2seq=False, target_seq_len=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            if is_seq2seq:
                # For Seq2SeqGRU
                target_seq_len = target_seq_len or Y_batch.size(1)
                logits = model(X_batch, target_seq_len).squeeze()
            else:
                # For base GRU
                hidden = torch.zeros(CONFIG['num_layers'], X_batch.size(0), model.hidden_size).to(device)
                _, outputs = model(X_batch, hidden)
                logits = outputs[:, -1]  # Only use final time step

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(Y_batch.cpu().numpy().flatten())

    # Save predictions to CSV
    os.makedirs("ckpts/gru", exist_ok=True)
    df = pd.DataFrame({
        "y_true": all_labels,
        "y_prob": all_probs,
        "y_pred": all_preds
    })
    df.to_csv("ckpts/gru/latest_preds.csv", index=False)
    print(" Saved predictions to ckpts/gru/latest_preds.csv")

    # ✅ Evaluate metrics
    metrics = evaluate_and_report(all_labels, all_probs, model_name="Seq2SeqGRU" if is_seq2seq else "GRU")

    # ✅ Save visualizations
    plot_roc_curve(all_labels, all_probs, save_path="ckpts/gru/roc_latest.png")
    plot_pr_curve(all_labels, all_probs, save_path="ckpts/gru/pr_latest.png")
    threshold_sweep(all_labels, all_probs)

    return all_preds, metrics
