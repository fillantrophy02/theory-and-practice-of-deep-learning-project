import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from models.gru_models.seq2seq_gru import Seq2SeqGRU
from training.trainer_seq2seq_gru import train_seq2seq
from config_custom.config_gru import CONFIG
from models.gru_models.metrics import (
    evaluate_and_report, plot_roc_curve, plot_pr_curve, threshold_sweep
)
from models.gru_models.data_loader import DataframeLoader, RainDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataframeLoader("train")
    X, Y = loader.split_df_into_seq2seq(future_steps=CONFIG['future_steps'])

    # Split into train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    positive = (Y_train == 1).sum()
    negative = (Y_train == 0).sum()
    pos_weight = torch.tensor([negative / positive])

    print(f" Rain (1): {positive} samples")
    print(f" No Rain (0): {negative} samples")
    print(f" Positive ratio: {positive / (positive + negative):.4f}")

    train_ds = RainDataset(X_train, Y_train, normalize=True)
    val_ds   = RainDataset(X_val, Y_val, normalize=True, scaler=train_ds.scaler)

    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    model = Seq2SeqGRU(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['output_size'],
                       CONFIG['num_layers'], CONFIG['dropout']).to(device)

    losses, accuracies = train_seq2seq(
    model, train_dl, CONFIG['num_epochs'], CONFIG['learning_rate'], device, pos_weight
)


    model.load_state_dict(torch.load("ckpts/gru/model_weights_seq2seq.pth"))
    print(" Loaded best Seq2Seq model weights for final evaluation")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, Y_batch in val_dl:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            Y_batch = Y_batch.unsqueeze(-1)  # Make sure shape is [batch, future_steps, 1]
            logits = model(X_batch, target_seq=None, teacher_forcing_ratio=0.0).squeeze()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(Y_batch.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    #  Metrics analysis
    metrics = evaluate_and_report(all_labels, all_probs, model_name="Seq2SeqGRU")
    print(" Metrics Summary:")
    for k, v in metrics.items():
        if k != "ConfusionMatrix":
            print(f"{k}: {v:.4f}")
    print("Confusion Matrix:")
    print(np.array(metrics["ConfusionMatrix"]))

    # Plots
    plot_roc_curve(all_labels, all_probs, save_path="ckpts/gru/roc_curve_s2s.png")
    plot_pr_curve(all_labels, all_probs, save_path="ckpts/gru/pr_curve_s2s.png")
    threshold_sweep(all_labels, all_probs)

    return losses, accuracies


if __name__ == '__main__':
    losses, accuracies = main()
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, marker='o')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()
