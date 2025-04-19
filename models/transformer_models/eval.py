import numpy as np
import torch
import torchmetrics
from models.transformer_models.data_loader import val_dataloader
from models.transformer_models.metrics import calculate_cm_metrics, plot_and_save_auc_curve, report
from config_custom.config_transformer import *

from models.transformer_models.model import TransformerForClassification

def evaluate_model(model, epoch=num_epochs-1):
    model.eval()

    metrics = {
        "val_auc": torchmetrics.AUROC('binary').to(device),
        "val_f1_score": torchmetrics.F1Score('binary').to(device),
        "val_accuracy": torchmetrics.Accuracy('binary').to(device)
    }

    with torch.no_grad():
        labels, preds, = [], []
        all_losses = []
        print("\nVal", end = '    ')

        for batch in val_dataloader:
            inputs_batch, outputs_batch = batch
            inputs_re = inputs_batch.to(device)
            outputs_re = outputs_batch.to(device)

            # Forward pass
            pred = model(inputs_re)
            loss = model.loss(pred.float(), outputs_re.float())

            labels.extend(outputs_re.cpu().numpy())
            preds.extend(pred)
            
            # Compute metrics
            for name in metrics:
                metrics[name](pred, outputs_re)

            all_losses.append(loss.item())
    
        avg_loss = sum(all_losses) / len(all_losses)
        # log_model_metric("val_loss", avg_loss, epoch)
        print(f'Loss: {avg_loss:.3f}', end = '    ')

        for name in metrics:
            value = metrics[name].compute().item()

            # log_model_metric(name, value, epoch)
            print(f'{name}: {value:.3f}', end='    ')

            metrics[name].reset()

        # plot_and_save_auc_curve("visualizations/roc.png", np.array(labels), np.array(preds))

if __name__ == '__main__':
    model = TransformerForClassification.to(device)
    model.load_state_dict(torch.load("ckpts/transformer/model.pth"))
    evaluate_model(model)