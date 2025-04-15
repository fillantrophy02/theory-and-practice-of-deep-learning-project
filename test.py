import numpy as np
import torch
import torchmetrics
from components.data_loader import train_dataloader, test_dataloader
from components.experiment_recorder import log_model_metric
from components.metrics import calculate_cm_metrics, plot_and_save_auc_curve, report
from config import *
from transformers import PatchTSTConfig, PatchTSTForClassification, EarlyStoppingCallback, Trainer, TrainingArguments, TrainerCallback

from models.transformer import TransformerForClassification

def test_model(model, epoch=num_epochs-1):
    model.eval()

    metrics = {
        "test_auc": torchmetrics.AUROC('binary').to(device),
        "test_f1_score": torchmetrics.F1Score('binary').to(device),
        "test_accuracy": torchmetrics.Accuracy('binary').to(device)
    }

    with torch.no_grad():
        labels, preds, = [], []
        print("\nTest", end = '    ')

        for batch in test_dataloader:
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
    
        log_model_metric("test_loss", loss, epoch)
        print(f'Loss: {loss.item():.3f}', end = '    ')

        for name in metrics:
            value = metrics[name].compute().item()

            log_model_metric(name, value, epoch)
            print(f'{name}: {value:.3f}', end='    ')

            metrics[name].reset()

        # plot_and_save_auc_curve("visualizations/roc.png", np.array(labels), np.array(preds))

if __name__ == '__main__':
    model = TransformerForClassification.to(device)
    model.load_state_dict(torch.load("ckpts/model.pth"))
    test_model(model)