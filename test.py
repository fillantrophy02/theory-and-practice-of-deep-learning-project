import numpy as np
import torch
import torchmetrics
from components.data_loader import train_dataloader, test_dataloader, input_size
from components.metrics import calculate_cm_metrics, plot_and_save_auc_curve, report
from config import device, num_epochs, no_of_days
from transformers import PatchTSTConfig, PatchTSTForClassification, EarlyStoppingCallback, Trainer, TrainingArguments, TrainerCallback

def test_model(model):
    model.eval()

    auc_metric = torchmetrics.AUROC('multiclass', num_classes=2).to(device)
    cm_metric = torchmetrics.ConfusionMatrix('multiclass', num_classes=2).to(device)
    f1_score_metric = torchmetrics.F1Score('multiclass', num_classes=2).to(device)

    with torch.no_grad():
        labels, preds, = [], []

        for batch in test_dataloader:
            inputs_batch, outputs_batch = batch
            inputs_re = inputs_batch.to(device)
            outputs_re = outputs_batch.to(device)

            results = model(past_values=inputs_re, target_values=outputs_re)
            pred = results.prediction_logits
            loss = results.loss

            labels.extend(outputs_re.cpu().numpy())
            preds.extend(np.argmax(pred.cpu().numpy(), axis=-1))
            
            # Compute metrics
            auc_metric(pred, outputs_re) 
            cm_metric(pred, outputs_re) 
            f1_score_metric(pred, outputs_re) 

        auc = auc_metric.compute()
        cm = cm_metric.compute()
        f1_score = f1_score_metric.compute()
        report(auc, cm, f1_score)

        plot_and_save_auc_curve("visualizations/roc.png", np.array(labels), np.array(preds))

if __name__ == '__main__':
    config = PatchTSTConfig(
            num_input_channels=input_size,
            num_targets=2, # number of classes
            context_length=no_of_days,
            patch_length=2,
        )
    model = PatchTSTForClassification(config).to(device)
    model.load_state_dict(torch.load("ckpts/model.pth"))
    test_model(model)