import sys
import torch
import torchmetrics
from transformers import PatchTSTConfig, PatchTSTForClassification, EarlyStoppingCallback, Trainer, TrainingArguments, TrainerCallback
from components.data_loader import train_dataloader, test_dataloader, input_size, train_ds, test_ds
from config import device, num_epochs, batch_size, no_of_days
import numpy as np
from sklearn.metrics import RocCurveDisplay, confusion_matrix, f1_score, roc_auc_score
import evaluate
from torch.utils.data import Dataset, DataLoader

def train_model(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    auc_metric = torchmetrics.AUROC('multiclass', num_classes=2).to(device)
    cm_metric = torchmetrics.ConfusionMatrix('multiclass', num_classes=2).to(device)
    f1_score_metric = torchmetrics.F1Score('multiclass', num_classes=2).to(device)

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()

            inputs_batch, outputs_batch = batch
            inputs_re = inputs_batch.to(device)
            outputs_re = outputs_batch.to(device)
            
            # Forward pass
            results = model(past_values=inputs_re, target_values=outputs_re)
            pred = results.prediction_logits
            loss = results.loss
            
            # Compute metrics
            auc_metric(pred, outputs_re) 
            cm_metric(pred, outputs_re) 
            f1_score_metric(pred, outputs_re) 
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        auc = auc_metric.compute()
        cm = cm_metric.compute()
        f1_score = f1_score_metric.compute()

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Training AUC: {auc.item():.4f}, Training F1-Score: {f1_score.item():.4f}')

        auc_metric.reset()
        cm_metric.reset()
        f1_score_metric.reset()

    torch.save(model.state_dict(), "ckpts/model.pth")

if __name__ == '__main__':  
    config = PatchTSTConfig(
        num_input_channels=input_size,
        num_targets=2, # number of classes
        context_length=no_of_days,
        patch_length=2,
    )
    model = PatchTSTForClassification(config).to(device)
    train_model(model)