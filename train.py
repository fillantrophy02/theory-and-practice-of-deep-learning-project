import sys
import torch
import torchmetrics
from transformers import PatchTSTConfig, PatchTSTForClassification, EarlyStoppingCallback, Trainer, TrainingArguments, TrainerCallback
from components.data_loader import train_dataloader, test_dataloader, input_size, train_ds, test_ds
from components.experiment_recorder import log_model_artifacts, log_model_metric
from config import *
from test import test_model

def train_model(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = {
        "auc": torchmetrics.AUROC('multiclass', num_classes=2).to(device),
        "f1_score": torchmetrics.F1Score('multiclass', num_classes=2).to(device),
        "accuracy": torchmetrics.Accuracy('multiclass', num_classes=2).to(device)
    }

    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]', end = ' ')

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
            for name in metrics:
                metrics[name](pred, outputs_re)
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        log_model_metric("loss", loss, epoch)
        print(f'Loss: {loss:.3f}', end = '    ')

        for name in metrics:
            value = metrics[name].compute().item()

            log_model_metric(name, value, epoch)
            print(f'{name}: {value:.3f}', end='    ')

            metrics[name].reset()

if __name__ == '__main__':  
    config = PatchTSTConfig(
        num_input_channels=input_size,
        num_targets=2, # number of classes
        context_length=no_of_days,
        patch_length=patch_length,
    )
    model = PatchTSTForClassification(config).to(device)
    train_model(model)
    log_model_artifacts(model)
    torch.save(model.state_dict(), "ckpts/model.pth")
    test_model(model)