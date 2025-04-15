import sys
import torch
import torchmetrics
from transformers import PatchTSTConfig, PatchTSTForClassification, EarlyStoppingCallback, Trainer, TrainingArguments, TrainerCallback
from components.data_loader import train_dataloader
from components.experiment_recorder import log_model_artifacts, log_model_metric
from config import *
from models.transformer import TransformerForClassification
from test import test_model

def train_model(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = {
        "auc": torchmetrics.AUROC('binary').to(device),
        "f1_score": torchmetrics.F1Score('binary').to(device),
        "accuracy": torchmetrics.Accuracy('binary').to(device)
    }

    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]', end = ' ')

        for batch in train_dataloader:
            optimizer.zero_grad()

            inputs_batch, outputs_batch = batch
            inputs_re = inputs_batch.to(device)
            outputs_re = outputs_batch.to(device)
            
            # Forward pass
            pred = model(inputs_re)
            loss = model.loss(pred.float(), outputs_re.float())
            
            # Compute metrics
            for name in metrics:
                metrics[name](pred, outputs_re)
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        log_model_metric("loss", loss, epoch)
        print(f'Loss: {loss.item():.3f}', end = '    ')

        for name in metrics:
            value = metrics[name].compute().item()

            log_model_metric(name, value, epoch)
            print(f'{name}: {value:.3f}', end='    ')

            metrics[name].reset()

if __name__ == '__main__':
    model = TransformerForClassification().to(device)
    train_model(model)
    log_model_artifacts(model)
    torch.save(model.state_dict(), "ckpts/model.pth")
    test_model(model)