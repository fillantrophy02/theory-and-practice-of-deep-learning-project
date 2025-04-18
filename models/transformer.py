import sys
import torch
import torchmetrics
from transformers import PatchTSTConfig, PatchTSTForClassification, EarlyStoppingCallback, Trainer, TrainingArguments, TrainerCallback
from models.transformer_models.data_loader import train_dataloader
from models.transformer_models.experiment_recorder import log_model_artifacts, log_model_metric
from config_custom.config_transformer import *
from models.transformer_models.model import TransformerForClassification
from models.transformer_models.eval import evaluate_model

def run_transformer(model):
    model = TransformerForClassification().to(device)
    train_model(model)
    log_model_artifacts(model)
    torch.save(model.state_dict(), "ckpts/transformer/model.pth")
    # evaluate_model(model)

def train_model(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    metrics = {
        "auc": torchmetrics.AUROC('binary').to(device),
        "f1_score": torchmetrics.F1Score('binary').to(device),
        "accuracy": torchmetrics.Accuracy('binary').to(device)
    }

    for epoch in range(num_epochs):
        model.train()
        print(f'\nEpoch [{epoch+1}/{num_epochs}]', end = ' ')

        all_losses = []

        for batch in train_dataloader:
            optimizer.zero_grad()

            inputs_batch, outputs_batch = batch
            inputs_re = inputs_batch.to(device)
            outputs_re = outputs_batch.to(device)
            
            # Forward pass
            pred = model(inputs_re) # (batch_size, target_seq_length, 1)
            loss = model.loss(pred.float(), outputs_re.float())

            # Compute metrics
            for name in metrics:
                for step in range(target_seq_length):
                    metrics[name](pred[:, step, :], outputs_re[:, step, :])
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            all_losses.append(loss.item())

        avg_loss = sum(all_losses) / len(all_losses)
        print(f'Loss: {avg_loss:.3f}', end = '    ')
        log_model_metric("loss", avg_loss, epoch)

        for name in metrics:
            value = metrics[name].compute().item()

            log_model_metric(name, value, epoch)
            print(f'{name}: {value:.3f}', end='    ')

            metrics[name].reset()

        evaluate_model(model, epoch=epoch)

if __name__ == '__main__':
    run_transformer()