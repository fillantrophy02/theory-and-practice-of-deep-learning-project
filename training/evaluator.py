
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def evaluate_model(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    
    all_targets = []
    all_predictions = []
    running_loss = 0.0
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Reshape targets to match model output shape
            targets = targets.squeeze(1)
            
            # Forward pass - unpack the tuple
            _, _, outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs.float(), targets.float())
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Convert to probabilities and binary predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Store predictions and targets for metrics calculation
            all_targets.extend(targets.cpu().numpy().flatten())
            all_predictions.extend(preds.cpu().numpy().flatten())
    
    # Convert to numpy arrays and ensure they're binary
    all_targets = np.array(all_targets).astype(int)
    all_predictions = np.array(all_predictions).astype(int)
    
    # Double-check that we only have 0s and 1s
    assert np.all(np.isin(all_targets, [0, 1])), "Targets contain values other than 0 and 1"
    assert np.all(np.isin(all_predictions, [0, 1])), "Predictions contain values other than 0 and 1"
    
    # Calculate metrics
    avg_loss = running_loss / len(test_dataloader.dataset)
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='binary', zero_division=0)
    cm = confusion_matrix(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_predictions)
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }