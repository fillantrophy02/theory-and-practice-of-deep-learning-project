import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model2(model, X_test, Y_test):
    model.eval()

    with torch.no_grad():
        X_test, Y_test = X_test.to(device), Y_test.to(device)
        
        _, _, predictions = model(X_test)
        
        probs = torch.sigmoid(predictions.squeeze(-1))
        predicted_labels = (probs >= 0.5).float()
        
        correct = (predicted_labels == Y_test.squeeze(-1)).sum().item()
        total = Y_test.numel() 
        accuracy = correct / total
        
        print(f"Test Accuracy: {accuracy:.4f}")