import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, X_test, Y_test):

    model.eval()

    with torch.no_grad():
        X_test, Y_test = X_test.to(device), Y_test.to(device)
        predictions = model(X_test)[2] # Output from LSTM model
        
        probs = torch.sigmoid(predictions.squeeze())
        predicted_labels = (probs >= 0.5).float()
        
        correct = (predicted_labels == Y_test).sum().item()
        total = Y_test.size(0)
        accuracy = correct / total
        
        print(f"Test Accuracy: {accuracy:.4f}")
