import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from data.data_processing import DataProcessingPipeline
from models.gru_classifier import GRU

from training.trainer_gru import train_model
from config.config import CONFIG

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load and preprocess data
    df = pd.read_csv('data/processed-data/train_pro.csv')
    data_processor = DataProcessingPipeline(df)
    X_train, Y_train = data_processor.prepare_tensor_data_classifier()

    # Step 2: Initialize model, criterion, and optimizer
    model = GRU(input_size=CONFIG['input_size'], hidden_size=CONFIG['hidden_size'], output_size=CONFIG['output_size']).to(device)

    # Step 3: Train the model
    losses, accuracies = train_model(model, X_train, Y_train, num_epochs=CONFIG['num_epochs'], batch_size=CONFIG['batch_size'])

    plt.figure(figsize=(10, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Loss", color='orange')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label="Accuracy", color='green')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Load and preprocess test set
    df_test = pd.read_csv('data/processed-data/test_pro.csv')
    test_processor = DataProcessingPipeline(df_test)
    X_test, Y_test = test_processor.prepare_tensor_data_classifier()
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        hidden = torch.zeros(X_test.size(0), model.hidden_size).to(device)
        _, logits = model(X_test, hidden)
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs >= 0.5).float()

        correct = (preds == Y_test).sum().item()
        accuracy = correct / Y_test.size(0)
        print(f"\n🧪 Test Accuracy: {accuracy:.4f}")

        print("\n📊 Classification Report on Test Set:")
        # Binarize Y_test to match prediction format
        y_true = (Y_test >= 0.5).float().cpu()
        y_pred = preds.cpu()

        print("\n📊 Classification Report on Test Set:")
        print(classification_report(y_true, y_pred, target_names=["No Rain", "Rain"]))

        # Confusion matrix
        print("\n🧩 Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()

