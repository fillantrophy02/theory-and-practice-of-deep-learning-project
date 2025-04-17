import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from data.data_processing import DataProcessingPipeline
from models.seq2seq_gru import Seq2SeqGRU
from training.trainer_seq2seq_gru import train_seq2seq
from config.config import CONFIG
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv('data/processed-data/train.csv')

    # Auto-generate city code mappings from the dataset itself
    city_names = sorted(df['Location'].unique())
    city_code_mappings = {name: i for i, name in enumerate(city_names)}

    processor = DataProcessingPipeline(df, city_code_mappings=city_code_mappings)
    processor.clean()
    X, Y = processor.prepare_tensor_data_seq2seq(CONFIG['sequence_length'], CONFIG['future_steps'])
    positive = (Y == 1).sum().item()
    negative = (Y == 0).sum().item()
    pos_weight = torch.tensor([negative / positive])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    model = Seq2SeqGRU(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['output_size'],
                       CONFIG['num_layers'], CONFIG['dropout']).to(device)

    losses, accuracies = train_seq2seq(model, dataloader, CONFIG['num_epochs'], CONFIG['learning_rate'], device, pos_weight)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            logits = model(X_batch, Y_batch.size(1)).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(Y_batch.cpu().numpy().flatten())

    # ✅ Print detailed metrics
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["No Rain", "Rain"]))

    # ✅ Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Rain", "Rain"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return losses, accuracies

if __name__ == '__main__':
    losses, accuracies = main()
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, marker='o')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()