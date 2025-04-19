import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.gru_models.gru_base import GRU
from models.gru_models.data_loader import DataframeLoader, RainDataset, get_balanced_sampler
from models.transformer_models.data_loader import train_dataloader, val_dataloader
from training.trainer_gru import train_model_dataloader
from models.gru_models.eval import evaluate_model
from config_custom.config_gru import CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_gru(use_existing_weights = True):
    # 1. Load and split data
    # x, y = DataframeLoader("train").split_df_into_sequences_with_labels()
    # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

    # labels = torch.tensor(y_train).squeeze()
    # num_rain = (labels == 1).sum().item()
    # num_no_rain = (labels == 0).sum().item()
    # total = len(labels)
    # print(f" Rain (1): {num_rain} samples")
    # print(f" No Rain (0): {num_no_rain} samples")
    # print(f" Positive ratio: {num_rain / total:.4f}")

    # # 2. Prepare datasets & loaders
    # train_ds = RainDataset(x_train, y_train, normalize=True)
    # sampler = get_balanced_sampler(y_train)
    # train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=sampler, num_workers=4, pin_memory=True)

    # val_ds = RainDataset(x_val, y_val, normalize=True, scaler=train_ds.scaler)
    # val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    train_dl = train_dataloader
    val_dl = val_dataloader

    # 3. Initialize model
    model = GRU(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['output_size'],
                dropout=CONFIG['dropout'], num_layers=CONFIG['num_layers']).to(device)
    print(" Model is on:", next(model.parameters()).device)

    # 4. Train model
    if not use_existing_weights:
        losses, accuracies = train_model_dataloader(
        model=model,
        train_dataloader=train_dl,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate']
        )


        print(" Training complete.")

        # 5. Save final weights
        torch.save(model.state_dict(), "ckpts/gru/model_weights_gru.pth")

    # 6. Load best model from early stopping (if available)
    best_weights_path = "ckpts/gru/model_weights_gru.pth"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(best_weights_path))
    else:
        model.load_state_dict(torch.load(best_weights_path, map_location=torch.device('cpu')))
    print(" Loaded best GRU model weights for final evaluation")

    # 7. Final evaluation on validation set
    evaluate_model(model, val_dl)

    # 8. Plot metrics
    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(losses, label="Loss", color='orange')
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Training Loss")
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(accuracies, label="Accuracy", color='green')
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.title("Training Accuracy")
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    run_gru()
