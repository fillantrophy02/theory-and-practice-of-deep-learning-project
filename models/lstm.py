import torch
import os, sys

from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from data.data_loader import RainDataset 

from models.lstm_model import LSTM
from models.lstm_dual_model import DualMemoryCellLSTM
from models.lstm_modulation import ModulationGateLSTM

from training.trainer import train_model
from training.evaluator import evaluate_model

from config_custom.config_lstm import CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_lstm(use_existing_weights = True):
    
    # Step 1: Initialize datasets using RainDataset
    train_ds = RainDataset(
        "train", 
        activate_undersampling=True, 
        scaler=None,
        seq_length=CONFIG["seq_length"], 
        target_seq_length=CONFIG["target_seq_length"]
    )
    
    test_ds = RainDataset(
        "test", 
        scaler=train_ds.scaler,  # Use the same scaler as training
        seq_length=CONFIG["seq_length"], 
        target_seq_length=CONFIG["target_seq_length"]
    )
    
    # Step 2: Create data loaders
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_ds, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False
    )
    
    # # Print dataset statistics
    # train_ds.report()
    # test_ds.report()
    
    # Step 3: Initialize model
    input_size = train_ds.x.shape[2]  # Number of features

    if CONFIG["model"] == 1:
        model = LSTM(
            input_size=input_size, 
            hidden_size=CONFIG["hidden_size"], 
            output_size=CONFIG["output_size"]
        ).to(device)


    elif CONFIG["model"] == 2:
        model = ModulationGateLSTM(
            input_size=train_ds.x.shape[2],  # Number of features
            hidden_size=CONFIG["hidden_size"],
            output_size=CONFIG["output_size"]
        ).to(device)
        
    elif CONFIG["model"] == 3:
        model = DualMemoryCellLSTM(
            input_size=train_ds.x.shape[2],  # Number of features
            hidden_size=CONFIG["hidden_size"],
            output_size=CONFIG["output_size"]
        ).to(device)
        
    train_model_flag = True

    model_type = CONFIG.get("model")
    model_weights_path = f"ckpts/lstm/model{model_type}_weights.pth"
    
    if use_existing_weights and os.path.exists(model_weights_path):
        train_model_flag = False
    
    if train_model_flag:
        train_model(model, train_dataloader, 
                    num_epochs=CONFIG["num_epochs"],
                    learning_rate=CONFIG["learning_rate"])
        
        torch.save(model.state_dict(), model_weights_path)
    else:
        print("Loading pre-trained weights...")
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    
    evaluate_model(model, test_dataloader)

if __name__ == "__main__":
    run_lstm()
