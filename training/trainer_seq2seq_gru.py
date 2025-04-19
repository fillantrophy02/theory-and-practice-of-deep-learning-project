import torch
import torch.nn as nn
import os

def train_seq2seq(model, train_dataloader, num_epochs, learning_rate, device, pos_weight, teacher_forcing_ratio=0.5):
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses = []
    epoch_accuracies = []

    save_path = "ckpts/gru/model_weights_seq2seq.pth"
    os.makedirs("ckpts/gru", exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, Y_batch in train_dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()

            # 🔁 Forward with autoregressive decoding
            outputs = model(X_batch, Y_batch, teacher_forcing_ratio=teacher_forcing_ratio).squeeze()

            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            correct += (preds == Y_batch).sum().item()
            total += Y_batch.numel()

            loss.backward()
            optimizer.step()

        accuracy = correct / total
        epoch_losses.append(total_loss / len(train_dataloader))
        epoch_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_losses[-1]:.4f}, Accuracy: {accuracy:.4f}")

        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), save_path)
            print("✅ Final Seq2Seq model saved.")

    return epoch_losses, epoch_accuracies
