import torch
import torch.nn as nn
import os


def train_seq2seq(model, train_dataloader, num_epochs, learning_rate, device, pos_weight):
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

            logits = model(X_batch, Y_batch.size(1)).squeeze()
            loss = criterion(logits, Y_batch)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == Y_batch).sum().item()
            total += Y_batch.numel()

            loss.backward()
            optimizer.step()

        accuracy = correct / total
        epoch_losses.append(total_loss / len(train_dataloader))
        epoch_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_losses[-1]:.4f}, Accuracy: {accuracy:.4f}")

        #  Save final model on last epoch
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), save_path)
            print(" Final Seq2Seq model saved.")

    return epoch_losses, epoch_accuracies
