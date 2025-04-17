import torch
import torch.nn as nn

def train_seq2seq(model, dataloader, num_epochs, learning_rate, device, pos_weight):
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, Y_batch in dataloader:
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
        epoch_losses.append(total_loss / len(dataloader))
        epoch_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_losses[-1]:.4f}, Accuracy: {accuracy:.4f}")

    return epoch_losses, epoch_accuracies
