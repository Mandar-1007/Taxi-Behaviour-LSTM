import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import TaxiDriverClassifier
from extract_feature import load_data, preprocess_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TaxiDriverDataset(Dataset):
    def __init__(self, X, y, device):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss, correct = 0, 0

    for X_batch, y_batch in tqdm(train_loader, desc="Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == y_batch).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = correct / len(train_loader.dataset)
    return train_loss, train_acc

def train_model():
    print("Loading training data...")
    X_train, y_train = load_data("/content/drive/MyDrive/03.BigDataAnalytics/7.Project2/data_5drivers/data_5drivers/*.csv")

    train_dataset = TaxiDriverDataset(X_train, y_train, device)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    output_dim = len(set(y_train))

    model = TaxiDriverClassifier(input_dim=X_train.shape[2], output_dim=output_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)  # Adjusted LR and weight decay

    # Learning Rate Scheduler: Reduce LR by 50% after 15 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_acc = 0
    patience = 7
    counter = 0

    for epoch in range(30):
        train_loss, train_acc = train(model, optimizer, criterion, train_loader, device)
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")

        scheduler.step()  # Adjust learning rate

        if train_acc > best_acc:
            best_acc = train_acc
            counter = 0
            torch.save(model.state_dict(), "taxi_model.pth")
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("Training complete. Best Accuracy:", best_acc)
