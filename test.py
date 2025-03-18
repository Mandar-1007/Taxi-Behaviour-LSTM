import os
import torch
from torch.utils.data import DataLoader, Dataset
from model import TaxiDriverClassifier
from extract_feature import load_data
from train import TaxiDriverDataset

def evaluate(model, criterion, test_loader, device):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == y_batch).sum().item()

    test_loss = total_loss / len(test_loader)
    test_acc = correct / len(test_loader.dataset)
    return test_loss, test_acc

def test_model(test_dir):
    device = torch.device("cpu")

    test_file_pattern = os.path.join(test_dir, "*.csv")
    X_test, y_test = load_data(test_file_pattern)

    test_dataset = TaxiDriverDataset(X_test, y_test, device)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = TaxiDriverClassifier(input_dim=X_test.shape[2], output_dim=5).to(device)
    model.load_state_dict(torch.load("taxi_model.pth", map_location=device))
    model.eval()

    test_loss, test_accu = evaluate(model, torch.nn.CrossEntropyLoss(), test_loader, device)

    print(f"Accuracy={test_accu:.4f}")
