from data import train_loader, test_loader, vocab
from model import RNNMODEL
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import json
import torchtext
# torchtext.disable_torchtext_deprecation_warning()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_evaluate(model, train_loader, test_loader, num_epochs=10, learning_rate=0.01):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    model.eval()
    all_greds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            all_greds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_greds)
    f1 = f1_score(all_labels, all_greds, average='macro')
    return acc, f1

#Thử nghiệm Pretrained và Scratch
results = {}
for pretrained in [True, False]:
    model = RNNMODEL(vocab_size=len(vocab), embedding_dim=100, hidden_dim=128, output_dim=3, pretrained=pretrained)
    key = f'RNN_Pretrained={pretrained}'
    acc, f1 = train_and_evaluate(model, train_loader, test_loader)
    results[key] = {'Accuracy': acc, 'F1-Score': f1}
    print(f"{key} - Accuracy: {acc}, F1-Score: {f1}")

# Lưu kết quả vào file JSON
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)