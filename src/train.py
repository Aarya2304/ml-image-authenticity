import torch
import torch.nn as nn
import torch.optim as optim

from dataset import dataset, dataloader
from model import CNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), "models/model.pth")
print("Model saved!")