import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

data_path = "data/real_and_fake_face"

dataset = datasets.ImageFolder(root=data_path, transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Classes:", dataset.classes)