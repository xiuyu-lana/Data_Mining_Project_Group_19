from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

print("")
print("*"*31)
print("* Flattened Feature Extractor *")
print("*"*31)

train_data = FashionMNIST(root='../Data', download=True, train=True, transform=transforms.ToTensor())
test_data = FashionMNIST(root='../Data', download=True, train=False, transform=transforms.ToTensor())

data = {'train': train_data,
        'test': test_data}


# feature size should be nx(28*28)
save_path = '../Data/Features/flattened_features.pt'
batch_size = 512

features = {}
for key, value in data.items():
    print(f'\nExtracting {key} Features...')

    loader = DataLoader(value, batch_size, shuffle=False, num_workers=3)

    running_features, running_labels = torch.tensor([]), torch.tensor([])
    for images, labels in tqdm(loader):
        running_features = torch.cat(
            [running_features, torch.flatten(images, start_dim=1)], dim=0)
        running_labels = torch.cat([running_labels, labels], dim=0)

    features[key] = [running_features.numpy(), running_labels.numpy()]

torch.save(features, save_path)
print("\nFeature Dict Saved to, ", save_path)