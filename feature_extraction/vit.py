from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.models import ViT_B_16_Weights, vit_b_16


print("")
print("*"*25)
print("* Vit Feature Extractor *")
print("*"*25)

train_data = FashionMNIST(root='../Data', download=True, train=True, transform=transforms.Compose(
    [transforms.Lambda(lambda x: x.convert('RGB')), ViT_B_16_Weights.IMAGENET1K_V1.transforms(antialias=True)]))
test_data = FashionMNIST(root='../Data', download=True, train=False, transform=transforms.Compose(
    [transforms.Lambda(lambda x: x.convert('RGB')), ViT_B_16_Weights.IMAGENET1K_V1.transforms(antialias=True)]))

data = {'train': train_data,
        'test': test_data}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = vit_b_16()
# Weights from https://download.pytorch.org/models/vit_b_16-c867db91.pth
model.load_state_dict(torch.load('../Data/Models/vit_b_16-c867db91.pth'))
model.heads = torch.nn.Flatten()

model.eval()
model.to(device)
print('\nVit_b_16 Model:')
print(model)


# feature size should be nx768

save_path = '../Data/Features/vit_b_16_features.pt'
batch_size = 256
num_workers = 2

features = {}
for key, value in data.items():
    print(f'\nExtracting {key} Features...')

    loader = DataLoader(value, batch_size, num_workers=num_workers)

    running_features, running_labels = torch.tensor([]), torch.tensor([])
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            running_features = torch.cat(
                [running_features, model(images).to('cpu')], dim=0)
            running_labels = torch.cat([running_labels, labels], dim=0)

    features[key] = [running_features.numpy(), running_labels.numpy()]

torch.save(features, save_path)
print("\nFeature Dict Saved to, ", save_path)