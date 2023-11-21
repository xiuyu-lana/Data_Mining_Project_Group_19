from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.models import vit_b_16, ViT_B_16_Weights

###
print("Downloading Dataset")
train_data = FashionMNIST(root='./Data', download=True, train=True, transform=transforms.Compose(
    [transforms.Lambda(lambda x: x.convert('RGB')), ViT_B_16_Weights.IMAGENET1K_V1.transforms(antialias=True)]))
test_data = FashionMNIST(root='./Data', download=True, train=False, transform=transforms.Compose(
    [transforms.Lambda(lambda x: x.convert('RGB')), ViT_B_16_Weights.IMAGENET1K_V1.transforms(antialias=True)]))
print('- Size of train data: ', train_data.shape)
print('- Size of test data: ', test_data.shape)

data = {'train': train_data,
        'test': test_data}
###

###
vit_save_path = './Data/Models/vit_b_16-c867db91.pth'
vit_download_link = 'https://download.pytorch.org/models/vit_b_16-c867db91.pth'
print("\n\nDownloading Vit Weights to ", vit_save_path, ' From ', vit_download_link)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nInitializing Vit_b_16 Feature Extractor on {device}")
model = vit_b_16()
model.load_state_dict(torch.load(vit_save_path))
model.heads = torch.nn.Flatten()

model.eval()
model.to(device)
print(model)
###

###
print('\n\nExtracting Features (sample)')
batch_size = 32
num_workers = 1
print('- batch_size: ', batch_size)

features = {}
for key, value in data.items():
    print(f'- {key} Features...')

    loader = DataLoader(value, batch_size, num_workers=num_workers)

    running_features, running_labels = torch.tensor([]), torch.tensor([])
    with torch.no_grad():
        for images, labels in tqdm(loader[:2]):
            images = images.to(device)
            running_features = torch.cat(
                [running_features, model(images).to('cpu')], dim=0)
            running_labels = torch.cat([running_labels, labels], dim=0)

    print('-- Shape of Features after extraction of 2 batches: ', running_features.shape)
###

###
save_path = './Data/Features/vit_b_16_features.pt'
print('\nDownloading features to ', save_path)
###