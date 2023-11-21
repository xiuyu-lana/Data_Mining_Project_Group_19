from tqdm import tqdm
import numpy as np
# from skimage.feature import hog

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST


print("")
print("*"*25)
print("* HOG Feature Extractor *")
print("*"*25)

train_data = FashionMNIST(root='../Data', download=True, train=True, transform=transforms.ToTensor())
test_data = FashionMNIST(root='../Data', download=True, train=False, transform=transforms.ToTensor())

data = {'train': train_data,
        'test': test_data}


save_path = '../Data/Features/hog_features.pt'
batch_size = 1

features = {}
for key, value in data.items():
    print(f'\nExtracting {key} Features...')

    loader = DataLoader(value, batch_size, shuffle=False, num_workers=3)

    running_features, running_labels = [], []
    for image, label in tqdm(loader):
        # Compute HOG features for each image
        image = np.squeeze(image.numpy())
        fd= hog(image, orientations=16, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        running_features.append(fd)
        running_labels.append(label.item())

    features[key] = [np.array(running_features), np.array(running_labels)]

torch.save(features, save_path)
print("\nFeature Dict Saved to, ", save_path)