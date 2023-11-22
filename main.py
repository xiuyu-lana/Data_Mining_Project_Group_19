import numpy as np
from tqdm import tqdm
from urllib import request
import gdown

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.models import vit_b_16, ViT_B_16_Weights

print('**************')
print('*  Group 19  *')
print('**************')

###
print('\n'+'-'*70)
print("Downloading Dataset")
train_data = FashionMNIST(root='./Data', download=True, train=True, transform=transforms.Compose(
    [transforms.Lambda(lambda x: x.convert('RGB')), ViT_B_16_Weights.IMAGENET1K_V1.transforms(antialias=True)]))
test_data = FashionMNIST(root='./Data', download=True, train=False, transform=transforms.Compose(
    [transforms.Lambda(lambda x: x.convert('RGB')), ViT_B_16_Weights.IMAGENET1K_V1.transforms(antialias=True)]))
print('- Size of train data: ', train_data.data.shape)
print('- Size of test data: ', test_data.data.shape)

data = {'train': train_data,
        'test': test_data}
###

###
print('\n'+'-'*70)
vit_save_path = './Data/Models/vit_b_16-c867db91.pth'
vit_download_link = 'https://download.pytorch.org/models/vit_b_16-c867db91.pth'
print("Downloading Vit Weights to ", vit_save_path)
print('From ', vit_download_link)
request.urlretrieve(vit_download_link, filename=vit_save_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nInitializing Vit_b_16 Feature Extractor on {device}")
model = vit_b_16()
model.load_state_dict(torch.load(vit_save_path))
model.heads = torch.nn.Flatten()

model.eval()
model.to(device)
###

###
print('\n'+'-'*70)
print('Extracting Features (SAMPLE)')
batch_size = 32
num_workers = 1
print('- batch_size: ', batch_size)

sample_batches = 10
features = {}
for key, value in data.items():
    print(f'\n- {key} Features...')

    loader = DataLoader(value, batch_size, num_workers=num_workers)

    running_features, running_labels = torch.tensor([]), torch.tensor([])
    with torch.no_grad():
        for batch_num, (images, labels) in enumerate(tqdm(loader)):
            images = images.to(device)
            running_features = torch.cat(
                [running_features, model(images).to('cpu')], dim=0)
            running_labels = torch.cat([running_labels, labels], dim=0)
            
            if batch_num == sample_batches-1:
                break

    print(f'-- Shape of {key} Features after extraction of {sample_batches} batches: ', running_features.shape)
###

###
print('\n'+'-'*70)
vit_features_save_path = './Data/Features/vit_b_16_features.pt'
vit_features_download_link = 'https://drive.google.com/uc?id=1sIohZ2lmFQHAOCtJLGYQ7EKra5qImLEM'
print('Downloading Features to', vit_features_save_path)
print('From ', vit_features_download_link)
# gdown.download(vit_features_download_link, vit_save_path, quiet=True)

print('\nLoading Downloaded Features')
data = torch.load(vit_features_save_path)
for key, value in data.items():
    print(f'{key.capitalize()} Feature (X, y): ', value[0].shape, value[1].shape)
###


###
print('\n'+'-'*70)
print('Performing Train Val Split\n')
X_train, X_val, y_train, y_val = train_test_split(data['train'][0], data['train'][1], test_size = 5000, stratify = data['train'][1], random_state = 10)
data['train'] = [X_train, y_train] 
data['val'] = [X_val, y_val]

for key, value in data.items():
    print(f'{key.capitalize()} Data (X, y): ', value[0].shape, value[1].shape)
###


###
print('\n'+'-'*70)
print("Doing Full PCA\n")
pca = PCA()
scalar = StandardScaler().fit(data['train'][0])
pca.fit(scalar.transform(data['train'][0]))
explained_variance_ratios = np.cumsum(pca.explained_variance_ratio_)

data['train'][0] = pca.transform(scalar.transform(data['train'][0]))
data['val'][0] = pca.transform(scalar.transform(data['val'][0]))
data['test'][0] = pca.transform(scalar.transform(data['test'][0]))

for key, value in data.items():
    print(f'{key.capitalize()} Data (X, y): ', value[0].shape, value[1].shape)
###


###
print('\n'+'-'*70)
print("Normalizing Data\n")
scalar = StandardScaler().fit(*data['train'])

data['train'][0] = scalar.transform(data['train'][0])
data['val'][0] = scalar.transform(data['val'][0])
data['test'][0] = scalar.transform(data['test'][0])

for key, value in data.items():
    print(f'{key.capitalize()} Data (X, y): ', value[0].shape, value[1].shape)
###

###
print('\n'+'-'*70)
print('Reordering Features based on ANOVA Score\n')
scores, _ = f_classif(*data['train'])

feature_score = list(zip(range(0, data['train'][0].shape[1]), scores))
feature_score = sorted(feature_score, key=lambda item: item[1], reverse=True)

feature_ordering = [item[0] for item in feature_score]

data['train'][0] = data['train'][0][:, feature_ordering]
data['val'][0] = data['val'][0][:, feature_ordering]
data['test'][0] = data['test'][0][:, feature_ordering]

for key, value in data.items():
    print(f'{key.capitalize()} Data (X, y): ', value[0].shape, value[1].shape)
###


###
print('\n'+'-'*70)
print('Creating a Subset of 278 principal Components\n')
n_component = 278

data['train'][0] = data['train'][0][:, :n_component]
data['val'][0] = data['val'][0][:, :n_component]
data['test'][0] = data['test'][0][:, :n_component]

for key, value in data.items():
    print(f'{key.capitalize()} Data (X, y): ', value[0].shape, value[1].shape)
###


###
print('\n'+'-'*70)
print('Training SVM (SAMPLE)')
print('- Training for 2 iterations')
clf = SVC(C= 25, break_ties= False, cache_size= 200, class_weight= None, coef0= 0.0, decision_function_shape= 'ovr', degree= 3, gamma= 'scale', kernel= 'rbf', max_iter= 2, probability= True, random_state= None, shrinking= True, tol= 0.001, verbose= False)
clf.fit(*data['train'])
train_preds = clf.predict(data['train'][0])
val_preds = clf.predict(data['val'][0])
print('\n- Train f1 Score of SVM: ', f1_score(data['train'][1], train_preds, average='macro'))
print('- Val f1 Score of SVM: ', f1_score(data['val'][1], val_preds, average='macro'))
###


###
print('\n'+'-'*70)
svm_download_link = 'https://drive.google.com/uc?id=14GSFYMX0_PqGR2h_IR9v0C-06-_z3DSO'
svm_save_path = './Data/Models/SVM_best.pt' 
print('Downloading Trained SVM to', vit_save_path)
print('From ', svm_download_link)
gdown.download(svm_download_link, svm_save_path, quiet=True)

print('\nLoading Trained SVM')
svm_clf = torch.load(svm_save_path)
print(f'- Hyperparameters of Trained SVM: ', svm_clf.get_params())

# print('\n- Training Report')
# print(classification_report(data['train'][1], svm_clf.predict(data['train'][0])))

print('\nValidation Report')
print(classification_report(data['val'][1], svm_clf.predict(data['val'][0])))

print('\nTesting Report')
print(classification_report(data['test'][1], svm_clf.predict(data['test'][0])))
###

print('\n'+'-'*70)
print('\n'+'-'*70)
print("THE END")