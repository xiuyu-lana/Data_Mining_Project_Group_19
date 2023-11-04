<h1 style="text-align:center;">ECEN 758 Project</h1>

## Dataset Description
- Name = Fashion MNIST
- Image Size = 1x28x28 (gray-scale)
- Num Images = 70000
- Num Classes = 10
- Num Images per Class = 10000 (balanced dataset)

## Data Augmentation (If time permits)
(Can be attempted later if best model is Overfitting)
- Auto Augmentation
- Handpicked (Random Horizontal Flip, Random Vertical Flip, Random Rotation, Random Resized Crop)

## Train Val Test Split
- Num Training Samples = 55000
- Num Validation Samples = 5000 (Use sklearn.model_selection.train_test_split(data['train'][0], data['train'][1], test_size = 5000, stratify = data['train'][1], random_state = 10))
- Num Testing Samples = 10000

## Dataset Transformation
### Feature Extraction
1. Num Flatten Features = 784
1. Num HOG Features =
1. Num Resnet18 Features = 512
1. Num Vit Features = 768
### PCA (Atleast 95% of explained variance ratio)
1. Num Flatten Features = 
1. Num HOG Features = 
1. Num Resnet18 Features = 
1. Num Vit Features = 250
### ANOVA Test (To be done later because we dont know what percentile/number of features to select)
1. Num Flatten Features = 
1. Num HOG Features = 
1. Num Resnet18 Features = 
1. Num Vit Features =

## EDA
- Descriptive statistics based on final features
- Few plots about data distribution
- Few plots of image examples from each class

## Models (Reporting Validation f1-score with default hyperparameters)
- Gaussian Naive Bayes
    - Flat
    - HOG
    - ResNet
    - Vit = 0.80
- Nearest Neighbor
    - Flat
    - HOG
    - ResNet
    - Vit = 0.87
- Decision Trees
    - Flat
    - HOG
    - ResNet
    - Vit = 0.76
- Random Forest
    - Flat
    - HOG
    - ResNet
    - Vit = 0.86
- Logistic Regression
    - Flat
    - HOG
    - ResNet
    - Vit = 0.90
- SVM
    - Flat
    - HOG
    - ResNet
    - Vit = 0.92
- Feed Forward Network
    - Flat
    - HOG
    - ResNet
    - Vit = 0.91
- Transformer (can try later if time permits)
    - Flat
    - HOG
    - ResNet
    - Vit = 

## Explainability
- Grad-CAM
- LIME

## Comparision
- Write Submission Report
- Website/Blog
