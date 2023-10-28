<h1 style="text-align:center;">ECEN 758 Project</h1>

## Dataset Description
- Name = Fashion MNIST
- Image Size = 1x28x28
- Num Classes = 10
- Num Training Samples = 55000
- Num Val Samples = 5000
- Num Testing Samples = 10000

## Feature Extraction
1. Num Flatten Features = 784
1. TODO: Handcrafted Features (HOG, SIFT, etc)
1. Num Resnet18 Features = 512
1. Num Vit_b_16 Features = 768

## Method
### EDA
- Few plots about data distribution
### Augmentation
(Can be attempted later if best model is Overfitting)
- Auto Augmentation
- Handpicked (Random Horizontal Flip, Random Vertical Flip, Random Rotation, Random Resized Crop)
### Dimensionality Reduction
- Chi Square Test
- Anova Test
- PCA
- SVD
### Models
- Naive Bayes
- Nearest Neighbor
- Decision Trees
- Linear Regression
- Logistic Regression
- SVM
- Dense Perceptron Layers
- Transformer
### Loss Functions
- Cross Entropy
- Is there a better loss function? Can we add auxilary losses maybe?

## Explainability
- Grad-CAM

## Implement SOTA
If we have time

## Comparision
- Write Submission Report
- Website/Blog