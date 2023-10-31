<h1 style="text-align:center;">ECEN 758 Project</h1>

## Dataset Description
- Name = Fashion MNIST
- Image Size = 1x28x28 (gray-scale)
- Num Classes = 10
- Num Training Samples = 55000
- Num Val Samples = 5000
- Num Testing Samples = 10000

## Dataset Transformation
### Feature Extraction
1. Num Flatten Features = 784
1. TODO: Handcrafted Features (HOG, SIFT, etc)
1. Num Resnet18 Features = 512
1. Num Vit Features = 768
### PCA
1. Num Vit Features = 250
### ANOVA Test
1. Num Vit Features (90 percentile) = 

## Method
### EDA
- Descriptive statistics based on features (after feature extraction and dimensionality reduction)
- Few plots about data distribution
- Few plots of image examples from each class
### Augmentation
(Can be attempted later if best model is Overfitting)
- Auto Augmentation
- Handpicked (Random Horizontal Flip, Random Vertical Flip, Random Rotation, Random Resized Crop)
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
- Least Squares
- else
## Explainability
- Grad-CAM
- LIME

## Comparision
- Write Submission Report
- Website/Blog
