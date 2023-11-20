<h1 style="text-align:center;">ECEN 758 Project</h1>

## Dataset Description
- Name = Fashion MNIST
- Image Size = 1x28x28 (gray-scale)
- Num Images = 70000
- Num Classes = 10
- Num Images per Class = 10000 (balanced dataset)

## Train Val Test Split
- Num Training Samples = 55000
- Num Validation Samples = 5000 (sklearn, random_state = 10, with stratification)
- Num Testing Samples = 10000

## EDA
- Plots about data distribution
- Image examples from each class

## Dataset Transformation
### 1. Feature Extraction
1. Num Flatten Features = 784
1. Num HOG Features = 256
1. Num Resnet18 Features = 512
1. Num Vit Features = 768
### 2. Normalization
### 3. Full PCA
### 4. Components Sorted by ANOVA Test Scores 
### 5. Feature Subsets (0.85 evr to 0.98 evr with a delta of 0.01 evr)
1. Num Flatten Components = [88, 97, 108, 119, 133, 149, 164, 184, 210, 242, 285, 328, 384]
1. Num HOG Components = [116, 121, 124, 128, 132, 137, 142, 146, 151, 156, 162, 177, 181]
1. Num Resnet18 Components = [132, 142, 155, 167, 184, 198, 208, 231, 251, 273, 298, 328, 378]
1. Num Vit Components = [114, 124, 136, 150, 168, 184, 198, 220, 244, 278, 317, 369, 425]

## Models (Reporting Validation f1-score with default hyperparameters)
- Gaussian Naive Bayes
    - Flat = 0.69
    - HOG = 0.73
    - ResNet = 0.78
    - Vit = 0.82
- Nearest Neighbor
    - Flat = 0.86
    - HOG = 0.70
    - ResNet = 0.85
    - Vit = 0.87
- Decision Trees
    - Flat = 0.77
    - HOG = 0.71
    - ResNet = 0.74
    - Vit = 0.75
- Random Forest
    - Flat = 0.87
    - HOG =  0.81
    - ResNet = 0.85
    - Vit = 0.86
- Logistic Regression
    - Flat = 0.85
    - HOG = 0.84
    - ResNet = 0.88
    - Vit = 0.90
- SVM
    - Flat = 0.89
    - HOG = 0.86
    - ResNet = 0.90
    - Vit = 0.92
- Feed Forward Network
    - Flat = 0.85
    - HOG = 0.83
    - ResNet = 0.88
    - Vit = 0.90

## Model Selection
Vit + subset with 278 components + SVM

## Hyperparameter Tuning of the SVM
- Regularization parameter C: {0.1, 1, 10, 25}
- Kernel: {'linear', 'poly', 'rbf', 'sigmoid'}
- Degree (when kernel is 'poly'): {2, 3, 4} 
  \
  \
During the hyperparameetr tuning, totally 24 models were trained (on the same 55000 samples) and validated (on the same 5000 samples). \
According to the F1 score, the best hyperparameter settings are: 

## Explainability
- LIME
  \
  pip install LIME \
  import lime  \
  import lime.lime_tabular \
  explainer = lime.lime_tabular.LimeTabularExplainer(training_data, feature_names=training_data.feature_names, class_names=true_class, verbose=True, mode='regression') \
  exp = explainer.explain_instance(data_instance, model.predict, num_features=5) \
  exp.show_in_notebook(show_table=True) \
  exp.as_list()

## Comparision
- Write Submission Report
- Website/Blog
