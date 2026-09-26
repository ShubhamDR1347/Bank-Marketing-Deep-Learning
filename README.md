# Bank Marketing Deep Learning

A Deep Learning project built with **PyTorch** using the **UCI Bank Marketing dataset**.

The main goal of this project was to learn and implement the complete Deep Learning workflow for a binary classification problem — predicting whether a bank customer subscribed to a term deposit.

This project was developed as part of my journey into **Deep Learning and PyTorch**, so the notebook also documents the learning process, experiments, and intermediate steps.

---

## Project Overview

The dataset contains information about customers contacted during a bank marketing campaign.

The target variable is:

* `no` → customer did not subscribe to a term deposit
* `yes` → customer subscribed to a term deposit

The dataset is highly imbalanced, with significantly more `no` than `yes` observations.

Because of this imbalance, the project does not rely on accuracy alone. Precision, Recall, F1-score, ROC-AUC, and the Confusion Matrix are also used for evaluation.

---

## Dataset

The project uses the **Bank Marketing dataset from the UCI Machine Learning Repository**.

The dataset used in this project contains:

* **45,211 observations**
* **16 input features**
* **1 target variable**

The features include customer information such as:

* Age
* Job
* Marital status
* Education
* Account balance
* Housing loan
* Personal loan
* Contact type
* Month
* Campaign information
* Previous campaign outcome
* Call duration

---

## Project Workflow

The project follows the following Deep Learning workflow:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Preprocessing
   ↓
Train / Validation / Test Split
   ↓
PyTorch Tensors
   ↓
DataLoader
   ↓
Neural Network
   ↓
Forward Propagation
   ↓
Loss Function
   ↓
Backpropagation
   ↓
Adam Optimizer
   ↓
Training
   ↓
Validation
   ↓
Dropout
   ↓
Early Stopping
   ↓
Threshold Optimization
   ↓
Final Test Evaluation
```

---

## Preprocessing

The dataset contains both numerical and categorical features.

### Numerical features

Numerical features were standardized using `StandardScaler`.

### Categorical features

Categorical features were converted into numerical form using `OneHotEncoder`.

The preprocessing pipeline produced **51 input features** for the neural network.

The preprocessing transformer was fitted only on the training data to avoid data leakage.

---

## Train / Validation / Test Split

The dataset was divided using a stratified split:

* **70% Training**
* **15% Validation**
* **15% Testing**

Stratification was used to maintain approximately the same class distribution across all three sets.

---

## Neural Network

The main model is a simple Multi-Layer Perceptron (MLP) built using PyTorch.

Architecture:

```text
Input: 51 features
       ↓
Linear(51 → 64)
       ↓
ReLU
       ↓
Dropout(0.30)
       ↓
Linear(64 → 32)
       ↓
ReLU
       ↓
Dropout(0.30)
       ↓
Linear(32 → 1)
       ↓
Output
```

The model contains **5,441 trainable parameters**.

---

## Loss Function

Because the target is binary, the project uses:

```python
BCEWithLogitsLoss
```

The dataset has a strong class imbalance, so a `pos_weight` was used to give more importance to the minority `yes` class.

The calculated positive-class weight was approximately:

```text
7.55
```

---

## Optimization

The neural network was trained using the **Adam optimizer** with a learning rate of:

```text
0.001
```

The training process included:

* Forward propagation
* Loss calculation
* Backpropagation
* Gradient calculation
* Parameter updates

---

## Dropout

Dropout was introduced as a regularization technique.

The model uses:

```text
Dropout = 0.30
```

Dropout randomly disables a proportion of activations during training, helping reduce reliance on specific neurons.

---

## Early Stopping

Early stopping was implemented using validation loss.

The model saved the best-performing parameters and stopped training when the validation loss failed to improve for several consecutive epochs.

This helps prevent unnecessary training and can reduce overfitting.

---

## Threshold Optimization

The model produces probabilities rather than directly producing `yes` or `no`.

Instead of automatically using a threshold of `0.5`, different thresholds were tested using the validation set.

The threshold producing the highest validation F1-score was:

```text
Best Validation Threshold = 0.76
```

This threshold was then fixed before evaluating the final test set.

---

## Final Test Results

The final model was evaluated on the previously unseen test set.

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **88.43%** |
| Precision | **50.33%** |
| Recall    | **77.30%** |
| F1-score  | **60.96%** |
| ROC-AUC   | **92.32%** |

### Final Confusion Matrix

```text
                 Predicted
                 No      Yes

Actual No       5384     605
Actual Yes       180     613
```

The results show that the model was able to identify a substantial proportion of the positive class, while also producing a number of false positive predictions.

This is why looking beyond accuracy was important for this project.

---

## What I Learned

This project helped me understand and practically implement several Deep Learning concepts:

* Neural network architecture
* Neurons, weights and biases
* Forward propagation
* Activation functions
* Binary classification
* Loss functions
* Class imbalance
* Backpropagation
* Gradients
* Adam optimization
* Batch training
* Epochs
* Validation
* Dropout
* Early stopping
* Classification thresholds
* Precision and Recall
* F1-score
* Confusion Matrix
* ROC-AUC
* Model saving and preprocessing pipelines

The main purpose of this project was not just to obtain a high accuracy score, but to understand **how a neural network learns and how its performance should be evaluated**.

---

## Saved Model

The trained model and preprocessing pipeline were saved for future use.

```text
models/
├── bank_marketing_nn.pth
└── bank_marketing_preprocessor.pkl
```

* `bank_marketing_nn.pth` → trained PyTorch model parameters
* `bank_marketing_preprocessor.pkl` → preprocessing pipeline

---

## Important Limitation

One important limitation of this dataset is the `duration` feature.

`duration` represents the duration of the current marketing call. Therefore, it would not be available when making a prediction **before the call begins**.

As a result, this project should primarily be considered a **learning and experimental Deep Learning project**, rather than a production-ready pre-call prediction system.

A future version could investigate a model that excludes `duration` for a more realistic deployment scenario.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* Matplotlib
* Jupyter Notebook

---

## Project Status

**Completed — Deep Learning learning project**

This project represents my first practical exploration of building and evaluating a neural network using PyTorch on a real-world tabular dataset.

The next stage of my Deep Learning journey will move from tabular data toward **Computer Vision and Convolutional Neural Networks (CNNs)**.
