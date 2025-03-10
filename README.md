**📌 FIRST PART : NYSE Stock Price Prediction using PyTorch**

**Objective**
Familiarize with PyTorch by implementing regression using Deep Neural Networks (DNN/MLP) on NYSE stock prices.

**Dataset**
- **Source**: [NYSE Dataset on Kaggle](https://www.kaggle.com/datasets/dgawlik/nyse)
- **Features**: `open`, `high`, `low`, `volume`
- **Target**: `close`

**Work Done**

### **1. Exploratory Data Analysis (EDA)**
- **Checked**: Missing values, statistical summary
- **Visualized**: Histograms, correlation heatmap, pairplot

### **2. Data Preprocessing**
- **Selected Features**: `open`, `high`, `low`, `volume`
- **Standardized**: Used `StandardScaler`
- **Split**: 80% train, 20% test

### **3. DNN Model with PyTorch**
- **Architecture**:
  - Input: 4 features
  - Hidden Layers: 64 → 32 (ReLU activation)
  - Output: 1 neuron (predict `close` price)
- **Loss Function**: MSE
- **Optimizer**: Adam (lr = 0.001)
- **Epochs**: 100

### **4. Hyperparameter Tuning**
- Used **GridSearch** to optimize:
  - Learning rate, optimizers, epochs, model architecture

### **5. Training & Evaluation**
- **Tracked**: Train/Test loss over 100 epochs
- **Visualized**: Loss vs. Epochs, Accuracy vs. Epochs

### **6. Regularization Techniques**
- Applied **Dropout & L2 regularization**
- **Improved** model generalization and reduced overfitting

**Results & Observations**
✔ Model effectively learned stock price relationships
✔ Regularization improved performance
✔ Best setup: **ReLU + Adam + L2 regularization**
✔ Loss graph showed smooth convergence



---

📌 Developed by: **Safae Hammouch**

