**📌 FIRST PART : NYSE Stock Price Prediction using PyTorch**

**What I Learned from This Lab**
During this lab, I gained hands-on experience with PyTorch for regression tasks, learning how to build and train a Deep Neural Network (DNN) for stock price prediction. I explored the NYSE dataset, performed Exploratory Data Analysis (EDA), and applied feature scaling to preprocess the data.

Through model training and hyperparameter tuning, I understood the impact of different architectures, optimizers, and learning rates on model performance. Additionally, I experimented with regularization techniques (Dropout & L2) to reduce overfitting and improve generalization.

Overall, this lab enhanced my understanding of deep learning for regression, model optimization, and the importance of visualizing training metrics to assess performance. It also strengthened my ability to apply ML concepts in real-world financial datasets.

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

