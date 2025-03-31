**📌 Atelier1 : NYSE Stock Price Prediction using PyTorch**

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

**📌 Atelier2: CNN, Faster R-CNN, and ViT for MNIST Classification**

### **What I Learned from This Lab**
During this lab, I gained hands-on experience with PyTorch to develop deep learning models for computer vision tasks. I built and trained CNN and Faster R-CNN architectures for MNIST classification, exploring different hyperparameters such as kernel size, padding, stride, optimizers, and regularization techniques.

I also fine-tuned pre-trained models (VGG16 and AlexNet) on MNIST and compared their performance against CNN and Faster R-CNN. Additionally, I implemented a Vision Transformer (ViT) from scratch following an online tutorial and evaluated its classification performance.

Through this process, I deepened my understanding of neural network architectures, transfer learning, and model optimization. The comparative analysis between CNN, Faster R-CNN, and ViT provided valuable insights into their strengths and weaknesses in image classification tasks.



### **Models Implemented**
- **CNN**: Built from scratch with convolutional, pooling, and fully connected layers.
- **Faster R-CNN**: Implemented for object detection and classification.
- **Transfer Learning**: Fine-tuned pre-trained VGG16 and AlexNet on MNIST.
- **Vision Transformer (ViT)**: Developed from scratch and applied to MNIST classification.

### **Comparison & Analysis**
- Evaluated models based on accuracy, F1-score, loss, and training time.
- Assessed the impact of transfer learning on performance.
- Compared CNN, Faster R-CNN, and ViT architectures.

----





📌 Developed by: **Safae Hammouch**

