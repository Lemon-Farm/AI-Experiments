# 📚 Deep Learning Experiment Project

Through various deep learning experiments, the theoretical foundations were reproduced in practice and their effects on model performance were analyzed.

## 📑 Table of Contents

1. [Sigmoid_vs_ReLU](#🟢-sigmoid_vs_relu)  
2. [BatchNorm_vs_GroupNorm](#🔵-batchnorm_vs_groupnorm)  
3. [Conv_Activation_Norm_Position](#🔶-conv_activation_norm_position)  
4. [Installation and Execution](#⚙️-installation-and-execution)

---

## 🟢 Sigmoid_vs_ReLU

### 🎯 Objective

- Compare activation functions: Sigmoid vs. ReLU  
- Measure performance difference as layer depth (number of blocks) increases from 1 to 8

### 🧠 Background

The sigmoid function suffers from vanishing gradients in certain regions, making it difficult to train deep networks. ReLU (Rectified Linear Unit), by preserving gradients in the positive region, enables stable training even in deep architectures.

### 🛠 Implementation Details

- Notebook: `Sigmoid_vs_ReLU.ipynb`  
- Framework: PyTorch  
- Dataset: CIFAR-10  
- Model design: each block = Convolution → BatchNorm → Activation  
- Branch on activation function (Sigmoid vs. ReLU) only  
- Training settings: identical epochs, learning rate, optimizer  
- Repeated experiments for block counts 1 through 8  
- Runtime: Google Colab T4 GPU

### 📊 Results Summary

| Number of Blocks | Sigmoid Accuracy (%) | ReLU Accuracy (%) |
| ---------------- | -------------------- | ----------------- |
| 1                | 19.00                | 23.01             |
| 2                | 21.75                | 29.19             |
| 3                | 22.00                | 32.63             |
| 4                | 22.95                | 35.18             |
| 5                | 21.93                | 35.18             |
| 6                | 22.55                | 36.20             |
| 7                | 21.18                | 37.49             |
| 8                | 18.59                | 37.61             |

> **Observation**: As depth increases, the Sigmoid model’s performance deteriorates significantly, whereas the ReLU model maintains stable performance even as it deepens.

---

## 🔵 BatchNorm_vs_GroupNorm

### 🎯 Objective

- Compare normalization methods: BatchNorm vs. GroupNorm  
- Evaluate across batch sizes of 2, 4, 8, 16, 32, 64, and 128

### 🧠 Background

Batch Normalization is stable when the batch size is large enough, but its statistical estimates become unreliable with small batches, leading to performance drops. Group Normalization uses per-channel group statistics instead of batch statistics, offering consistent performance regardless of batch size.

### 🛠 Implementation Details

- Notebooks:  
  - `BatchNorm_vs_GroupNorm.ipynb` (baseline)  
  - `BatchNorm_vs_GroupNorm - Many Epochs.ipynb` (more epochs to match paper settings)  
- Framework: PyTorch  
- Dataset: CIFAR-100  
- Model design: ConvNet blocks (Convolution → Normalization → ReLU)  
  - Branch on normalization layer (BatchNorm vs. GroupNorm)  
  - GroupNorm: fixed to 32 groups  
- Training settings: identical hyperparameters  
- Repeated experiments for each batch size  

### 📊 Results Summary

| Batch Size | BatchNorm Accuracy (%) | GroupNorm Accuracy (%) |
| ---------- | ---------------------- | ---------------------- |
| 2          | 10.01                  | 30.89                  |
| 4          | 33.81                  | 40.46                  |
| 8          | 44.55                  | 44.27                  |
| 16         | 49.68                  | 48.85                  |
| 32         | 52.16                  | 50.65                  |
| 64         | 53.18                  | 51.37                  |
| 128        | 52.05                  | 50.09                  |

> **Observation**: BatchNorm lags significantly for batch sizes < 8 but matches GroupNorm within 3% for larger batches. GroupNorm remains stable across all sizes.

---

## 🔶 Conv_Activation_Norm_Position

### 🎯 Objective

Analyze the impact of changing the order of layers (Convolution → Normalization → Activation)

### 🧠 Background

The standard sequence is Convolution → BatchNorm → ReLU, but altering this order may affect performance.

### 🛠 Implementation Details

- Notebook: `Conv_Activation_Norm_Position.ipynb`  
- Framework: PyTorch  
- Dataset: CIFAR-10  
- Model design: 8 blocks with:  
  1. Conv → BatchNorm → ReLU  
  2. Conv → ReLU → BatchNorm  
  3. Conv → LeakyReLU → BatchNorm  
- Training settings: identical hyperparameters

### 📊 Results Summary

| Configuration                      | Accuracy (%) |
| ---------------------------------- | ------------ |
| Conv → BatchNorm → ReLU            | 62.64        |
| Conv → ReLU → BatchNorm            | 58.50        |
| Conv → LeakyReLU → BatchNorm       | 60.09        |

> **Analysis**: The standard order (Conv→BN→ReLU) yields the highest accuracy, though differences are modest.

---

## ⚙️ Installation and Execution

```bash
git clone https://github.com/Lemon-Farm/AI-Experiments.git
cd AI-Experiments
# Open your preferred notebook and run
```
