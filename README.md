# 📚 딥러닝 실험 프로젝트

여러 딥러닝 실험을 통해 이론적 배경을 실제 환경에서 재현하고, 이들이 모델 성능에 미치는 영향을 분석하였습니다.

## 📑 목차

1. [Sigmoid\_vs\_ReLU](#🟢-sigmoid_vs_relu)
2. [BatchNorm\_vs\_GroupNorm](#🔵-batchnorm_vs_groupnorm)
3. [Conv\_Activation\_Norm\_Position](#🔶-conv_activation_norm_position)
4. [설치 및 실행 방법](#⚙️-설치-및-실행-방법)

---

## 🟢 Sigmoid\_vs\_ReLU

### 🎯 목표

* 활성화 함수 비교: Sigmoid vs ReLU
* 레이어 깊이(블록 수) 1\~8 증가 시 성능 차이 확인

### 🧠 배경

Sigmoid 함수는 특정 구간에서 gradient 소실(vanishing gradient)이 발생하여 깊은 네트워크 학습이 어려워집니다. ReLU(Rectified Linear Unit)는 양수 영역에서 gradient가 보존되어, 깊은 구조에서도 안정적으로 학습할 수 있습니다.

### 🛠 구현 내용

* 코드: `Sigmoid_vs_ReLU.ipynb`
* 프레임워크: PyTorch
* 데이터셋: CIFAR-10
* 모델 구성: 하나의 블록 = Convolution → BatchNorm → Activation
* 활성화 함수만 Sigmoid / ReLU로 분기
* 학습 설정: 동일한 epoch, learning rate, optimizer 적용
* 반복 실험: 블록 수 1\~8 각각에 대해 학습 및 평가
* 실험 환경: Google Colab T4 GPU

### 📊 결과 요약

| 블록 수 | Sigmoid 정확도 (%) | ReLU 정확도 (%) |
| ---- | --------------- | ------------ |
| 1    | 19.00           | 23.01        |
| 2    | 21.75           | 29.19        |
| 3    | 22.00           | 32.63        |
| 4    | 22.95           | 35.18        |
| 5    | 21.93           | 35.18        |
| 6    | 22.55           | 36.20        |
| 7    | 21.18           | 37.49        |
| 8    | 18.59           | 37.61        |

> **관찰**: 블록 수 증가 시 Sigmoid 모델의 성능이 크게 저하되는 반면, ReLU 모델은 깊어져도 안정적인 성능 유지.

---

## 🔵 BatchNorm\_vs\_GroupNorm

### 🎯 목표

* 정규화 기법 비교: BatchNorm vs GroupNorm
* 배치 크기(batch size) = 2, 4, 8, 16, 32, 64, 128로 설정

### 🧠 배경

Batch Normalization은 배치 크기가 충분할 때 안정적이나, 작을 경우 통계 추정이 불안정하여 성능 저하. Group Normalization은 배치 통계가 아닌 채널 그룹 통계를 사용하여 배치 크기에 무관한 일관된 성능 제공.

### 🛠 구현 내용

* 코드:

  * `BatchNorm_vs_GroupNorm.ipynb` (기본 실험)
  * `BatchNorm_vs_GroupNorm - Many Epochs.ipynb` (논문 구현과 유사하도록 에포크 증가)
* 프레임워크: PyTorch
* 데이터셋: CIFAR-100
* 모델 구성: ConvNet 블록(Convolution → Norm → ReLU)

  * Norm 레이어만 BatchNorm / GroupNorm 분기
  * GroupNorm: 그룹 수 32로 고정
* 학습 조건: 동일한 하이퍼파라미터
* 반복 실험: 각 batch size에 대해 학습 및 평가

### 📊 결과 요약

| Batch Size | BatchNorm 정확도 (%) | GroupNorm 정확도 (%) |
| ---------- | ----------------- | ----------------- |
| 2          | 10.01             | 30.89             |
| 4          | 33.81             | 40.46             |
| 8          | 44.55             | 44.27             |
| 16         | 49.68             | 48.85             |
| 32         | 52.16             | 50.65             |
| 64         | 53.18             | 51.37             |
| 128        | 52.05             | 50.09             |

> **확인사항**: BatchNorm은 batch size ≥ 8 이상에서 GroupNorm 대비 3% 이내 차이를 보이지만, 그 이하에서는 성능이 급락. GroupNorm은 모든 배치 크기에서 안정적 성능 유지.

---

## 🔶 Conv\_Activation\_Norm\_Position

### 🎯 목표

레이어 구성 순서 변경(Conv → Norm → Activation)의 성능 영향 분석

### 🧠 배경

일반적으로 Convolution → BatchNorm → ReLU 순으로 구성하지만, 순서 변경이 모델 성능에 미치는 영향은 연구 여지가 있습니다.

### 🛠 구현 내용

* 코드: `Conv_Activation_Norm_Position.ipynb`
* 프레임워크: PyTorch
* 데이터셋: CIFAR-10
* 모델 구성: 총 8개 블록

  1. Conv → BatchNorm → ReLU
  2. Conv → ReLU → BatchNorm
  3. Conv → LeakyReLU → BatchNorm
* 학습 조건: 동일한 하이퍼파라미터

### 📊 결과 요약

| 구성 방법                        | 정확도 (%) |
| ---------------------------- | ------- |
| Conv → BatchNorm → ReLU      | 62.64   |
| Conv → ReLU → BatchNorm      | 58.50   |
| Conv → LeakyReLU → BatchNorm | 60.09   |

> **분석**: 순서 변경에 따른 정확도 차이는 크지 않으나, 표준 순서(Conv→BN→ReLU)가 가장 우수한 성능 보임.

---

## ⚙️ 설치 및 실행 방법

```bash
git clone https://github.com/Lemon-Farm/AI-Experiments.git
cd AI-Experiments
```

각 실험 노트북(`.ipynb`)을 열어 실행하면 결과를 재현할 수 있습니다.

---
