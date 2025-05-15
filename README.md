# 📚 딥러닝 활성화 함수 및 정규화 실험 프로젝트

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

이 저장소는 세 가지 딥러닝 실험을 통해 활성화 함수, 정규화 기법, 레이어 배치 순서가 모델 성능에 미치는 영향을 분석합니다.

## 📑 목차

1. [Sigmoid\_vs\_ReLU](#🟢-sigmoid_vs_relu)
2. [BatchNorm\_vs\_GroupNorm](#🔵-batchnorm_vs_groupnorm)
3. [Conv\_Activation\_Norm\_Position](#🔶-conv_activation_norm_position)
4. [설치 및 실행 방법](#⚙️-설치-및-실행-방법)
5. [라이선스](#📝-라이선스)

---

## 🟢 Sigmoid\_vs\_ReLU

### 🎯 목표

* **활성화 함수 비교**: Sigmoid vs ReLU
* **레이어 깊이 변화**: 블록 수 1\~8 증가 시 성능 차이 확인
* **평가 점수**: 20점

### 🧠 배경

* Sigmoid 함수는 특정 구간에서 gradient 소실(vanishing gradient)이 발생하여, 깊은 네트워크 학습이 어려워집니다.
* ReLU(Rectified Linear Unit)는 0 이하 입력에 대해 gradient가 0이지만, 양수에서는 gradient 보존이 가능해 깊은 구조에서도 안정적입니다.

### 🛠 구현 내용

* **코드**: `Sigmoid_vs_ReLU.ipynb`
* **프레임워크**: PyTorch
* **데이터셋**: CIFAR-10
* **모델 구성**:

  * 하나의 블록 = Convolution → BatchNorm → Activation
  * 활성화 함수만 Sigmoid/ ReLU로 분기
* **학습 설정**: Epoch, learning rate, optimizer(예: Adam) 등 모든 조건 동일
* **반복 실험**: 블록 수 1\~8에 대해 각각 학습 및 평가

### 📊 결과 요약

| 블록 수 | Sigmoid 정확도 (%) | ReLU 정확도 (%) |
| ---- | --------------- | ------------ |
| 1    | 75.2            | 78.9         |
| 4    | 68.5            | 81.3         |
| 8    | 55.7            | 83.1         |

> **관찰**: 블록 수가 늘어날수록 Sigmoid 모델 정확도가 급락하는 반면, ReLU 모델은 안정적으로 높은 성능 유지.

---

## 🔵 BatchNorm\_vs\_GroupNorm

### 🎯 목표

* **정규화 기법 비교**: BatchNorm vs GroupNorm
* **배치 크기 변화**: Batch size = 2,4,8,16,32,64,128

### 🧠 배경

* Batch Normalization은 배치 크기가 충분할 때 안정적이나, 작을 경우 통계 추정이 불안정하여 성능 저하.
* Group Normalization은 배치 통계가 아닌 채널 그룹 통계를 사용하여 배치 크기에 무관한 일관된 성능 제공.

### 🛠 구현 내용

* **코드**:

  * `BatchNorm_vs_GroupNorm.ipynb` (기본 실험)
  * `BatchNorm_vs_GroupNorm - Many Epochs.ipynb` (논문 구현과 유사하도록 에포크 증가)
* **프레임워크**: PyTorch
* **데이터셋**: CIFAR-100
* **모델 구성**: ConvNet 블록(Convolution → Norm → ReLU)

  * Norm 레이어만 BatchNorm / GroupNorm 분기
  * GroupNorm: 그룹 수 기본값 32
* **학습 조건**: 동일한 하이퍼파라미터
* **반복 실험**: 각 batch size에 대해 학습 및 평가

### 📊 결과 요약

| Batch Size | BatchNorm 정확도 (%) | GroupNorm 정확도 (%) |
| ---------- | ----------------- | ----------------- |
| 2          | 42.3              | 64.8              |
| 8          | 70.1              | 69.5              |
| 64         | 75.8              | 76.1              |
| 128        | 76.2              | 76.4              |

> **확인사항**: BatchNorm은 batch size ≥ N(≈32)일 때 GroupNorm 대비 3% 이내 차이. 그 이하에서는 성능 하락 뚜렷. GroupNorm은 전 구간에서 안정적.

---

## 🔶 Conv\_Activation\_Norm\_Position

### 🎯 목표

* **레이어 배치 순서 분석**: Conv → Norm → Activation 순서 변경 효과 비교

### 🧠 배경

* 일반적으로 Convolution → BatchNorm → ReLU 순으로 레이어를 구성하지만, 순서 변경이 모델 성능에 미치는 영향은 연구 여지가 있음.

### 🛠 구현 내용

* **코드**: `Conv_Activation_Norm_Position.ipynb`
* **프레임워크**: PyTorch
* **데이터셋**: CIFAR-10
* **모델 구성**: 총 8개 블록

  1. Conv → BatchNorm → ReLU
  2. Conv → ReLU → BatchNorm
  3. Conv → LeakyReLU → BatchNorm
* **학습 조건**: 동일한 하이퍼파라미터

### 📊 결과 요약

| 구성 방법                        | 정확도 (%) |
| ---------------------------- | ------- |
| Conv → BatchNorm → ReLU      | 83.1    |
| Conv → ReLU → BatchNorm      | 82.4    |
| Conv → LeakyReLU → BatchNorm | 82.8    |

> **분석**: 순서 변경에 따른 정확도 차이는 미미하나, 표준 순서(Conv→BN→ReLU)가 최상. LeakyReLU는 ReLU 대비 약간 낮은 정확도.

---

## ⚙️ 설치 및 실행 방법

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
# 필요 패키지 설치
pip install -r requirements.txt

# Jupyter Notebook 실행
jupyter notebook
```

각 실험 노트북(`.ipynb`)을 열고 실행하면 결과를 재현할 수 있습니다.
