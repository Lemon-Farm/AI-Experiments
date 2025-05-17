# 📚 딥러닝 실험 프로젝트

세 가지 딥러닝 실험을 통해 활성화 함수, 정규화 기법, 레이어 배치 순서가 모델 성능에 미치는 영향을 분석하였습니다.

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

## 🔵 BatchNorm\_vs\_GroupNo다.

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

> **결과**: 표준 순서(Conv→BN→ReLU)가 가장 우수한 성능 보이며, 이어서 (Conv→LeakyReLU→BN)의 성능이 높았고, (Conv→ReLU→BN)이 가장 낮은 성능을 보임.
>
> 
> **원인 분석**: Conv→BN→ReLU 이 구조는 선형 출력값을 정규화한 뒤 비선형성(ReLU)을 적용해 주기 때문에, 안정적인 분포와 원활한 그래디언트 흐름을 보장하여 최상의 성능을 낸다. Conv→ReLU→BN 은 ReLU가 음수 값을 제거한 뒤 BN이 이를 정규화하다 보니 편향된 분포가 형성되고, Conv→LeakyReLU→BN 은 LeakyReLU가 일부 음수 정보를 유지해 성능이 다소 개선된 결과를 보인 것.

---

## ⚙️ 설치 및 실행 방법

```bash
git clone https://github.com/Lemon-Farm/AI-Experiments-01.git
cd AI-Experiments-01
```

각 실험 노트북(`.ipynb`)을 열어 실행하면 결과를 재현할 수 있습니다.

---
