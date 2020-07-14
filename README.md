# ERMvsARMvsMixup
Emperical Risk Minimisation, Adversarial Risk Minimisation, Mixup
## Prerequisites

- PyTorch 1.4+

## Overview
Given a highly imbalanced dataset, I wanted to investigate how adversarial training might help in increasing performance. Additionally I also compare it with mixup augmentation. Project Gradient Descent (l-infinity ball, untargeted) along with Focal loss is used to perform adversarial training. So in total 6 different ways are used to train the network.

1. <b>Emperical Risk Minimisation</b>
    - Vanilla CCE Loss with SGDM
    - Focal Loss with Class Frequency (alpha in paper) with SGDM
    - Focal Loss without Class Frequency (alpha in paper) with SGDM

2. <b>Adversarial Risk Minimisation</b>
    - PGD (l-infinity ball, untargeted) with focal loss (without class frequency)

3. <b>Mixup Augmentation</b>
    - Vanilla CCE Loss for mixup
    - Focal Loss with Class Frequency (alpha in paper) with SGDM

## Dataset
The dataset has 23 attributes. Some of the important features are:
1.	ra, dec — right ascension and declination respectively.
2.	u, g, r, i, z — filter bands (a.k.a. photometric system or astronomical magnitudes)
3.	nuv_mag — the near-UV aperture magnitude of the source, through a fixed 8
                       arc second aperture. <a href = "https://heasarc.nasa.gov/W3Browse/all/uit.html"> [1] </a> 
                       
(Other columns are linear combinations of u, g, r, i, z, nuv_mag) 

| Type  | Number of Samples |
| ------- | ---------- |
| Class 0 (Galaxy) | 894 |  
| Class 1 (Star) | 5540 |
| Class 2 (Quasar) | 1972 |

## Experiments
Since this dataset was provided as a part of class project and was collected by our instructor, I don't really know the bounds or the distribution of the data. So i just took a shot for ϵ (projection value) for PGD. Hence there might be better values for performing adversarial training. For Focal loss recommended default value γ = 2 is used.

## Setup
- batch-size = 128
- opt = SGDM
- step size,momentum = 0.1,0.9
- 80-20 split

## Results
Since dataset is imbalanced, particularly Class 0 is minority, I report F1-score of Class 0 along with accuracy of all classes.
| Category | Model   | F1 Score for Class 0| Test Acc.  |
| ---------- | ------- | ---------- | ---------  |
| ERM | Vanilla CCE Loss with SGDM | 0.68 | 90.90 |
| ERM | Focal Loss with Class Frequency | 0.71 | 90.72 |
| ERM | Focal Loss without Class Frequency | 0.70 | 91.0 |
| ARM | PGD ( ℓ ∞ ball, untargeted) with focal loss | 0.72 | 90.80 |
| Mixup | Vanilla CCE Loss for mixup | 0.66 | 91.2 |
| Mixup | Focal Loss with Class Frequency | 0.67 | 89.42 |
