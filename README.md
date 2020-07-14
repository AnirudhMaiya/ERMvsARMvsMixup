# ERMvsARMvsMixup
Emperical Risk Minimisation, Adversarial Risk Minimisation, Mixup
## Prerequisites

- PyTorch 1.4+

## Overview
Given a highly imbalanced dataset, I wanted to investigate how adversarial training might help in increasing performance. Additionally I also compare it with mixup augmentation. Project Gradient Descent (l-infinity ball, untargeted) along with Focal loss is used to perform adversarial training. So in total 6 different ways are used to train the network.

1. Emperical Risk Minimisation
    - Vanilla CCE Loss with SGDM
    - Focal Loss with Class Frequency (alpha in paper) with SGDM
    - Focal Loss without Class Frequency (alpha in paper) with SGDM

2. Adversarial Risk Minimisation
    - PGD (l-infinity ball, untargeted) with focal loss

3. 
