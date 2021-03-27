# ZeroGrad : Mitigating and Explaining Catastrophic Overfitting in FGSM Adversarial Training

This repository consists of two methods proposed to prevent a phenomenon called *catastrophic overfitting* in single-step Linf adversarial training, which was first introduced in [Wong et al.](https://arxiv.org/abs/2001.03994) for FGSM-RS adversarial training.

## ZeroGrad

This method prevents catastrophic overfitting by zeroing out a percentage of small gradients while taking the gradient step. Compared to normal FGSM, ZeroGrad has a slightly increased training time and can be implemented by adding the following lines of code (`grad` is the gradient w.r.t to input data):

```python
q_grad = torch.quantile(torch.abs(grad).view(grad.size(0), -1), q_val, dim=1)
grad[torch.abs(grad) < q_grad.view(grad.size(0), 1, 1, 1)] = 0
```

## MultiGrad

MultiGrad zeros out the inconsistent gradients within the epsilon ball around a sample. This is performed by calculating the gradient vectors (w.r.t input data) for a few initial random perturbations in the epsilon ball and zeroing the gradients whose sign is not consistent among all of these vectors.

The calculation of gradients for these random perturbations can be done in parallel using multiple computation units (e.g. GPUs), therefore this method can be used to train a model with a small time-overhead compared to normal FGSM.

The following code is used to implement the explained method (`zeroing_th` can be considered equal to `samples`, which is the number of random samples):

```python
g = sum([torch.sign(grads[i]) for i in range(samples)])
grad = torch.where(torch.abs(g) < 
              (zeroing_th - (samples - zeroing_th)),
              torch.zeros_like(g), g)
```

## Training the models

The code we used as our template is from [Robust Overfitting Github Repository (Rice et al.)](https://github.com/locuslab/robust_overfitting). 

The following commands are used to train the models whose accuracies are reported in the paper.

ZeroGrad on CIFAR-10:
`python train_cifar.py --epochs=30 --lr-schedule=cyclic --lr-max=0.2 --attack=zerograd --zero-qval=0.35`

MultiGrad on CIFAR-10:
`python train_cifar.py --epochs=30 --lr-schedule=cyclic --lr-max=0.2 --attack=multigrad --multi-samps=3`

ZeroGrad on CIFAR-100:
`python train_cifar100.py --epochs=30 --lr-schedule=cyclic --lr-max=0.2 --attack=zerograd --zero-qval=0.45`

MultiGrad on CIFAR-100:
`python train_cifar100.py --epochs=30 --lr-schedule=cyclic --lr-max=0.2 --attack=multigrad --multi-samps=3`

ZeroGrad on SVHN:
`python train_svhn.py --epochs=15 --lr-schedule=cyclic --lr-max=0.01 --attack=zerograd --zero-qval=0.7 --full-test`

MultiGrad on SVHN:
`python train_svhn.py --epochs=15 --lr-schedule=cyclic --lr-max=0.01 --attack=multigrad --multi-samps=3 --full-test`

