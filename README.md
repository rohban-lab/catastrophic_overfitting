# ZeroGrad : Mitigating and Explaining Catastrophic Overfitting in FGSM Adversarial Training


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

