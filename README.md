# tau-ResNet

This repo reimplements the experiments presented in "Convergence Theory of Learning Over-parameterized ResNet: A Full Characterization" (https://arxiv.org/abs/1903.07120).

The codes were tested using Python 3.6 and Pytorch 1.0.

# Experiments on CIFAR10

First open the cifar floder.

The following command trains the Resnet110 baseline model. All depths supported are 20/32/44/56/110/1202. The program would download the dataset automatically if this is first running.

```
CUDA_VISIBLE_DEVICES=0 python cifar_train.py --arch resnet110 --sess baseline
```

The network architecture and hyperparameters are the same as in "Deep Residual Learning for Image Recognition". The result can be found in the result folder. 

The authors of "Convergence Theory of Learning Over-parameterized ResNet: A Full Characterization" suggest adding a scale factor tau=1/sqrt(L) at the output of each residual block, where L is the number of residual blocks (54 for resnet110). 

The following command trains the Resnet110 model with tau=0.136.

```
CUDA_VISIBLE_DEVICES=0 python cifar_train.py --arch resnet110 --tau 0.136 --sess tau0.136
```

The following chart shows the comparison between tau-ResNet and ResNet with different depths.

<img src="cifar-bn.png" width="650" height="500">

You can also train Resnet model without batch normalization. The following command trains Resnet110 model without any bn layer.

```
CUDA_VISIBLE_DEVICES=0 python cifar_train.py --arch resnet110_nobn --tau 0.136 --sess tau0.136_nobn
```

When withoutbn, the training explodes for larger tau.

# Experiments on ImageNet

Go to the imagenet folder. You need to download the ImageNet classification dataset from http://www.image-net.org/challenges/LSVRC/2012/ first.

The authors set L as the largest number of blocks over all stages. The following command trains the Resnet101 model with tau=0.4. All depths supported are 50/101/152. 

```
python imagenet_train.py --arch resnet101 --tau 0.4 --sess imagenet_tau0.4 --data_dir data_folder_path
```

The following command trains the Resnet101 model without batch normalization.

```
python imagenet_train.py --arch resnet101_nobn --tau 0.4 --sess imagenet_tau0.4_nobn --data_dir data_folder_path
```
