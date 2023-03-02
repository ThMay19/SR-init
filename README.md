# SR-init
Offical Codes for SR-init: AN INTERPRETABLE LAYER PRUNING METHOD

**Abstract**: Despite the popularization of deep neural networks (DNNs) in many fields, it is still challenging to deploy state-of-the-art models to resource-constrained devices due to high computational overhead. Model pruning provides a feasible solution to the aforementioned challenges. However, the interpretation of existing pruning criteria is always overlooked. To counter this issue, we propose a novel layer pruning method by exploring the Stochastic Re-initialization. Our SR-init method is inspired by the discovery that the accuracy drop due to stochastic re-initialization of layer parameters differs in various layers. On the basis of this observation, we come up with a layer pruning criterion, i.e., those layers that are not sensitive to stochastic re-initialization (low accuracy drop) produce less contribution to the model and could be pruned with acceptable loss. Afterward, we experimentally verify the interpretability of SR-init via feature visualization. The visual explanation demonstrates that SR-init is theoretically feasible, thus we compare it with state-of-the-art methods to further evaluate its practicability. As for ResNet56 on CIFAR-10 and CIFAR-100, SR-init achieves a great reduction in parameters (63.98% and 37.71%) with an ignorable drop in top-1 accuracy (-0.56% and 0.8%). With ResNet50 on ImageNet, we achieve a 15.59% FLOPs reduction by removing 39.29% of the parameters, with only a drop of 0.6% in top-1 accuracy.

 This figure is visual interpretation for Stochastic Re-initialization (SR-init).
![image](https://github.com/huitang-zjut/SR-init/blob/main/img/visualization.png)

# Useage
To calculate the estimation accuracy:

```python
python get_estimation_accuracy.py --arch resnet56 --set cifar10 --num_classes 10 --random_seed 1 --evaluate

```
