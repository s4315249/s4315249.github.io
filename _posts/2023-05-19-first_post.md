# ResNet18: A Powerful CNN Architecture for Image Classification

## Why ResNet18?

There are several variants of ResNet - ResNet18, ResNet34, ResNet50, etc. The numbers following ResNet signifies the number of neural network layers. In other words, ResNet18 is the simplest of the ResNet models, capable of being trained within a few minutes even on a CPU, while sacrificing some accuracy in comparison to ResNet arhcitecture of higher layers.

## Implementation on fastai
In fastai, ResNet18 is implemented by passing it as an input parameter into the [vision_learner()](https://docs.fast.ai/vision.learner.html) method. See below screenshot of an example implementation:

![Image of screenshot](images/image.png)
