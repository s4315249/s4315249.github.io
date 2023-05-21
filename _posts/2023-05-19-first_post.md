# ResNet18: A Powerful CNN Architecture for Image Classification

## Why ResNet18?

There are several variants of ResNet - ResNet18, ResNet34, ResNet50, etc. The numbers following ResNet signifies the number of neural network layers. In other words, ResNet18 is the simplest of the ResNet models, capable of being trained within a few minutes even on a CPU, while sacrificing some accuracy in comparison to ResNet arhcitecture of higher layers.

## Implementation on fastai
In fastai, ResNet18 is implemented by passing it as an input parameter into the [vision_learner()](https://docs.fast.ai/vision.learner.html) method. See below screenshot of an example implementation:

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
```

where *dls* is the DataBlock object that contains a **training set** and **validation set**, which is initialised prior to this line as follows:


```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=6)
```
