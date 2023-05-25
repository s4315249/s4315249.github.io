# Does Fine Tuning Really Help?

## What is fine tuning?

The process of fine-tuning is often referred to as performing "network surgery" - cutting off the final set of FC layers, replacing the head with a *new* set of FC layers with random initialisations, freezing all layers below the head, then training the network at a very small learning rate to learn patterns from the *previously learned* CONV layers in the network (Rosebrock, 2019).

See below illustration for a visual representation of this process:

<img src="https://b2633864.smushcdn.com/2633864/wp-content/uploads/2019/06/fine_tuning_keras_freeze_unfreeze.png?lossy=1&strip=1&webp=1" width="400">
<p style="font-size:10px">A typical fine tuning process starts with freezing early layers in network, training the fully connected (FC) layers, unfreezing the early layers and training all layers.</p>


## So how is it actually implemented?

```python
learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(5)
```

The implementation is quite simple. The model is first trained using your model of choice. Then, the `fine_tune()` function is called on the model with an `epoch` number as its input parameter.

An `epoch` is a complete pass through the entire training dataset, during which the model processes and learns from each sample in the training dataset once. 

Depending on the dataset, the ideal epoch count will vary. For dataset of higher complexity, the epoch count may need to be higher to fine tune the model over multiple iterations, while an epoch count that is too high for a simple classification task may lead to overfitting of the data.

So to answer whether it helps to fine tune your deep learning model - yes, of course it does. As long as you don't overdo it with the epochs! Take below example of `resnet18` fine tuned to 10 epochs:

<img src="/images/image.png" alt="epoch 10 results" style="width: 300px;">

It can be observed that the final `accuracy` metrics of the model has actually dropped from its highest point which is on the 5th epoch (89.6% to 88.5%), indicating that the model has potentially overfitted the dataset.

By monitoring the epoch at which the validation loss is at the lowest, it can further verify when further training may not improve the performance significantly, which in this case happens to be the same epoch - the 5th epoch.

Therefore, the ideal number of epochs for this particular example is `5`, and it goes to show that higher iteration of fine tuning is not always better.
