# Confusion Matrix and Related Metrics

## What is a Confusion Matrix?

An intuitive analytical tool that can provide useful insight into our trained model is the Confusion matrix.

<table>
      <tr>
          <th rowspan="2">Actual</th>
          <th>Negative</th>
          <th>FP</th>
          <th>TN</th>
      </tr>
      <tr>
          <td>Positive</td>
          <td style="text-align: center">TP</td>
          <td style="text-align: center">FN</td>
      </tr>
      <tr>
          <td></td>
          <td></td>
          <td>Positive</td>
          <td>Negative</td>
      </tr>
      <tr>
          <td></td>
          <td></td>
          <td colspan="2">Predicted</td>
      </tr>
</table>

Above is a simple representation of a confusion matrix for a binary classification.

It can be observed that there are four main components for a confusion matrix:

1. True Positives (TP) - number of instances that are correctly predicted (belonging to a specific class)
2. True Negatives (TN) - number of instances that are correctly predicted as negative (i.e. not belonging to a specific class)
3. False Positives (FP) - number of instances that are incorrectly predicted as positive (i.e. predicted to belong to a specific class) when in fact they belong to a different class
4. False Negatives (FN) - number of instances that are incorrectly predicted as negative (i.e. predicted to not belong to a specific class) when in fact they do belong to that class.

For a non-binary classification such as this, the number of classes `n` determines the size of the matrix as `n x n`. 

The diagonal from top left to bottom right represents TP for each class while the off-diagonal elements represent FP and FN for each class.

FP and FN are also known as Type I and Type II error respectively, and the lower their values, the better the model is at predicting the correct classification.

In fastai, it is simply implemented by passing the model as an input parameter into `from_learner()` function within `ClassificationInterpretation` and then plotting it using `plot_confusion_matrix()`.

```python
interep = ClassificationInterpretation.from_learner(learn)
interep.plot_confusion_matrix()
```

## Some Useful Relevant Metrics

Precision, Recall and F1-Score are some of the important metrics that can be determined through the confusion matrix values from above. Here is a brief explanation of what each metrics describe about the model:

1. Precision - how correctly the predicted cases turned out to be positive
$$Precision =\frac{TP}{TP+FP}$$

2. Recall - how many of the actual positive cases were able to predict correctly with the model
$$Recall = \frac{TP}{TP+FN}$$

3. F1-Score - a harmonic mean of Precision and Recall, gives a combined idea of the two metrics, but has poor interpretability as it does not report on which of the two is being maximised
$$F1Score = \frac{2}{\frac{1}{Recall}+\frac{1}{Precision}}$$

## Resources
1. Bhandari A., 2023, Understanding & Interpreting Confusion Matrices for Machine Learning, https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/
