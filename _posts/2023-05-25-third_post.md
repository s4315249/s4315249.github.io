# t-SNE - What Exactly Is It?

## T-Distributed Stochastic Neighbour Embedding (t-SNE)

**t-distributed stochastic neighbour embedding (t-SNE)** is arguable one of the most difficult ideas to understand as a beginner just starting on their deep learning journey. I know I definitely struggled with grasping its concept. 

To put it simply, t-SNE allows for the visualisation of similarities between different data points by first taking them at a high-dimensional space and then displaying them in a low-dimensional space (typically the 2D plane).

t-SNE comprises of two main stages - probability distribution over pairs of high-dimensional objects and a similar probability distribution over the points in the low-dimensional map such that the *Kullback-Leibler (KL)* divergence is minimised.

## 1. High-Dimensional Probability Distribution

In the first stage of t-SNE, the algorithm aims to create a high-dimensional probability distribution for the data points. The high-dimensional space represents the original features or attributes of the data points. The goal is to determine the similarity or relationship between pairs of data points based on their feature similarities.

## 2. Low-Dimensional Probability Distribution with KL Divergence Minimisation

In the second stage of t-SNE, the algorithm aims to create a low-dimensional probability distribution for the data points. The low-dimensional space represents the embedding or visualization of the data points, typically in two or three dimensions for visualization purposes.

The algorithm iteratively adjusts the positions of the data points in the low-dimensional space to minimize the KL divergence between the high-dimensional and low-dimensional probability distributions. KL divergence is a measure of the difference between two probability distributions.

By minimizing the KL divergence, t-SNE tries to find an arrangement of data points in the low-dimensional space that captures the similarities and structure observed in the high-dimensional space. The iterative process continues until a stable low-dimensional embedding is achieved.

## Machine Learning Implementation

```python
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import numpy as np

x, y = learn.get_preds(0)
z = x.numpy()

tsne = TSNE()
embed = tsne.fit_transform(z)
embed.shape

tsne_data_frame = pd.DataFrame({'tsne_1': embed[:,0], 'tsne_2': embed[:,1], 'label': y})
figure, axes = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_data_frame, axes=axes, s=120, palette='tab10', legend='full')
limits = (embed.min()-10, embed.max()+10)
axes.set_xlim(limits)
axes.set_ylim(limits)
axes.set_aspect('equal')
axes.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
```

Above is a simple barebone implementation of t-SNE using scikit-learn library. `get_pred()` function is used to obtain predictions `x` and labels `y` from the pretrained model `learn`. `x` is converted to a NumPy array and assigned to `z` to which the `fit_transform()` function is called to map the high-dimensional data to a low-dimensional space. The resulting embeddings are assigned to `embed`.

Then, a pandas DataFrame `tsne_data_frame` is initialised to store the embedding coordinates `tsne_1` and `tsne_2`. Finally, the resulting t-SNE embedding can be visualised via `scatterplot()`.

See belew result:
<img src="/images/tsne_embedding.jpg" alt="tsne results" style="width: 400px;">

## Resources
1. Wattenberg, et al., 2016, How to Use t-SNE Effectively, http://doi.org/10.23915/distill.00002
