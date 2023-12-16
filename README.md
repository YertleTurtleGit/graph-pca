<font size="7">**Graph PCA**</font>

[![Generate README](https://github.com/YertleTurtleGit/graph-pca/actions/workflows/readme.yml/badge.svg)](https://github.com/YertleTurtleGit/graph-pca/actions/workflows/readme.yml)
<a target="_blank" href="https://colab.research.google.com/github/YertleTurtleGit/graph-pca/blob/main/README.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Performs PCA with optional graph distance for neighborhood composition.

(Still under heavy development.)

<!-- START doctoc -->
<!-- END doctoc -->


# Installation


```python
BUILD_BY_YOURSELF = False

if BUILD_BY_YOURSELF:
    !pip install maturin
    !maturin develop
else:
    !pip install -q git+https://github.com/YertleTurtleGit/graph-pca
```

# Example


```python
!pip install -q numpy opencv-python matplotlib requests
```


```python
import numpy as np
from matplotlib import pyplot as plt
import graph_pca
from graph_pca import Feature
```


```python
# generate test data

fig, ax = plt.subplots(figsize=(20, 20))
ax.set_facecolor("white")
text = "YEAHU"
ax.text(
    0.5,
    0.5,
    text,
    fontsize=50,
    color="black",
    ha="center",
    va="center",
    transform=ax.transAxes,
)

ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

fig.canvas.draw()
image = np.array(fig.canvas.renderer.buffer_rgba())
plt.close(fig)

h, w, _ = image.shape
grid = np.array(np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h)))
points = np.dstack([grid[0, :, :], grid[1, :, :]])
points = points[image[:, :, 0] == 0]
points[:, 1] *= -1

plt.gca().set_aspect("equal")
_ = plt.scatter(points[:, 0], points[:, 1], s=3)
```


    
![png](README_files/README_6_0.png)
    



```python
radius = 0.02
features = [Feature.Eigenvalues, Feature.PrincipalComponentValues]
pca_count = points.shape[1]
```

## Classic PCA


```python
max_edge_length = (
    radius  # This ignores graph neighborhood, because max_edge_length >= radius
)
eigenvalues_xy_graph, pca_xy_graph = np.array(
    graph_pca.calculate_features(points, features, radius, max_edge_length)
)
```


```python
figure, axes = plt.subplots(1, pca_count, figsize=(3.5 * pca_count, 1.5))
for n in range(pca_count):
    _ = axes[n].scatter(points[:, 0], points[:, 1], c=pca_xy_graph[:, n], s=3)
    axes[n].axis("equal")
    axes[n].set_title(f"PCA {n+1}")
```


    
![png](README_files/README_10_0.png)
    


## Graph PCA


```python
max_edge_length = 0.001
eigenvalues_xy_graph, pca_xy_graph = np.array(
    graph_pca.calculate_features(points, features, radius, max_edge_length)
)
```


```python
figure, axes = plt.subplots(1, pca_count, figsize=(3.5 * pca_count, 1.5))
for n in range(pca_count):
    _ = axes[n].scatter(points[:, 0], points[:, 1], c=pca_xy_graph[:, n], s=3)
    axes[n].axis("equal")
    axes[n].set_title(f"Graph PCA {n+1}")
```


    
![png](README_files/README_13_0.png)
    

