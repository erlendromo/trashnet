import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_pca_feature_space(X, y, title="Feature Space Visualization"):
    """
    Visualize high-dimensional feature vectors using PCA (2D scatter plot).

    Parameters:
    ----------
    X : list or np.ndarray
        Shape: (n_samples, n_features)
        Combined feature vectors for all images.

    y : list or np.ndarray
        Shape: (n_samples,)
        Class labels for each image.
        Example:
        ["paper", "glass", "metal", "plastic", "cardboard"]

    title : str
        Plot title

    Returns:
    -------
    None
        Displays PCA scatter plot.
    """

    X = np.array(X)
    y = np.array(y)

    # Reduce to 2 dimensions using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Unique class labels
    classes = np.unique(y)

    # Optional fixed colors for your trash categories
    color_map = {
        "paper": "blue",
        "cardboard": "saddlebrown",
        "plastic": "red",
        "metal": "gray",
        "glass": "green"
    }

    plt.figure(figsize=(10, 8))

    for cls in classes:
        idx = y == cls

        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=cls,
            alpha=0.7,
            s=60,
            c=color_map.get(cls, None)
        )

    plt.title(title, fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional: print explained variance
    # print("Explained variance ratio:")
    # print(f"PC1: {pca.explained_variance_ratio_[0]:.4f}")
    # print(f"PC2: {pca.explained_variance_ratio_[1]:.4f}")
    # print(f"Total: {sum(pca.explained_variance_ratio_):.4f}")


def plot_tsne_feature_space(X, y, title="t-SNE Feature Space Visualization"):
    """
    Visualize high-dimensional feature vectors using t-SNE (2D scatter plot).

    Parameters
    ----------
    X : list or np.ndarray
        Shape: (n_samples, n_features)
        Combined feature vectors for all images

    y : list or np.ndarray
        Shape: (n_samples,)
        Class labels for each image
        Example:
        ["paper", "glass", "metal", "plastic", ...]

    title : str
        Plot title

    Returns
    -------
    None
        Displays t-SNE scatter plot
    """

    X = np.array(X)
    y = np.array(y)

    # t-SNE dimensionality reduction
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=42
    )

    X_tsne = tsne.fit_transform(X)

    # Class labels
    classes = np.unique(y)

    # Fixed colors for trash categories
    color_map = {
        "paper": "blue",
        "cardboard": "saddlebrown",
        "plastic": "red",
        "metal": "gray",
        "glass": "green"
    }

    plt.figure(figsize=(10, 8))

    for cls in classes:
        idx = y == cls

        plt.scatter(
            X_tsne[idx, 0],
            X_tsne[idx, 1],
            label=cls,
            alpha=0.75,
            s=70,
            c=color_map.get(cls, None)
        )

    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_umap_feature_space(X, y, title="UMAP Feature Space Visualization"):
    """
    Visualize high-dimensional feature vectors using UMAP (2D scatter plot).

    Parameters
    ----------
    X : list or np.ndarray
        Shape: (n_samples, n_features)
        Combined feature vectors for all images

    y : list or np.ndarray
        Shape: (n_samples,)
        Class labels for each image

    title : str
        Plot title

    Returns
    -------
    None
        Displays UMAP scatter plot
    """

    X = np.array(X)
    y = np.array(y)

    # UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )

    X_umap = reducer.fit_transform(X)

    # Class labels
    classes = np.unique(y)

    # Fixed colors
    color_map = {
        "paper": "blue",
        "cardboard": "saddlebrown",
        "plastic": "red",
        "metal": "gray",
        "glass": "green"
    }

    plt.figure(figsize=(10, 8))

    for cls in classes:
        idx = y == cls

        plt.scatter(
            X_umap[idx, 0],
            X_umap[idx, 1],
            label=cls,
            alpha=0.75,
            s=70,
            c=color_map.get(cls, None)
        )

    plt.title(title, fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
