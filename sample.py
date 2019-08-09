from sklearn.datasets.samples_generator import make_blobs

import dimensionreduction

# Data preparation
vectors, _ = make_blobs(n_samples=400,
                        centers=[[1.0,1.0,1.0], [-1.0,-1.0,-1.0]],
                        cluster_std=0.5)

# Test PCA
outputs = dimensionreduction.reduce_dimension(
        vectors=vectors,
        method="pca",
        pre_normalization=True,
        target_dim=2,
        params={})

# Test T-SNE
outputs = dimensionreduction.reduce_dimension(
        vectors=vectors,
        method="tsne",
        pre_normalization=True,
        target_dim=2,
        params={"perplexity": 100})

# Test Spectral Embedding
outputs = dimensionreduction.reduce_dimension(
        vectors=vectors,
        method="spectral",
        pre_normalization=True,
        target_dim=2,
        params={})

print("OK")
