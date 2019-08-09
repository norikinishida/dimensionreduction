import numpy as np

from .pca import PCA
from .tsne import TSNE
from .spectral import SpectralEmbedding

def normalize(vectors):
    norm = np.linalg.norm(vectors, axis=1)[:,None]
    return vectors / norm

def reduce_dimension(vectors, method, pre_normalization, target_dim, params={}):
    """
    :type vectors: numpy.ndarray(shape=(N, dim), dtype=float)
    :type method: str
    :type pre_normalization: bool
    :type target_dim: int
    :type params: {str: Any}
    :rtype: numpy.ndarray(shape=(N, target_dim), dtype=float)
    """
    # Normalization
    if pre_normalization:
        vectors = normalize(vectors)

    # Dimension reduction
    if method == "pca":
        model = PCA(target_dim)
    elif method == "tsne":
        model = TSNE(target_dim, perplexity=params["perplexity"])
    elif method == "spectral":
        model = SpectralEmbedding(target_dim)
    else:
        raise ValueError("Unknown method=%s" % method)

    vectors = model.forward(vectors)

    return vectors

