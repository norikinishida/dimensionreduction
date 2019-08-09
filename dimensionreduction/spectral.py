from sklearn import manifold

class SpectralEmbedding(object):

    def __init__(self, target_dim):
        self.target_dim = target_dim
        self.model = manifold.SpectralEmbedding(n_components=target_dim)

    def forward(self, vectors):
        """
        :type vectors: numpy.ndarray(shape=(N,dim), dtype=float)
        :rtype: numpy.ndarray(shape=(N,target_dim), dtype=float)
        """
        vectors = self.model.fit_transform(vectors)
        return vectors
