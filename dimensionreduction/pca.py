from sklearn import decomposition

class PCA(object):

    def __init__(self, target_dim):
        self.target_dim = target_dim
        self.model = decomposition.PCA(n_components=target_dim)

    def forward(self, vectors):
        """
        :type vectors: numpy.ndarray(shape=(N,dim), dtype=float)
        :rtype: numpy.ndarray(shape=(N,target_dim), dtype=float)
        """
        self.model.fit(vectors)
        vectors = self.model.transform(vectors)
        return vectors
