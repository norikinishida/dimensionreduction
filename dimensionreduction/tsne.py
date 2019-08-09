import sklearn.manifold

class TSNE(object):

    def __init__(self, target_dim, perplexity):
        self.target_dim = target_dim
        self.perplexity = perplexity
        self.model = sklearn.manifold.TSNE(n_components=target_dim,
                                           perplexity=perplexity,
                                           random_state=0)

    def forward(self, vectors):
        """
        :type vectors: numpy.ndarray(shape=(N,dim), dtype=float)
        :rtype: numpy.ndarray(shape=(N,target_dim), dtype=float)
        """
        vectors = self.model.fit_transform(vectors)
        return vectors
