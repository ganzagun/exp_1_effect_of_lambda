from ..walker import RandomWalker
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm


class TqdmCallback(CallbackAny2Vec):
    """Callback to show progress bar for Word2Vec training"""
    def __init__(self, total_epochs):
        self.epoch = 0
        self.total_epochs = total_epochs
        self.pbar = None
    
    def on_train_begin(self, model):
        self.pbar = tqdm(total=self.total_epochs, desc="Training Word2Vec", leave=True)
    
    def on_epoch_end(self, model):
        self.epoch += 1
        self.pbar.update(1)
    
    def on_train_end(self, model):
        self.pbar.close()


class DeepWalk:
    def __init__(self, graph, walk_length=80, num_walks=10, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = RandomWalker(
            graph, p=1, q=1, )
        print(f"\nInitializing DeepWalk with {len(graph.nodes())} nodes")
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=0)
        print(f"Generated {len(self.sentences)} random walks")

    def train(self, embed_size=128, window_size=5, workers=1, iter=3, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size  # vector_size, embed_size, 1433
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter
        
        # Add callback for progress tracking
        if "callbacks" not in kwargs:
            kwargs["callbacks"] = [TqdmCallback(iter)]

        print(f"\nTraining DeepWalk embeddings (size={embed_size}, window={window_size}, epochs={iter})")
        model = Word2Vec(**kwargs)

        self.w2v_model = model
        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings
