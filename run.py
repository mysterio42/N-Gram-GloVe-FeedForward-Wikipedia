from models.embeddings import WikipediaEmbedding
from models.nn import Glove
from utils.network.glove import dump_weights

if __name__ == '__main__':
    wiki_emb = WikipediaEmbedding(vocab_size=2000)
    wiki_emb.build()

    glove_model = Glove(embedding=wiki_emb, hidden_dim=50, n_gram=10)
    glove_model.build()
    glove_model.train(epochs=20)
    dump_weights('glove_model_50', glove_model.W, glove_model.U)
