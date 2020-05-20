from models.embeddings import WikipediaEmbedding
from models.nn import Glove
from utils.exps import Exps
from utils.network.glove import dump_weights, dump_word_encoding
from utils.params import parse_args

if __name__ == '__main__':

    args = parse_args()

    if args.load:

        exp = Exps()
        exp.load_params('weights/glove_model_50.npz', 'embeddings/w2i.json')

        rets_analogy = exp.analogy('king', 'man', 'woman')

        rets_neighs = exp.top_neigs('king')

        print(f'analogy: {rets_analogy} \n neighs: {rets_neighs}')

    else:

        wiki_emb = WikipediaEmbedding(vocab_size=2000)

        wiki_emb.build()
        dump_word_encoding('w2i', wiki_emb.word2idx)

        glove_model = Glove(embedding=wiki_emb, hidden_dim=100, n_gram=10)

        glove_model.build()
        glove_model.train(epochs=20)

        dump_weights('glove_model_50', glove_model.W, glove_model.U)
