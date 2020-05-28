from models.embeddings import WikipediaEmbedding
from models.nn import Glove
from utils.data import to_csv
from utils.network.glove import dump_word_encoding
from utils.params import parse_args
from utils.service import Exps

if __name__ == '__main__':

    args = parse_args()

    if args.load:

        exp = Exps()
        # exp.load_params('weights/glove_model_50.npz', 'embeddings/w2i.json')
        exp.load_pmi_params('weights/pmi_ALS.npz', 'embeddings/w2i (1).json')

        words = ['japan', 'japanese', 'england', 'english', 'australia', 'australian', 'china', 'chinese', 'italy',
                 'italian', 'french', 'france', 'spain', 'spanish']
        exp.visualize_countries_pmi(words)
        word = 'king'

        # rets_analogy = exp.analogy(word, 'man', 'woman')
        rets_analogy = exp.analogy_pmi(word, 'man', 'woman')

        rets_neighs = exp.top_neigs_pmi(word)

        to_csv(rets_neighs)

        print(f'analogy: {rets_analogy} \n neighs: {rets_neighs}')

    else:

        wiki_emb = WikipediaEmbedding(vocab_size=2000)

        wiki_emb.build()
        dump_word_encoding('w2i', wiki_emb.word2idx)

        glove_model = Glove(embedding=wiki_emb, hidden_dim=100, n_gram=10)

        # glove_model.build()
        # glove_model.train(epochs=20)
        # glove_model.build_train_SVD()
        glove_model.build_train_pmi()