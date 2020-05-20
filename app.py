from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from utils.service import Exps

app = Flask(__name__)
api = Api(app)

exp = Exps()
exp.load_params('weights/glove_model_50.npz', 'embeddings/w2i.json')


class TopNeighs(Resource):
    def post(self):
        posted_data = request.get_json()
        assert 'top_k' in posted_data
        assert 'word' in posted_data

        neighs = exp.top_neigs(posted_data['word'], posted_data['top_k'])
        return jsonify(neighs)


class Analogy(Resource):

    def post(self):
        posted_data = request.get_json()

        assert 'pos1' in posted_data
        assert 'neg1' in posted_data
        assert 'pos2' in posted_data

        rets = exp.analogy(posted_data['pos1'], posted_data['neg1'], posted_data['pos2'])
        return jsonify(rets)


api.add_resource(TopNeighs, '/neighs')
api.add_resource(Analogy, '/analogy')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
