import requests

url_neighs = 'http://0.0.0.0:5000/neighs'
url_analogy = 'http://0.0.0.0:5000/analogy'

neighs_data = {
    'top_k': 4,
    'word': 'king'
}
analogy_data = {
    'pos1': 'king',
    'neg1': 'man',
    'pos2': 'woman',
}

if __name__ == '__main__':
    r = requests.post(url_neighs, json=neighs_data)
    print(r.text)

    r = requests.post(url_analogy, json=analogy_data)
    print(r.text)
