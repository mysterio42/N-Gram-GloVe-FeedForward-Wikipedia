import json
import os

import pandas as pd

EMBEDDING_DIR = 'embedding/'
PREDS_DIR = 'preds/'


def to_json(data, name):
    """
    Dump the json


    :param data: Bi-Gram json
    :param name: file name
    :return:
    """
    path = EMBEDDING_DIR + name + '.json'
    if os.path.isfile(path):
        print(f'data saved at {path} already exist')
        return

    with open(path, 'w') as f:
        json.dump(data, f)
    print(f'data saved at {path}')


def to_csv(name, last_word, ret_neighs):
    with open(PREDS_DIR + name + '.txt', 'w') as f:
        f.write(pd.DataFrame([ret_neighs], index=[last_word])
                .to_markdown()
                )
