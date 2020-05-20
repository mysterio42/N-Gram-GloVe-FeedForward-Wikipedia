import pandas as pd

from utils.network.glove import generate_model_name

PREDS_DIR = 'preds/'


def to_csv(ret_neighs):
    rets = ret_neighs['method']
    for meth, solution in rets.items():
        path = PREDS_DIR + solution['word'] + '-' + 'top_neighs' + '-' + meth + '-' + generate_model_name(3) + '.txt'
        with open(path, 'w') as f:
            f.write(pd.DataFrame([solution['neigh_dist']], index=[solution['word']])
                    .to_markdown()
                    )
