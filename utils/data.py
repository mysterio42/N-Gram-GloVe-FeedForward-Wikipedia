import pandas as pd

PREDS_DIR = 'preds/'


def to_csv(name, last_word, ret_neighs):
    with open(PREDS_DIR + name + '.txt', 'w') as f:
        f.write(pd.DataFrame([ret_neighs], index=[last_word])
                .to_markdown()
                )
