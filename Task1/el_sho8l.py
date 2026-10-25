import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from models import NN, confusion_matrix, accuracy

FEATURES = ["CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation", "BodyMass"]
CSV = Path(__file__).resolve().parent / "data" / "penguins.csv"


def load_data(path=CSV):
    df = pd.read_csv(path, na_values=["NA"])
    num_cols = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
    df[num_cols] = df.groupby("Species")[num_cols].transform(lambda c: c.fillna(c.median()))
    df["OriginLocation"] = df["OriginLocation"].astype("category").cat.codes   
    
    return df


def split(df, class1, class2, f1, f2, n_train=30):
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for cls, label in ((class1, 1), (class2, -1)):
        X = df[df["Species"] == cls][[f1, f2]].to_numpy(dtype=float)
        idx = np.random.permutation(len(X))
        X_tr.append(X[idx[:n_train]]);  y_tr += [label] * n_train
        X_te.append(X[idx[n_train:]]);  y_te += [label] * (len(X) - n_train)
    return np.vstack(X_tr), np.array(y_tr), np.vstack(X_te), np.array(y_te)


def standardize(X_train, X_test):
    """Scale both sets using the TRAIN mean/std only. Returns scaled sets + mean + std."""
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


if __name__ == "__main__":
    df = load_data()
    species = ["Adelie", "Chinstrap", "Gentoo"]
    results = []
    for c1, c2 in itertools.combinations(species, 2):
        for f1, f2 in itertools.combinations(FEATURES, 2):
            X_tr, y_tr, X_te, y_te = split(df, c1, c2, f1, f2)
            X_tr, X_te, _, _ = standardize(X_tr, X_te)
            for name in ("perceptron", "adaline"):
                model = NN(lr=0.01, epochs=200)
                getattr(model, "perc" if name == "perceptron" else "adaline")(X_tr, y_tr)
                acc = accuracy(confusion_matrix(y_te, model.predict(X_te)))
                results.append((acc, name, c1, c2, f1, f2))
    for acc, name, c1, c2, f1, f2 in sorted(results, reverse=True):
        print(f"{acc:6.1%}  {name:<10} {c1} vs {c2:<10} {f1} + {f2}")
