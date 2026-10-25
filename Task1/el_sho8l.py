import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from models import NN, confusion_matrix, accuracy

FEATURES = ["CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation", "BodyMass"]
penguins = Path(__file__).resolve().parent / "data" / "penguins.csv"


def load_preprocess(path=penguins):
    df = pd.read_csv(path)

    num_cols = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]

    ##nulls
    #fill w median of each soecie
    df[num_cols] = df.groupby("Species")[num_cols].transform(lambda c: c.fillna(c.median()))

    #string encoding
    df["OriginLocation"] = df["OriginLocation"].astype("category").cat.codes   
    
    return df

#split in each class 
def split(df, class1, class2, f1, f2, n_train=30):

    X_tr, y_tr, X_te, y_te = [], [], [], []

    for cls, label in ((class1, 1), (class2, -1)):
        X = df[df["Species"] == cls][[f1, f2]].to_numpy(dtype=float)
        
        idx = np.random.permutation(len(X))
        X_tr.append(X[idx[:n_train]]);  y_tr += [label] * n_train
        X_te.append(X[idx[n_train:]]);  y_te += [label] * (len(X) - n_train)
    return np.vstack(X_tr), np.array(y_tr), np.vstack(X_te), np.array(y_te)


def standardize(X_train, X_test):
    #scale both train, test using TRAIN mean/std bas 
    mean= X_train.mean(axis=0)
    std =  X_train.std(axis=0)

    std[std == 0] = 1.0

    return (X_train - mean) / std, (X_test - mean) / std, mean, std

#the 5 combos used in the report (class1, class2, feature1, feature2)
REPORT_COMBOS = [
    ("Adelie",    "Gentoo",    "CulmenLength",  "FlipperLength"),
    ("Chinstrap", "Gentoo",    "CulmenDepth",   "OriginLocation"),
    ("Adelie",    "Chinstrap", "CulmenLength",  "CulmenDepth"),
    ("Adelie",    "Chinstrap", "FlipperLength", "BodyMass"),
    ("Adelie",    "Chinstrap", "CulmenDepth",   "OriginLocation"),
]


def train_and_test(df, algo, c1, c2, f1, f2, lr=0.01, epochs=100, threshold=0.1, bias=True, seed=None):
    if seed is not None:
        np.random.seed(seed)   #same seed = same split for both algorithms
    X_tr, y_tr, X_te, y_te = split(df, c1, c2, f1, f2)
    X_tr, X_te, _, _ = standardize(X_tr, X_te)
    model = NN(lr=lr, epochs=epochs, bias=bias, threshold=threshold)
    getattr(model, "perc" if algo == "perceptron" else "adaline")(X_tr, y_tr)
    return confusion_matrix(y_te, model.predict(X_te))


def report_confusion_matrices(df, seed=0):
    from matplotlib.figure import Figure   #no pyplot, so no pop-up windows
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)

    for n, (c1, c2, f1, f2) in enumerate(REPORT_COMBOS, start=1):
        fig = Figure(figsize=(9, 4))
        axes = fig.subplots(1, 2)

        for ax, algo in zip(axes, ("perceptron", "adaline")):
            cm = train_and_test(df, algo, c1, c2, f1, f2, seed=seed)

            #image
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1).max())
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels([c1, c2]); ax.set_yticklabels([c1, c2])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"{algo.title()} - accuracy {accuracy(cm):.1%}")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")

            #text version (copy this and send it to me)
            print(f"#{n} {algo.upper()} | {c1} vs {c2} | {f1} + {f2}")
            print(f"{'':>12}{c1:>12}{c2:>12}   (rows = actual, columns = predicted)")
            print(f"{c1:>12}{cm[0, 0]:>12}{cm[0, 1]:>12}")
            print(f"{c2:>12}{cm[1, 0]:>12}{cm[1, 1]:>12}")
            print(f"Accuracy: {accuracy(cm):.1%}\n")

        fig.suptitle(f"{c1} vs {c2}: {f1} + {f2}")
        fig.tight_layout()
        path = out / f"cm_{n}_{c1}_{c2}_{f1}_{f2}.png"
        fig.savefig(path, dpi=200)
        print(f"Saved: {path}\n")


def report_averaged_accuracy(df, runs=20):
    #mean +- std over many random splits, for the 5 report combos
    print("Averaged accuracy over", runs, "random splits")
    for c1, c2, f1, f2 in REPORT_COMBOS:
        line = f"{c1} vs {c2} | {f1} + {f2}: "
        for algo in ("perceptron", "adaline"):
            accs = [accuracy(train_and_test(df, algo, c1, c2, f1, f2, seed=s)) for s in range(runs)]
            line += f"{algo} {np.mean(accs):.1%} +- {np.std(accs):.1%}   "
        print(line)


if __name__ == "__main__":
    df = load_preprocess()

    species = ["Adelie", "Chinstrap", "Gentoo"]
    results = []

    #each comb of class 
    for c1, c2 in itertools.combinations(species, 2):
        #each comb of feats
        for f1, f2 in itertools.combinations(FEATURES, 2):
            X_tr, y_tr, X_te, y_te = split(df, c1, c2, f1, f2)
            X_tr, X_te, _, _ = standardize(X_tr, X_te)

            for name in ("perceptron", "adaline"):
                model = NN(lr=0.01, epochs=200)

                getattr(model, "perc" if name == "perceptron" else "adaline")(X_tr, y_tr)

                acc = accuracy(confusion_matrix(y_te, model.predict(X_te)))

                results.append((acc, name, c1, c2, f1, f2))

    #printing bas             
    for acc, name, c1, c2, f1, f2 in sorted(results, reverse=True):
        print(f"{acc:6.1%}  {name:<10} {c1} vs {c2:<10} {f1} + {f2}")
    
    report_confusion_matrices(df)
    report_averaged_accuracy(df)
