import argparse
import gzip
import urllib.request
from pathlib import Path

import numpy as np


MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"          # torchvision's mirror
MNIST_CACHE = Path(__file__).resolve().parents[1] / "data" / "mnist"


def mnist_digit(name):
    """'mnist7' -> 7 (digit 7 vs the rest); None for any other name."""
    return int(name[5:]) if name.startswith("mnist") and name[5:].isdigit() else None


def _mnist_file(name, offset):
    path = MNIST_CACHE / name
    if not path.exists():                                      # downloaded once, then cached
        MNIST_CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MNIST_URL + name, path)
    with gzip.open(path) as f:
        return np.frombuffer(f.read(), np.uint8, offset=offset)


def load_mnist(digit, n=10000, seed=0, pca=30, test=False):
    """(X, y) for n training images: y = 1 if the image shows `digit`, else 0; X on the top `pca` components."""
    images = _mnist_file("train-images-idx3-ubyte.gz", 16).reshape(-1, 784) / 255.0
    labels = _mnist_file("train-labels-idx1-ubyte.gz", 8)
    idx = np.random.default_rng(seed).choice(len(labels), n, replace=False)
    mean = images[idx].mean(0)
    X, y = images[idx] - mean, (labels[idx] == digit).astype(float)
    if pca is None:
        keep = X.std(0) > 0                            # raw pixels, constant border pixels dropped
        proj = lambda A: A[:, keep]
    else:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        proj = lambda A: A @ Vt[:pca].T
    if not test:
        return proj(X), y
    Xt = _mnist_file("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 784) / 255.0    # test=True: + the 10k test images
    yt = (_mnist_file("t10k-labels-idx1-ubyte.gz", 8) == digit).astype(float)
    return proj(X), y, proj(Xt - mean), yt


UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
UCI_CACHE = Path(__file__).resolve().parents[1] / "data" / "uci"


def _uci_lines(path):
    local = UCI_CACHE / Path(path).name
    if not local.exists():                                     # downloaded once, then cached
        UCI_CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(UCI_URL + path, local)
    return local.read_text().strip().splitlines()


def _wine():
    A = np.array([l.split(";") for l in _uci_lines("wine-quality/winequality-red.csv")[1:]], float)
    return A[:, :-1], A[:, -1]                                  # quality score (0-10)


def _airfoil():
    A = np.array([l.split() for l in _uci_lines("00291/airfoil_self_noise.dat")], float)
    return A[:, :-1], A[:, -1]                                  # sound pressure level (dB)


def _breast_cancer():
    rows = [l.split(",") for l in _uci_lines("breast-cancer-wisconsin/wdbc.data")]
    X = np.array([r[2:] for r in rows], float)                  # 30 features; r[0] is the id
    return X, np.array([r[1] == "M" for r in rows], float)      # malignant = 1, benign = 0


# name -> (loader, rows, features, binary labels)
UCI = {"wine": (_wine, 1599, 11, False), "airfoil": (_airfoil, 1503, 5, False),
       "breast_cancer": (_breast_cancer, 569, 30, True)}


def binary_labels(name):
    """True when y is a 0/1 class label (logistic victim), False for a real-valued target."""
    return mnist_digit(name) is not None or (name in UCI and UCI[name][3])


def dataset_size(name, n=10000, test=0.0):
    """Points in the pool: n, capped at the rows a UCI dataset has left after holding out the test share."""
    return min(n, UCI[name][1] - round(test * UCI[name][1])) if name in UCI else n


def _raw(name, n, noise, seed, pca, test=0.0):
    """(X, y, X_test, y_test) before standardisation, for a UCI dataset or 'mnist<k>'."""
    digit = mnist_digit(name)
    rng = np.random.default_rng(seed)
    if digit is not None:
        if test:                                               # any test > 0: MNIST's own 10k test images
            X, y, Xt, yt = load_mnist(digit, n, seed, pca, test=True)
        else:
            X, y = load_mnist(digit, n, seed, pca)
            Xt, yt = X[:0], y[:0]
    else:
        A, b = UCI[name][0]()
        held = rng.choice(len(b), round(test * len(b)), replace=False) if test else np.arange(0)
        pool = np.setdiff1d(np.arange(len(b)), held)
        if n < len(pool):                                      # subsample a smaller pool
            pool = pool[rng.choice(len(pool), n, replace=False)]
        X, y, Xt, yt = A[pool], b[pool], A[held], b[held]
    if noise > 0:                                              # pool labels only
        y = y + noise * y.std() * rng.standard_normal(len(y))
    return X, y, Xt, yt


def n_features(name, pca=30):
    if name in UCI:
        return UCI[name][2]
    if mnist_digit(name) is None:
        raise ValueError(f"unknown dataset {name!r}: expected one of {list(UCI)} or mnist0..mnist9")
    return pca if pca is not None else load_dataset(name, pca=None)[0].shape[1]


def load_split(name, n=10000, noise=0.0, seed=0, pca=30, test=0.2, standardize_y=True):
    """(X, y, X_test, y_test): pool and held-out points, both scaled with the pool's mean and std."""
    X, y, Xt, yt = _raw(name, n, noise, seed, pca, test)
    mu, sd = X.mean(0), X.std(0)
    X, Xt = (X - mu) / sd, (Xt - mu) / sd
    if standardize_y:
        m, s = y.mean(), y.std()
        y, yt = (y - m) / s, (yt - m) / s
    return X, y, Xt, yt


def load_dataset(name, n=10000, noise=0.0, seed=0, standardize_y=True, pca=30):
    """(X, y) with every feature at mean 0, variance 1 over the n points (y too, by default)."""
    return load_split(name, n, noise, seed, pca, 0.0, standardize_y)[:2]


def save_csv(name, X, y, path):
    cols = [f"{'pc' if mnist_digit(name) is not None else 'x'}{i + 1}" for i in range(X.shape[1])] + ["y"]
    np.savetxt(path, np.column_stack([X, y]), delimiter=",", header=",".join(cols), comments="")


def main():
    p = argparse.ArgumentParser(
        description="Write victim datasets (UCI, MNIST one-vs-rest) as CSVs.",
        epilog="e.g. python src/datasets.py --names wine mnist3")
    p.add_argument("--names", nargs="+", default=[*UCI, "mnist1"],
                   help=f"any of {list(UCI)}, mnist0..mnist9 (digit vs rest)")
    p.add_argument("--pca", type=int, default=30, help="mnist: principal components kept")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--noise", type=float, default=0.0, help="noise std as a fraction of std(y)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw", action="store_true", help="original units instead of standardised")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "notebooks" / "data"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.names:
        if a.raw:
            X, y = _raw(name, a.n, a.noise, a.seed, a.pca)[:2]
        else:
            X, y = load_dataset(name, a.n, a.noise, a.seed, pca=a.pca)
        tag = "raw" if a.raw else "std"
        path = out / f"{name}_n{a.n}_noise{a.noise:g}_seed{a.seed}_{tag}.csv"
        save_csv(name, X, y, path)
        print(f"{name:13} d={X.shape[1]:2}  y mean={y.mean():10.4g} std={y.std():10.4g}  -> {path}")


if __name__ == "__main__":
    main()
