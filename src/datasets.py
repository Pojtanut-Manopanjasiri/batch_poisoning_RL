import argparse
import gzip
import urllib.request
from pathlib import Path

import numpy as np


def detpep8(X):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    cs = np.cumsum(X[:, 2:8], axis=1)[:, 1:]                   # x3+x4, ..., x3+...+x8
    outer = (np.arange(4, 9) * np.log1p(cs)).sum(1)
    return (4 * (x1 - 2 + 8 * x2 - 8 * x2**2) ** 2 + (3 - 4 * x2) ** 2
            + 16 * np.sqrt(x3 + 1) * (2 * x3 - 1) ** 2 + outer)


_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H6_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_H6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])


def hart6sc(X):
    # Picheny et al. (2012) rescaling, as on the SFU page (its R files swap hart6/hart6sc)
    inner = (_H6_A[None] * (X[:, None, :] - _H6_P[None]) ** 2).sum(2)
    return -(2.58 + (_H6_ALPHA * np.exp(-inner)).sum(1)) / 1.94


def borehole(X):
    rw, r, Tu, Hu, Tl, Hl, L, Kw = X.T
    lr = np.log(r / rw)
    return 2 * np.pi * Tu * (Hu - Hl) / (lr * (1 + 2 * L * Tu / (lr * rw**2 * Kw) + Tu / Tl))


def otlcircuit(X):
    Rb1, Rb2, Rf, Rc1, Rc2, beta = X.T
    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    den = beta * (Rc2 + 9) + Rf
    return ((Vb1 + 0.74) * beta * (Rc2 + 9) / den + 11.35 * Rf / den
            + 0.74 * Rf * beta * (Rc2 + 9) / (den * Rc1))


def piston(X):
    M, S, V0, k, P0, Ta, T0 = X.T
    A = P0 * S + 19.62 * M - k * V0 / S
    V = S / (2 * k) * (np.sqrt(A**2 + 4 * k * (P0 * V0 / T0) * Ta) - A)
    return 2 * np.pi * np.sqrt(M / (k + S**2 * (P0 * V0 / T0) * (Ta / V**2)))


def robot(X):
    ang = np.cumsum(X[:, :4], axis=1)                          # theta1, theta1+theta2, ...
    L = X[:, 4:8]
    return np.hypot((L * np.cos(ang)).sum(1), (L * np.sin(ang)).sum(1))


def wingweight(X):
    Sw, Wfw, A, Lam, q, lam, tc, Nz, Wdg, Wp = X.T
    c = np.cos(np.deg2rad(Lam))
    return (0.036 * Sw**0.758 * Wfw**0.0035 * (A / c**2) ** 0.6 * q**0.006 * lam**0.04
            * (100 * tc / c) ** -0.3 * (Nz * Wdg) ** 0.49 + Sw * Wp)


def friedman(X):
    x1, x2, x3, x4, x5 = X.T
    return 10 * np.sin(np.pi * x1 * x2) + 20 * (x3 - 0.5) ** 2 + 10 * x4 + 5 * x5


def ishigami(X, a=7.0, b=0.1):
    x1, x2, x3 = X.T
    return np.sin(x1) + a * np.sin(x2) ** 2 + b * x3**4 * np.sin(x1)


def detpep10curv(X):
    x1, x2, x3 = X.T
    return (4 * (x1 - 2 + 8 * x2 - 8 * x2**2) ** 2 + (3 - 4 * x2) ** 2
            + 16 * np.sqrt(x3 + 1) * (2 * x3 - 1) ** 2)


def detpep10exp(X):
    x1, x2, x3 = X.T
    return 100 * (np.exp(-2 / x1**1.75) + np.exp(-2 / x2**1.5) + np.exp(-2 / x3**1.25))


def welch(X):
    x = {i + 1: X[:, i] for i in range(20)}
    return (5 * x[12] / (1 + x[1]) + 5 * (x[4] - x[20]) ** 2 + x[5] + 40 * x[19] ** 3 - 5 * x[19]
            + 0.05 * x[2] + 0.08 * x[3] - 0.03 * x[6] + 0.03 * x[7] - 0.09 * x[9] - 0.01 * x[10]
            - 0.07 * x[11] + 0.25 * x[13] ** 2 - 0.04 * x[14] + 0.06 * x[15] - 0.01 * x[17]
            - 0.03 * x[18])


# name -> (function, [(input, low, high), ...]); domains from the SFU pages
FUNCTIONS = {
    "detpep8": (detpep8, [(f"x{i}", 0.0, 1.0) for i in range(1, 9)]),
    "hart6sc": (hart6sc, [(f"x{i}", 0.0, 1.0) for i in range(1, 7)]),
    "borehole": (borehole, [("rw", 0.05, 0.15), ("r", 100, 50000), ("Tu", 63070, 115600),
                            ("Hu", 990, 1110), ("Tl", 63.1, 116), ("Hl", 700, 820),
                            ("L", 1120, 1680), ("Kw", 9855, 12045)]),
    "otlcircuit": (otlcircuit, [("Rb1", 50, 150), ("Rb2", 25, 70), ("Rf", 0.5, 3),
                                ("Rc1", 1.2, 2.5), ("Rc2", 0.25, 1.2), ("beta", 50, 300)]),
    "piston": (piston, [("M", 30, 60), ("S", 0.005, 0.020), ("V0", 0.002, 0.010), ("k", 1000, 5000),
                        ("P0", 90000, 110000), ("Ta", 290, 296), ("T0", 340, 360)]),
    "robot": (robot, [(f"theta{i}", 0.0, 2 * np.pi) for i in range(1, 5)]
              + [(f"L{i}", 0.0, 1.0) for i in range(1, 5)]),
    "wingweight": (wingweight, [("Sw", 150, 200), ("Wfw", 220, 300), ("A", 6, 10),
                                ("Lambda", -10, 10), ("q", 16, 45), ("lambda", 0.5, 1),
                                ("tc", 0.08, 0.18), ("Nz", 2.5, 6), ("Wdg", 1700, 2500),
                                ("Wp", 0.025, 0.08)]),
    "friedman": (friedman, [(f"x{i}", 0.0, 1.0) for i in range(1, 6)]),
    "ishigami": (ishigami, [(f"x{i}", -np.pi, np.pi) for i in range(1, 4)]),
    "detpep10curv": (detpep10curv, [(f"x{i}", 0.0, 1.0) for i in range(1, 4)]),
    "detpep10exp": (detpep10exp, [(f"x{i}", 0.0, 1.0) for i in range(1, 4)]),
    "welch": (welch, [(f"x{i}", -0.5, 0.5) for i in range(1, 21)]),
}


def sample_unit(n, d, rng, design="uniform"):
    if design == "lhs":                                        # one point per stratum per axis
        u = (np.arange(n)[:, None] + rng.random((n, d))) / n
        return np.take_along_axis(u, rng.random((n, d)).argsort(0), axis=0)
    return rng.random((n, d))


def generate(name, n=10000, noise=0.0, seed=0, design="uniform"):
    """(X in original units, X scaled to [0,1], y); noise = Gaussian std as a fraction of std(y)."""
    fn, dom = FUNCTIONS[name]
    lo, hi = np.array([d[1] for d in dom], float), np.array([d[2] for d in dom], float)
    rng = np.random.default_rng(seed)
    U = sample_unit(n, len(dom), rng, design)
    X = lo + U * (hi - lo)
    y = fn(X)
    if noise > 0:
        y = y + noise * y.std() * rng.standard_normal(n)
    return X, U, y


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


def load_mnist(digit, n=10000, seed=0, pca=30):
    """(X, y) for n training images: y = 1 if the image shows `digit`, else 0; X on the top `pca` components."""
    images = _mnist_file("train-images-idx3-ubyte.gz", 16).reshape(-1, 784) / 255.0
    labels = _mnist_file("train-labels-idx1-ubyte.gz", 8)
    idx = np.random.default_rng(seed).choice(len(labels), n, replace=False)
    X, y = images[idx] - images[idx].mean(0), (labels[idx] == digit).astype(float)
    if pca is None:
        return X[:, X.std(0) > 0], y                   # raw pixels, constant border pixels dropped
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:pca].T, y


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


def dataset_size(name, n=10000):
    """Points in the pool: n, capped at the number of rows of a UCI dataset."""
    return min(n, UCI[name][1]) if name in UCI else n


def _raw(name, n, noise, seed, design, pca):
    """(X, y) before standardisation, for a UCI dataset, 'mnist<k>' or an SFU function."""
    digit = mnist_digit(name)
    if name in FUNCTIONS:
        X, _, y = generate(name, n, noise, seed, design)
        return X, y
    rng = np.random.default_rng(seed)
    if digit is not None:
        X, y = load_mnist(digit, n, seed, pca)
    else:
        X, y = UCI[name][0]()
        if n < len(y):                                         # subsample a smaller pool
            idx = rng.choice(len(y), n, replace=False)
            X, y = X[idx], y[idx]
    if noise > 0:
        y = y + noise * y.std() * rng.standard_normal(len(y))
    return X, y


def n_features(name, pca=30):
    if name in UCI:
        return UCI[name][2]
    if mnist_digit(name) is None:
        return len(FUNCTIONS[name][1])
    return pca if pca is not None else load_dataset(name, pca=None)[0].shape[1]


def load_dataset(name, n=10000, noise=0.0, seed=0, design="uniform", standardize_y=True, pca=30):
    """(X, y) with every feature at mean 0, variance 1 over the n points (y too, by default)."""
    X, y = _raw(name, n, noise, seed, design, pca)
    X = (X - X.mean(0)) / X.std(0)
    if standardize_y:
        y = (y - y.mean()) / y.std()
    return X, y


def save_csv(name, X, y, path):
    cols = ([d[0] for d in FUNCTIONS[name][1]] if name in FUNCTIONS
            else [f"{'pc' if mnist_digit(name) is not None else 'x'}{i + 1}"
                  for i in range(X.shape[1])]) + ["y"]
    np.savetxt(path, np.column_stack([X, y]), delimiter=",", header=",".join(cols), comments="")


def main():
    p = argparse.ArgumentParser(
        description="Write victim datasets (UCI, MNIST one-vs-rest, SFU test functions) as CSVs.",
        epilog="e.g. python src/datasets.py --names wine airfoil breast_cancer mnist1")
    p.add_argument("--names", nargs="+", default=["all"],
                   help=f"any of {list(UCI)}, mnist0..mnist9 (digit vs rest), {list(FUNCTIONS)}, "
                        "or all (SFU only)")
    p.add_argument("--pca", type=int, default=30, help="mnist: principal components kept")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--noise", type=float, default=0.0, help="noise std as a fraction of std(y)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--design", choices=["uniform", "lhs"], default="uniform")
    p.add_argument("--raw", action="store_true", help="original units instead of standardised")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "notebooks" / "data"))
    a = p.parse_args()
    names = list(FUNCTIONS) if a.names == ["all"] else a.names
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        if a.raw:
            X, y = _raw(name, a.n, a.noise, a.seed, a.design, a.pca)
        else:
            X, y = load_dataset(name, a.n, a.noise, a.seed, a.design, pca=a.pca)
        tag = "raw" if a.raw else "std"
        path = out / f"{name}_n{a.n}_noise{a.noise:g}_seed{a.seed}_{a.design}_{tag}.csv"
        save_csv(name, X, y, path)
        print(f"{name:13} d={X.shape[1]:2}  y mean={y.mean():10.4g} std={y.std():10.4g}  -> {path}")


if __name__ == "__main__":
    main()
