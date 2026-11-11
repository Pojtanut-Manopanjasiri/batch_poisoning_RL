import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

import experiment as ex
from env import Environment

ABOUT = ("pred[policy][k]: predictions on the test points of the victim model left after test episode episodes[k] "
         "(initial, target, star: the victim's start, the attack target and theta*, one row each); prob: P(label = 1), "
         "logistic only; features and labels as the victim sees them (standardised for real datasets)")


def pickles(paths):
    """Merged results pickles: given directly, or the top-level *.pkl of a run directory."""
    out = []
    for p in map(Path, paths):
        out += sorted(p.glob("*.pkl")) if p.is_dir() else [p]
    return out


def test_set_name(env, ecfg):
    if not ecfg.dataset:
        return f"{ex.TEST_POINTS} fresh synthetic draws (seed {ex.TEST_SEED})"
    if not len(env.stream.y_test):
        return f"{ecfg.dataset}: the training pool (dataset_test=0)"
    held = "MNIST test images" if ecfg.dataset.startswith("mnist") else f"{ecfg.dataset_test:.0%} held-out rows"
    return f"{ecfg.dataset}: {held}"


def rounded(a):
    return np.round(np.asarray(a, dtype=float), 6).tolist()     # 6 decimals keep the files small


def export(path, episodes, out_dir):
    with open(path, "rb") as f:
        res = pickle.load(f)
    if "victim" not in res["runs"][0]:
        sys.exit(f"{path}: no saved victim models (results from before they were added); rerun")
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)        # the same float mode the run used
    env = Environment(res["ecfg"], res["cfg"])
    X, y = ex.test_set(env)
    logistic = env.victim.task == "logistic"
    k = res["eval_episodes"] if episodes == "all" else min(int(episodes), res["eval_episodes"])
    folder = Path(out_dir or path.parent) / f"{path.stem}_predictions"
    folder.mkdir(parents=True, exist_ok=True)
    for seed, run in zip(res["seeds"], res["runs"]):
        pred, prob = {}, {}
        for n, st in run["victim"].items():
            th = torch.as_tensor(st["theta"][:k], device=env.device)
            check = ex.victim_stats(env, th, (X, y))       # the rebuilt test set must be the one that was scored
            for key, v in check.items():
                if key != "theta" and not np.allclose(v, st[key][:k] if np.ndim(st[key]) else st[key], rtol=1e-6, atol=0):
                    sys.exit(f"{path} seed {seed} {n}: {key} differs from the saved scores; test set not reproduced")
            o = torch.stack([ex.victim_output(env, t, X) for t in th]).detach().cpu()
            if logistic:
                pred[n], prob[n] = (o > 0).int().tolist(), rounded(torch.sigmoid(o))
            else:
                pred[n] = rounded(o)
        doc = {"victim": res["ecfg"].dataset or res["ecfg"].task, "task": env.victim.task, "seed": seed,
               "result_file": path.name, "test_set": test_set_name(env, res["ecfg"]), "n_points": len(y),
               "episodes": list(range(k)), "about": ABOUT, "features": rounded(X.cpu()),
               "label": y.int().tolist() if logistic else rounded(y.cpu()), "pred": pred}
        if logistic:
            doc["prob"] = prob
        dest = folder / f"seed{seed}.json"
        tmp = dest.with_name(dest.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, dest)
        print(f"{dest}  ({dest.stat().st_size / 2**20:.1f} MB, {len(y)} points, {len(pred)} models x {k} episode(s))")


def main():
    p = argparse.ArgumentParser(description="Write each seed's test points, true labels and the victim models' "
                                            "predictions as JSON, from merged results pickles.")
    p.add_argument("results", nargs="+", help="merged results pickles, or run directories holding them")
    p.add_argument("--episodes", default="1", help="first K test episodes' final models per policy, or all")
    p.add_argument("--out", help="parent folder for <pickle>_predictions/ (default: next to each pickle)")
    a = p.parse_args()
    if a.episodes != "all" and not (a.episodes.isdigit() and int(a.episodes) > 0):
        sys.exit(f"--episodes {a.episodes}: expected a positive integer or all")
    for path in pickles(a.results):
        export(path, a.episodes, a.out)


if __name__ == "__main__":
    main()
