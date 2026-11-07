import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import experiment as ex
from runs_io import load_records


def merge(recs):
    """Per-(seed, learner) records of one victim -> the notebooks' results dict."""
    refs = {r["seed"]: r for r in recs if r["learner"] == "refs"}
    fits = {(r["learner"], r["seed"]): r for r in recs if r["learner"] != "refs"}
    learners = [n for n in ex.LEARNERS if any(n == k[0] for k in fits)]
    seeds = sorted(refs)
    for r in recs:                        # every job must describe the same victim and shared settings
        assert r["ecfg"] == recs[0]["ecfg"] and r["cfg"] == recs[0]["cfg"], f"mixed settings in {r['argv']}"
    runs = []
    for s in seeds:
        run = {"costs": dict(refs[s]["costs"]), "hist": {}, "times": {}, "logs": {}, "ref": refs[s]["ref"],
               "profile": dict(refs[s]["profile"]), "victim": dict(refs[s]["victim"]), "params": {}}
        for n in learners:
            r = fits[(n, s)]
            run["costs"][n], run["profile"][n], run["victim"][n] = r["cost"], r["profile"], r["victim"]
            run["hist"][n], run["times"][n], run["logs"][n], run["params"][n] = r["hist"], r["time"], r["log"], r["params"]
        runs.append(run)
    r0, first = refs[seeds[0]], {n: fits[(n, seeds[0])] for n in learners}
    return {"ecfg": r0["ecfg"], "lam": r0["cfg"].lam, "seeds": tuple(seeds),
            "episodes": {n: r["lcfg"].K_iters * r["lcfg"].N_episodes for n, r in first.items()},
            "episodes_per_update": r0["cfg"].N_episodes,
            "hp": {n: r["hp"] for n, r in first.items()}, "trials": {n: r["trials"] for n, r in first.items()},
            "cfg": r0["cfg"], "eval_episodes": r0["eval_episodes"], "runs": runs}


def main():
    p = argparse.ArgumentParser(description="Merge a submit.sh run into one results pickle per victim.")
    p.add_argument("run_dir")
    p.add_argument("--suffix", default="", help="file-name suffix the notebook expects, e.g. _fair")
    a = p.parse_args()
    by_name = defaultdict(list)
    for r in load_records(a.run_dir):
        by_name[r["name"]].append(r)
    for name, recs in by_name.items():
        res = merge(recs)
        path = Path(a.run_dir) / ex.result_name(name, res["ecfg"], res["lam"], res["cfg"].myopic_prior, a.suffix)
        with open(path, "wb") as f:
            pickle.dump(res, f)
        costs = {n: sum(r["costs"][n] for r in res["runs"]) / len(res["runs"]) for n in res["runs"][0]["costs"]}
        print(f"{name}: {len(res['seeds'])} seeds, mean test cost "
              + "  ".join(f"{n}={c:.3f}" for n, c in costs.items()) + f"\nsaved {path}")


if __name__ == "__main__":
    main()
