import argparse
import json
from collections import defaultdict
from pathlib import Path

from runs_io import load_records


def main():
    p = argparse.ArgumentParser(description="Pick each learner's best sweep config (lowest mean tuning score).")
    p.add_argument("sweep_dirs", nargs="+", help="one or more sweep.sh run directories")
    p.add_argument("--out", help="default: <first sweep dir>/selected.json")
    a = p.parse_args()
    recs = [r for d in a.sweep_dirs for r in load_records(d)]
    scores, hps, refs = defaultdict(list), {}, {}
    for r in recs:
        if r["learner"] == "refs":
            refs[r["name"]] = r["score"]
            continue
        key = (r["name"], r["learner"].lower(), r["tag"])
        scores[key].append(r["score"])
        hps[key] = r["hp"]
    out = {"refs": refs, "seeds": sorted({r["seed"] for r in recs if r["learner"] != "refs"}),
           "score_seed": recs[0]["score_seed"], "selected": {}}
    groups = defaultdict(dict)
    for (name, learner, tag), v in scores.items():
        groups[(name, learner)][tag] = sum(v) / len(v)
    for (name, learner), mean in sorted(groups.items()):
        best = min(mean, key=mean.get)
        out["selected"].setdefault(name, {})[learner] = dict(
            hp=hps[(name, learner, best)], score=mean[best], all=mean,
            trials=sum(len(scores[(name, learner, t)]) for t in mean))
        print(f"{name:9}{learner:6}best {best} {hps[(name, learner, best)]}  score {mean[best]:.3f}   "
              + "  ".join(f"{t}={m:.3f}" for t, m in mean.items()))
    for name, s in refs.items():
        print(f"{name:9}refs  " + "  ".join(f"{k}={v:.3f}" for k, v in s.items()))
    path = Path(a.out or Path(a.sweep_dirs[0]) / "selected.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
