import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np

CONFIG_FIELDS = ["c_edit", "eta", "m", "T", "d"]


def _fmt(x):
    if x is None:
        return "na"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def config_tag(cfg):
    return (f"ce{_fmt(cfg['c_edit'])}_eta{_fmt(cfg['eta'])}_m{_fmt(cfg['m'])}"
            f"_T{_fmt(cfg['T'])}_d{_fmt(cfg['d'])}")


def print_summary(out, cfg):
    """Per-method cost across seeds, best first, with each seed's own value."""
    summary, runs = out["summary"], out["runs"]
    names = sorted(summary, key=lambda n: summary[n]["mean"])
    null = summary.get("null", {}).get("mean")
    seeds = out["seeds"]

    print(f"\n  task={out['task']}  mode={out['mode']}  learner={out['learner']}"
          + (f"  axis={out['sens_axis']}" if out.get("sens_axis") else ""))
    print(f"  m={cfg['m']} T={cfg['T']} d={cfg['d']} eta={cfg['eta']} "
          f"c_edit={cfg['c_edit']} lam={cfg['lam']} L={cfg['L']} B={cfg['B']}  "
          f"seeds={len(seeds)}")

    per_seed = len(seeds) <= 10
    head = f"  {'method':14s}{'mean':>10s}{'std':>9s}{'vs null':>9s}"
    if per_seed:
        head += "   " + "".join(f"{f's{s}':>8s}" for s in seeds)
    print("\n" + head)
    print("  " + "-" * (len(head) - 2))
    for n in names:
        st = summary[n]
        rel = f"{(null - st['mean']) / null * 100:+.1f}%" if null else "--"
        mark = "*" if n == names[0] else " "
        line = f"{mark} {n:14s}{st['mean']:10.4f}{st['std']:9.4f}{rel:>9s}"
        if per_seed:
            by_seed = {r["seed"]: r["costs"].get(n) for r in runs}
            line += "   " + "".join(
                f"{by_seed[s]:8.3f}" if by_seed.get(s) is not None else f"{'--':>8s}"
                for s in seeds)
        print(line)
    print("  (lower is better; * = best mean; vs null = cost reduction over no attack)")
    incomplete = [str(r["seed"]) for r in runs if r.get("complete") is False]
    if incomplete:
        print(f"  WARNING: seed(s) {', '.join(incomplete)} incomplete (interrupted); their "
              "missing methods are excluded from the means")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--outdir", default="output")
    args = p.parse_args()

    # glob broadly (any tag/timestamp scheme in the directory name) -- (task, mode) and
    # the sweep config are both verified from each file's own content just below, not
    # from this pattern, so it doesn't matter if the naming convention changes later.
    pattern = os.path.join(args.outdir, f"{args.task}_*seed*", "results.json")
    groups = {}
    for path in glob.glob(pattern):
        with open(path) as f:
            run = json.load(f)
        if run["mode"] != args.mode:
            continue
        cfg = {**run["env_config"], **run["method_config"]}
        # learner is part of the key: fvi and fevi runs at the same config are
        # different algorithms and must not be averaged into one summary
        key = tuple(cfg[f] for f in CONFIG_FIELDS) + (run.get("sens_axis"),
                                                      run.get("learner"))
        groups.setdefault(key, []).append((path, run))

    if not groups:
        raise SystemExit(f"no results found matching {pattern!r} with mode={args.mode!r}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    for key, items in groups.items():
        items.sort(key=lambda pr: pr[1]["seed"])
        paths = [p for p, _ in items]
        runs = [r for _, r in items]
        cfg = {**runs[0]["env_config"], **runs[0]["method_config"]}

        names = sorted({n for r in runs for n in r["costs"]})
        summary = {
            n: {"mean": float(np.mean([r["costs"][n] for r in runs if n in r["costs"]])),
                "std": float(np.std([r["costs"][n] for r in runs if n in r["costs"]]))}
            for n in names
        }

        learner = runs[0].get("learner", "fvi")
        out = {"task": args.task, "mode": args.mode, "learner": learner,
               "sens_axis": runs[0].get("sens_axis"),
               "seeds": [r["seed"] for r in runs], "runs": runs, "summary": summary}
        tag = config_tag(cfg)
        out_path = os.path.join(
            args.outdir, f"{args.task}_{args.mode}_{learner}_{tag}_results_{timestamp}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        # the per-seed files are now folded in -- remove them (and their directory, if
        # nothing else, e.g. no --diagnostics figures, is left in it)
        for path in paths:
            os.remove(path)
            run_dir = os.path.dirname(path)
            try:
                os.rmdir(run_dir)
            except OSError:
                pass   # not empty (e.g. --diagnostics figures live there too) -- leave it

        print(f"wrote {out_path}  ({len(runs)} seed(s): {out['seeds']})")
        print_summary(out, cfg)


if __name__ == "__main__":
    main()
