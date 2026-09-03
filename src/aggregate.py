"""Fold every currently-available seed's results.json for one (task, mode) into one or
more combined files -- run_seeds.sh/sweep.sh call this automatically after training;
run it by hand any time to snapshot whatever seeds exist right now.

    python src/aggregate.py --task regression --mode sort
    python src/aggregate.py --task regression --mode lowrank

Matching is by ACTUAL config content (each per-seed results.json's own env_config/
method_config), not by directory name -- run.py's own run_dir naming includes a config
tag for human readability (see run.py's config_tag), but this script never relies on
parsing it. That matters because sensitivity/*.sh sweeps write many seed-dirs for the
SAME (task, mode) at DIFFERENT c_edit/eta/m/T/d/rank_r/k_max, all sitting in output/ at
once if you run more than one sweep concurrently -- grouping by content means each
distinct config gets its own output file even if their seed-dirs are interleaved,
instead of one sweep's seeds silently drifting into another's summary. The grouping key
also includes each run's "sens_axis" tag (set by sensitivity/*.sh via run.py's
--sens-axis) so two different axis scripts that happen to produce the identical config
(e.g. every axis's own baseline point) still never fold into the same combined file.

Per-seed results.json files are transient: each is folded in and then deleted, so only
the combined snapshot(s) persist. Every distinct config found gets its own fresh
output/<task>_<mode>_<tag>_results_<timestamp>.json (tag = the same c_edit/eta/m/T/d/
rank_r/k_max summary run.py's run_dir uses) -- it does not update a previous one, so
separate runs (e.g. training more seeds later) don't overwrite each other; if you want
everything in one place, load and concatenate the snapshots you care about.

    {"task":..., "mode":..., "seeds": [...], "runs": [{"seed":0, "costs": {...}}, ...],
     "summary": {"first-order": {"mean":..., "std":...}, ...}}
"""
import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np

CONFIG_FIELDS = ["c_edit", "eta", "m", "T", "d", "rank_r", "k_max"]


def _fmt(x):
    if x is None:
        return "na"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def config_tag(cfg):
    return (f"ce{_fmt(cfg['c_edit'])}_eta{_fmt(cfg['eta'])}_m{_fmt(cfg['m'])}"
            f"_T{_fmt(cfg['T'])}_d{_fmt(cfg['d'])}_r{_fmt(cfg['rank_r'])}_k{_fmt(cfg['k_max'])}")


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
        key = tuple(cfg[f] for f in CONFIG_FIELDS) + (run.get("sens_axis"),)
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

        out = {"task": args.task, "mode": args.mode, "sens_axis": runs[0].get("sens_axis"),
               "seeds": [r["seed"] for r in runs], "runs": runs, "summary": summary}
        tag = config_tag(cfg)
        out_path = os.path.join(args.outdir, f"{args.task}_{args.mode}_{tag}_results_{timestamp}.json")
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


if __name__ == "__main__":
    main()
