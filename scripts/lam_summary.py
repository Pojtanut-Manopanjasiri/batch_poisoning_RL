import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))     # the pickles hold src's config classes

import numpy as np

ORDER = ["null", "greedy", "TRPO", "BDQ", "FEVI", "FEVI-exact"]
COLORS = {"null": "#5b5b5b", "greedy": "#eb6834", "TRPO": "#4a3aa7", "BDQ": "#e34948", "FEVI": "#2a78d6",
          "FEVI-exact": "#12a594"}
MARKERS = {"null": "D", "greedy": "s", "TRPO": "^", "BDQ": "X", "FEVI": "o", "FEVI-exact": "P"}
STYLES = {"null": "--", "greedy": "-.", "FEVI-exact": (0, (4, 1.5))}       # FEVI-exact: dashed, a test-time variant
COLUMNS = ["victim", "lam", "policy", "seeds", "cost", "cost_std", "vs_null", "minus_greedy", "edit_cost",
           "target_cost", "edits_per_step", "final_gap"]
PANELS = [("vs_null", "test cost / null's cost"),
          ("target_cost", r"distance part $\sum_t \frac{1}{2}\|\theta_{t+1} - \tilde{\theta}\|^2$"),
          ("edits_per_step", r"edits per step $\sum_i |a_i - 1|$")]


def load(paths):
    """Merged results pickles (or run directories holding them) -> {victim label: [results, one per lambda]}."""
    groups = defaultdict(list)
    for p in map(Path, paths):
        for f in (sorted(p.glob("*.pkl")) if p.is_dir() else [p]):
            with open(f, "rb") as fh:
                r = pickle.load(fh)
            groups[r["ecfg"].dataset or r["ecfg"].task].append(r)
    return {k: sorted(v, key=lambda r: r["lam"]) for k, v in groups.items()}


def rows(name, results):
    """One row per (lambda, policy): test cost over seeds, split into its edit and distance-to-target parts."""
    out = []
    for r in results:
        rho, runs = r["lam"] / (2 * r["ecfg"].m), r["runs"]
        cost = {n: np.array([run["costs"][n] for run in runs]) for n in ORDER if n in runs[0]["costs"]}
        for n, c in cost.items():
            pert = np.array([run["profile"][n]["pert"] for run in runs])        # (seeds, T): mean edits per step
            edit = rho * pert.sum(1)
            out.append({"victim": name, "lam": r["lam"], "policy": n, "seeds": len(runs), "cost": c.mean(),
                        "cost_std": c.std(), "vs_null": (c / cost["null"]).mean(),
                        "vs_null_std": (c / cost["null"]).std(), "minus_greedy": (c - cost["greedy"]).mean(),
                        "edit_cost": edit.mean(), "target_cost": (c - edit).mean(),
                        "target_cost_std": (c - edit).std(), "edits_per_step": pert.mean(),
                        "edits_per_step_std": pert.mean(1).std(),
                        "final_gap": np.mean([run["profile"][n]["gap"][-1] for run in runs])})
    return out


def table(rs):
    lams = sorted({r["lam"] for r in rs})
    names = [n for n in ORDER if any(r["policy"] == n for r in rs)]
    cell = {(r["lam"], r["policy"]): r for r in rs}
    for key, title, fmt in (("cost", "test cost (edit part + distance part)", "8.3f"),
                            ("vs_null", "test cost / null's cost", "8.3f"),
                            ("minus_greedy", "test cost - greedy's cost", "+8.3f"),
                            ("target_cost", "distance part of the cost (comparable across lambda)", "8.3f"),
                            ("edits_per_step", "edits per step", "8.2f")):
        print(f"\n[{rs[0]['victim']}] {title}\n{'lambda':>8}" + "".join(f"{n:>12}" for n in names))
        for lam in lams:
            print(f"{lam:8g}" + "".join(f"{format(cell[lam, n][key], fmt):>12}" if (lam, n) in cell else f"{'-':>12}"
                                        for n in names))
    print()
    for n in [n for n in names if n not in ("null", "greedy")]:
        mine = [cell[lam, n] for lam in lams if (lam, n) in cell]
        low, rel, gr = (min(mine, key=lambda r: r[k]) for k in ("cost", "vs_null", "minus_greedy"))
        print(f"{n:11} lowest cost at lambda={low['lam']:g} ({low['cost']:.3f}); lowest vs. null at "
              f"lambda={rel['lam']:g} ({rel['vs_null']:.3f}); best against greedy at lambda={gr['lam']:g} "
              f"({gr['minus_greedy']:+.3f})")
    print("raw cost rises with lambda for every policy (lambda prices the edits): compare policies within a row")


def plot(rs, path=None):
    """Cost relative to null, the distance part and the edits, against lambda (evenly spaced); bands: ± 1 std over seeds."""
    import matplotlib.pyplot as plt
    lams = sorted({r["lam"] for r in rs})
    x = {lam: i for i, lam in enumerate(lams)}
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6 * len(PANELS), 4.3))
    for ax, (key, ylabel) in zip(axes, PANELS):
        for n in [n for n in ORDER if any(r["policy"] == n for r in rs)]:
            mine = [r for r in rs if r["policy"] == n]
            xs, mu = [x[r["lam"]] for r in mine], np.array([r[key] for r in mine])
            sd = np.array([r[key + "_std"] for r in mine])
            ax.plot(xs, mu, ls=STYLES.get(n, "-"), marker=MARKERS[n], color=COLORS[n], lw=2, label=n)
            ax.fill_between(xs, mu - sd, mu + sd, color=COLORS[n], alpha=0.15, lw=0)
        ax.set_xticks(range(len(lams)), [f"{lam:g}" for lam in lams])
        ax.set_xlabel(r"edit-cost weight $\lambda$ (evenly spaced)")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.85")
        ax.set_axisbelow(True)
    axes[0].set_title(rs[0]["victim"], loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    if path:
        for ext in ("pdf", "png"):
            fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    return fig


def main():
    p = argparse.ArgumentParser(description="Compare the policies across runs that differ in the edit-cost weight.")
    p.add_argument("results", nargs="+", help="merged results pickles, or run directories holding them")
    p.add_argument("--out", required=True, help="folder for lam_summary.csv and cost_vs_lambda_<victim>.pdf/.png")
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    out, everything = Path(a.out), []
    out.mkdir(parents=True, exist_ok=True)
    for name, results in load(a.results).items():
        rs = rows(name, results)
        table(rs)
        plot(rs, out / f"cost_vs_lambda_{name}")
        everything += rs
    with open(out / "lam_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(everything)
    print(f"saved {out}/lam_summary.csv and cost_vs_lambda_<victim>.pdf/.png")


if __name__ == "__main__":
    main()
