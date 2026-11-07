import pickle
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))     # records pickle src dataclasses


def job_outputs(run_dir):
    """The --out pickle of every line in run_dir/jobs.txt."""
    outs = []
    for line in (Path(run_dir) / "jobs.txt").read_text().splitlines():
        args = shlex.split(line)
        outs.append(Path(args[args.index("--out") + 1]))
    return outs


def load_records(run_dir):
    outs = job_outputs(run_dir)
    missing = [p for p in outs if not p.exists()]
    if missing:
        sys.exit(f"{len(missing)} of {len(outs)} jobs have no output, e.g.\n  "
                 + "\n  ".join(map(str, missing[:10]))
                 + "\nresubmit with the same settings: finished jobs are skipped")
    recs = []
    for p in outs:
        with open(p, "rb") as f:
            recs.append(pickle.load(f))
    return recs
