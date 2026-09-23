"""Turn simulate.py's summary.json into markdown tables (+ an MAE-curve plot).

    python report.py results/main
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

COLS = [
    ("mae_final", "MAE@T", "{:.2f}"),
    ("mae_auc", "MAE-AUC", "{:.2f}"),
    ("success_rate", "succ", "{:.2f}"),
    ("abs_success_gap", "|succ-π*|", "{:.2f}"),
    ("zpd_rate", "ZPD", "{:.2f}"),
    ("too_hard_rate", "too hard", "{:.2f}"),
    ("class_entropy", "H(class)", "{:.2f}"),
    ("max_repeat_run", "max run", "{:.1f}"),
    ("weak_target_rate", "weak-hit", "{:.2f}"),
    ("prereq_violation_rate", "prereq viol", "{:.2f}"),
]


def pool(rows):
    """Average conditions across tutors (each tutor has the same N)."""
    out = {}
    for k, _, _ in COLS + [("learning_gain", "", ""), ("mae_curve", "", "")]:
        v = [r[k] for r in rows]
        out[k] = np.mean(np.array(v, float), 0)
    return out


def table(groups, key_name, cols):
    hdr = f"| {key_name} | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [hdr, sep]
    for name, r in groups:
        lines.append(f"| {name} | " + " | ".join(fmt.format(float(r[k])) for k, _, fmt in cols) + " |")
    return "\n".join(lines)


def main(res_dir):
    res_dir = Path(res_dir)
    data = json.loads((res_dir / "summary.json").read_text())
    conds = data["conditions"]
    cfg = data["config"]
    by = defaultdict(list)
    for c in conds:
        by[(c["learn_rate"], c["init"], c["policy"])].append(c)
    pols = cfg["policies"] if isinstance(cfg["policies"], list) else list(cfg["policies"])
    pols = [p for p in pols if any(k[2] == p for k in by)]
    inits = [i for i in cfg["inits"] if any(k[1] == i for k in by)]
    etas = sorted({k[0] for k in by})
    tutors = sorted({c["tutor"] for c in conds})

    md = [f"# Bug-selection simulation — {res_dir.name}\n",
          f"Tutors (learner source): {', '.join(tutors)}; "
          f"{conds[0]['n_learners']} learners/tutor; T={cfg['rounds']} rounds; "
          f"{cfg['seeds']} seeds; π*={cfg['pi_star']}. Pooled over tutors unless noted.\n"]

    eta0 = etas[0]
    md.append(f"## 1. Policies (init = socratic_pop, learning η={eta0})\n")
    md.append(table([(p, pool(by[(eta0, "socratic_pop", p)])) for p in pols], "policy", COLS))

    md.append(f"\n\n## 2. Initial belief (policy = yaksha, η={eta0})\n")
    cols = [("mae_curve", "MAE@0", "{:.2f}")] + COLS[:3]
    rows = []
    for i in inits:
        r = pool(by[(eta0, i, "yaksha")])
        r = {**r, "mae_curve": r["mae_curve"][0]}
        rows.append((i, r))
    md.append(table(rows, "init", cols))

    md.append(f"\n\n## 3. Per tutor (socratic_pop, yaksha vs random, η={eta0})\n")
    tcols = [("mae0", "MAE@0", "{:.2f}"), ("mae_final", "MAE@T yaksha", "{:.2f}"),
             ("mae_rand", "MAE@T random", "{:.2f}"), ("mae_pop0", "MAE@0 pop_only", "{:.2f}"),
             ("success_rate", "succ yaksha", "{:.2f}")]
    rows = []
    for t in tutors:
        y = next(c for c in by[(eta0, "socratic_pop", "yaksha")] if c["tutor"] == t)
        rnd = next(c for c in by[(eta0, "socratic_pop", "random")] if c["tutor"] == t)
        pop = next(c for c in by[(eta0, "pop_only", "yaksha")] if c["tutor"] == t)
        rows.append((t, {"mae0": y["mae_curve"][0], "mae_final": y["mae_final"],
                         "mae_rand": rnd["mae_final"], "mae_pop0": pop["mae_curve"][0],
                         "success_rate": y["success_rate"]}))
    md.append(table(rows, "tutor", tcols))

    for eta in etas[1:]:
        md.append(f"\n\n## 4. With learning (η={eta}, init = socratic_pop)\n")
        lcols = [("learning_gain", "Σ level gain", "{:.2f}")] + COLS[:3] + [COLS[6], COLS[8]]
        md.append(table([(p, pool(by[(eta, "socratic_pop", p)])) for p in pols], "policy", lcols))

    (res_dir / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for p in ["yaksha", "info_only", "success_only", "random", "round_robin"]:
            if (eta0, "socratic_pop", p) in by:
                ax[0].plot(pool(by[(eta0, "socratic_pop", p)])["mae_curve"], label=p)
        ax[0].set(xlabel="round", ylabel="belief MAE", title="Policies (socratic_pop init)")
        for i in inits:
            ax[1].plot(pool(by[(eta0, i, "yaksha")])["mae_curve"], label=i)
        ax[1].set(xlabel="round", title="Initial belief (yaksha)")
        for a in ax:
            a.legend(fontsize=8); a.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(res_dir / "mae_curves.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/main")
