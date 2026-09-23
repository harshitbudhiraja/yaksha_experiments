"""Simulated-learner evaluation of the Yaksha bug-selection scheme.

Learners are the simulated students from `proficiency_diagnosis`: each
interactive episode there gives
    p_true  the student's ground-truth 12-stage profile (what the student
            model was told to act out), and
    p_pred  the Socratic tutor's estimate of it after the dialogue, plus the
            tutor's per-stage confidence.
That is exactly the input the bug-injection stage gets in Yaksha. Following
the paper's cold-start rule, the initial belief b_0 takes S1-S6 from the
Socratic estimate and seeds S7-S12 from the population average.

Each round, every (beta, d) pair is scored with

    Score = l1 * I  -  l2 * |pi_hat - pi*|  +  l3 * <q_beta, w_t>  -  l4 * rho_t(beta)

and the argmax is shown to the learner. The learner's response is sampled
from the 2PL model at its *true* conjunctive ability
theta = min_{k in K(beta)} s_k, and the factorised belief is updated by an
assumed-density (per-stage marginal) Bayes step. The LLM realiser and the
debugging dialogue are abstracted away: a round's outcome is one Bernoulli
draw, so this measures the selection rule in isolation.

Optionally (--learn-rates > 0) the learner improves: a success on a bug that
was a stretch for its weakest required stage raises that stage by one level
with probability eta. That is a modelling assumption, not a finding.

Usage:
    python simulate.py --tutors qwen3-coder-30b qwen25-coder-14b ... \
        --rounds 30 --seeds 5 --output results/main
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.stages import STAGE_IDS  # noqa: E402
from qmatrix import A, B, BUG_IDS, K, LEVELS, PRE, Q, REQ  # noqa: E402

HERE = Path(__file__).resolve().parent
PD_RESULTS = HERE.parent / "proficiency_diagnosis" / "results" / "cross_model"

D_GRID = np.arange(1.0, 5.01, 0.5)      # candidate difficulties d
ND = len(D_GRID)
NL = len(LEVELS)
N_PRE = 6                                # S1..S6 are pre-coding

# lambda = (diagnostic, success, targeting, variety)
POLICIES = {
    "yaksha":        (1, 1, 1, 1),
    "no_info":       (0, 1, 1, 1),
    "no_success":    (1, 0, 1, 1),
    "no_targeting":  (1, 1, 0, 1),
    "no_variety":    (1, 1, 1, 0),
    "info_only":     (1, 0, 0, 0),       # classic CAT: max Fisher information
    "success_only":  (0, 1, 0, 0),       # pure 85%-rule difficulty matching
    "random":        None,
    "round_robin":   None,               # cycle classes, fixed mid difficulty
}
INITS = ["socratic_pop", "pop_only", "tutor_all12", "oracle"]


# ----------------------------------------------------------------------------
# learners

def load_learners(tutor: str):
    path = PD_RESULTS / tutor / "rollouts.jsonl"
    rows = [json.loads(l) for l in path.open()]
    out = []
    for r in rows:
        if r.get("_meta") or r.get("condition") != "interactive":
            continue
        if not r.get("p_pred") or any(s not in r["p_pred"] for s in STAGE_IDS):
            continue
        conf = {}
        for t in reversed(r.get("turns", [])):
            if t.get("parse_ok") and t.get("confidence"):
                conf = t["confidence"]
                break
        out.append({
            "profile_id": r["profile_id"], "archetype": r["archetype"],
            "task_id": r["task_id"],
            "p_true": [int(r["p_true"][s]) for s in STAGE_IDS],
            "p_pred": [int(r["p_pred"][s]) for s in STAGE_IDS],
            "conf": [float(conf.get(s, 0.5) or 0.5) for s in STAGE_IDS],
        })
    return out


def point_belief(pred, conf):
    """Discretised Gaussian around the tutor's level; lower confidence ->
    wider. Confidence is capped at 0.8 because LLM tutors report 0.9 almost
    everywhere, and a near-point-mass prior on a wrong level never recovers."""
    c = np.clip(conf, 0.0, 0.8)
    sigma = 0.6 + 1.2 * (1 - c)                      # [N, K]
    d = LEVELS[None, None, :] - pred[..., None]
    b = np.exp(-0.5 * (d / sigma[..., None]) ** 2)
    return b / b.sum(-1, keepdims=True)


def init_belief(kind, p_true, p_pred, conf, pop):
    N = len(p_true)
    pop_b = np.broadcast_to(pop, (N, K, NL)).copy()
    if kind == "pop_only":
        return pop_b
    tut = point_belief(p_pred, conf)
    if kind == "tutor_all12":
        return tut
    if kind == "socratic_pop":
        b = pop_b
        b[:, :N_PRE] = tut[:, :N_PRE]
        return b
    if kind == "oracle":
        return point_belief(p_true, np.full(p_true.shape, 0.8))
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# model pieces

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# P[b, d, l] = sigma(a_b (l - d))
P_TAB = sigmoid(A[:, None, None] * (LEVELS[None, None, :] - D_GRID[None, :, None]))
FI_TAB = (A[:, None, None] ** 2) * P_TAB * (1 - P_TAB)


def survival(b):
    """S(l) = Pr(s >= l), last axis levels."""
    return np.flip(np.cumsum(np.flip(b, -1), -1), -1)


def theta_pmf(b):
    """pmf of theta(s, beta) = min over K(beta) of s_k under the factorised
    belief, for every class: [N, B, L]  (eq. survival)."""
    S = survival(b)                                               # [N,K,L]
    Sm = np.where(REQ[None, :, :, None], S[:, None, :, :], 1.0)   # [N,B,K,L]
    ST = Sm.prod(axis=2)                                          # [N,B,L]
    return ST - np.concatenate([ST[..., 1:], np.zeros_like(ST[..., :1])], -1)


def need_weights(b, tau_lvl):
    """w_tk = u_tk * (5 - s_bar_tk) * prod_{j in pre(k)} 1[s_bar_tj >= tau]."""
    sbar = (b * LEVELS).sum(-1)                                   # [N,K]
    H = -(b * np.log(np.clip(b, 1e-12, None))).sum(-1) / np.log(NL)
    ok = sbar >= tau_lvl                                          # [N,K]
    gate = np.where(PRE[None], ok[:, None, :], True).all(-1)      # [N,K]
    return H * (5 - sbar) * gate


def minmax(x, axis):
    lo = x.min(axis=axis, keepdims=True)
    hi = x.max(axis=axis, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-9)


def select(policy, b, hist, rng, args, t):
    """Returns chosen class index [N] and difficulty index [N]."""
    N = b.shape[0]
    if policy == "random":
        return rng.integers(0, B, N), rng.integers(0, ND, N)
    if policy == "round_robin":
        return np.full(N, t % B), np.full(N, int(np.argmin(abs(D_GRID - 3.0))))

    l1, l2, l3, l4 = POLICIES[policy]
    pmf = theta_pmf(b)                                            # [N,B,L]
    info = np.einsum("nbl,bdl->nbd", pmf, FI_TAB)                 # E_b[I]
    pihat = np.einsum("nbl,bdl->nbd", pmf, P_TAB)                 # eq. pihat
    succ = np.abs(pihat - args.pi_star)
    tgt = need_weights(b, args.tau_lvl) @ Q.T                     # [N,B]
    gam = args.gamma ** np.arange(hist.shape[1])                  # [W]
    rho = ((hist[:, :, None] == np.arange(B)) * gam[None, :, None]).sum(1)

    flat = lambda x: x.reshape(N, -1)
    terms = [
        flat(info),
        flat(succ),
        flat(np.broadcast_to(tgt[:, :, None], (N, B, ND))),
        flat(np.broadcast_to(rho[:, :, None], (N, B, ND))),
    ]
    if args.normalize:
        terms = [minmax(x, 1) for x in terms]
    score = l1 * terms[0] - l2 * terms[1] + l3 * terms[2] - l4 * terms[3]
    score = score + 1e-7 * rng.random(score.shape)                # random tie-break
    idx = score.argmax(1)
    return idx // ND, idx % ND


def update_belief(b, beta, y, d):
    """Assumed-density Bayes update of each required stage's marginal:
    Pr(y | s_k = l) = sum_m Pr(min_{j != k} s_j = m) * P(y | min(l, m))."""
    N = b.shape[0]
    req = REQ[beta]                                               # [N,K]
    a = A[beta][:, None]
    S = survival(b)
    Sreq = np.where(req[..., None], S, 1.0)                       # [N,K,L]
    newb = b.copy()
    # P(y | theta = l) for this round's item
    py = sigmoid(a * (LEVELS[None, :] - d[:, None]))              # [N,L]
    py = np.where(y[:, None] == 1, py, 1 - py)
    minlv = np.minimum(LEVELS[:, None], LEVELS[None, :]) - 1      # [L(l),L(m)]
    for k in range(K):
        rows = req[:, k]
        if not rows.any():
            continue
        others = np.delete(Sreq, k, axis=1).prod(1)               # [N,L]
        pm = others - np.concatenate([others[:, 1:], np.zeros((N, 1))], 1)
        # if no other required stage, S_others == 1 so pm puts all mass on 5
        lik = (pm[:, None, :] * py[:, minlv]).sum(-1)             # [N,L(l)]
        post = b[:, k] * lik
        post /= post.sum(-1, keepdims=True)
        newb[rows, k] = post[rows]
    return newb


# ----------------------------------------------------------------------------
# episode loop

def run_condition(learners, init, policy, eta, args, seed, pop):
    rng = np.random.default_rng(seed)
    p_true0 = np.array([l["p_true"] for l in learners])
    p_pred = np.array([l["p_pred"] for l in learners])
    conf = np.array([l["conf"] for l in learners])
    N = len(learners)

    s = p_true0.copy()
    b = init_belief(init, p_true0, p_pred, conf, pop)
    hist = np.full((N, args.window), -1)
    weak3 = np.argsort(p_true0 + 1e-3 * rng.random(p_true0.shape), 1)[:, :3]
    weak3_mask = np.zeros((N, K), bool)
    np.put_along_axis(weak3_mask, weak3, True, 1)

    T = args.rounds
    mae = np.zeros((T + 1, N))
    mae_pre = np.zeros((T + 1, N))
    mae_code = np.zeros((T + 1, N))
    ys = np.zeros((T, N))
    ptrue = np.zeros((T, N))
    betas = np.zeros((T, N), int)
    ds = np.zeros((T, N))
    prereq_viol = np.zeros((T, N))
    weak_hit = np.zeros((T, N))

    def record(t):
        err = np.abs((b * LEVELS).sum(-1) - s)
        mae[t], mae_pre[t], mae_code[t] = err.mean(1), err[:, :N_PRE].mean(1), err[:, N_PRE:].mean(1)

    record(0)
    for t in range(T):
        beta, di = select(policy, b, hist, rng, args, t)
        d = D_GRID[di]
        req = REQ[beta]
        theta = np.where(req, s, 99).min(1)
        p = sigmoid(A[beta] * (theta - d))
        y = (rng.random(N) < p).astype(int)

        # a required stage whose own prerequisite is truly unmet
        unmet = s < args.tau_lvl                                  # [N,K]
        viol = (req & (PRE[None] & unmet[:, None, :]).any(-1)).any(1)

        ys[t], ptrue[t], betas[t], ds[t] = y, p, beta, d
        prereq_viol[t] = viol
        weak_hit[t] = (req & weak3_mask).any(1)

        b = update_belief(b, beta, y, d)
        hist = np.concatenate([beta[:, None], hist[:, :-1]], 1)

        if eta > 0:
            kstar = np.where(req, s, 99).argmin(1)
            stretch = d >= s[np.arange(N), kstar] - 1.5
            gain = (y == 1) & stretch & (rng.random(N) < eta)
            s[np.arange(N), kstar] = np.minimum(5, s[np.arange(N), kstar] + gain)
        record(t + 1)

    # per-episode variety stats
    ent, maxrun, distinct = [], [], []
    for n in range(N):
        c = np.bincount(betas[:, n], minlength=B) / T
        nz = c[c > 0]
        ent.append(float(-(nz * np.log(nz)).sum() / np.log(B)))
        distinct.append(int((c > 0).sum()))
        run = best = 1
        for i in range(1, T):
            run = run + 1 if betas[i, n] == betas[i - 1, n] else 1
            best = max(best, run)
        maxrun.append(best)

    succ_rate = ys.mean(0)
    return {
        "mae_curve": mae.mean(1).tolist(),
        "mae_pre_curve": mae_pre.mean(1).tolist(),
        "mae_code_curve": mae_code.mean(1).tolist(),
        "success_curve": ys.mean(1).tolist(),
        "mae_final": float(mae[-1].mean()),
        "mae_pre_final": float(mae_pre[-1].mean()),
        "mae_code_final": float(mae_code[-1].mean()),
        "mae_auc": float(mae.mean()),
        "success_rate": float(succ_rate.mean()),
        "abs_success_gap": float(np.abs(succ_rate - args.pi_star).mean()),
        "mean_p_true": float(ptrue.mean()),
        "zpd_rate": float(((ptrue >= 0.7) & (ptrue <= 0.95)).mean()),
        "too_hard_rate": float((ptrue < 0.5).mean()),
        "too_easy_rate": float((ptrue > 0.97).mean()),
        "class_entropy": float(np.mean(ent)),
        "distinct_classes": float(np.mean(distinct)),
        "max_repeat_run": float(np.mean(maxrun)),
        "weak_target_rate": float(weak_hit.mean()),
        "prereq_violation_rate": float(prereq_viol.mean()),
        "mean_difficulty": float(ds.mean()),
        "learning_gain": float((s - p_true0).sum(1).mean()),
        "class_hist": np.bincount(betas.ravel(), minlength=B).tolist(),
    }


def average(runs):
    out = {}
    for k in runs[0]:
        v = [r[k] for r in runs]
        if isinstance(v[0], list):
            out[k] = np.mean(np.array(v, float), 0).tolist()
        else:
            out[k] = float(np.mean(v))
            out[k + "_sd"] = float(np.std(v))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tutors", nargs="+",
                    default=["qwen3-coder-30b", "qwen25-coder-14b", "gemma4-31b",
                             "gptoss-20b", "llama31-8b"])
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--inits", nargs="+", default=INITS)
    ap.add_argument("--learn-rates", nargs="+", type=float, default=[0.0, 0.3])
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pi-star", type=float, default=0.85)
    ap.add_argument("--gamma", type=float, default=0.7, help="repetition decay")
    ap.add_argument("--window", type=int, default=5, help="W, repetition window")
    ap.add_argument("--tau-lvl", type=float, default=2.5,
                    help="prerequisite readiness threshold on mean level")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false",
                    help="use raw score terms instead of per-round min-max scaling")
    ap.add_argument("--output", default=str(HERE / "results" / "main"))
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = {"config": {**vars(args), "bug_classes": BUG_IDS,
                          "d_grid": D_GRID.tolist(), "policies": POLICIES},
               "conditions": []}
    t0 = time.time()
    for tutor in args.tutors:
        learners = load_learners(tutor)
        p_true = np.array([l["p_true"] for l in learners])
        # population-average prior: per-stage level histogram, add-one smoothed
        pop = np.stack([np.bincount(p_true[:, k] - 1, minlength=NL) + 1
                        for k in range(K)]).astype(float)
        pop /= pop.sum(-1, keepdims=True)
        print(f"[{tutor}] {len(learners)} simulated learners", flush=True)
        for eta in args.learn_rates:
            for init in args.inits:
                for pol in args.policies:
                    runs = [run_condition(learners, init, pol, eta, args, 1000 * sd + 7, pop)
                            for sd in range(args.seeds)]
                    res = {"tutor": tutor, "learn_rate": eta, "init": init,
                           "policy": pol, "n_learners": len(learners), **average(runs)}
                    results["conditions"].append(res)
                    print(f"  eta={eta} {init:13s} {pol:13s} "
                          f"MAE0={res['mae_curve'][0]:.2f} MAE_T={res['mae_final']:.2f} "
                          f"succ={res['success_rate']:.2f} ent={res['class_entropy']:.2f} "
                          f"weak={res['weak_target_rate']:.2f} "
                          f"viol={res['prereq_violation_rate']:.2f} "
                          f"gain={res['learning_gain']:.2f}", flush=True)
    (out / "summary.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {out/'summary.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
