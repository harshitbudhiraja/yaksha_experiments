"""Bug-class catalogue B, its Q-matrix, and the stage prerequisite graph used
by the Yaksha bug-selection scheme (paper Sec. "Bug Injection Scheme").

Each bug class beta carries a loading vector q_beta in {0, 0.3, 0.6, 1.0}^12
over the S1..S12 stages of common/stages.py:
    0    stage not involved
    0.3  peripheral
    0.6+ required ("load-bearing": K(beta) = {k : q_beta,k >= TAU_Q})
and a 2PL discrimination a_beta. The first twelve classes are the
representative bug class of each stage (Table "twelve stages"); the last four
are extra classes so that several classes compete for the same stages.

These loadings/discriminations are hand-set design values, not fitted from
data -- the simulation measures how the *selection rule* behaves given a
Q-matrix, not how good this particular Q-matrix is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.stages import STAGE_IDS  # noqa: E402

K = len(STAGE_IDS)
LEVELS = np.arange(1, 6)          # L = {1..5}
TAU_Q = 0.6                       # loading threshold for "required"

# (id, description, {stage: loading}, a_beta)
BUG_CLASSES = [
    ("bound_misread",      "Inclusive bound read as exclusive",
     {"S1": 1.0, "S9": 0.6, "S11": 0.3}, 1.2),
    ("missing_edge_branch", "Missing branch for empty input",
     {"S2": 1.0, "S11": 0.6, "S9": 0.3}, 1.4),
    ("greedy_for_dp",      "Greedy choice where DP is required",
     {"S3": 1.0, "S4": 0.6, "S2": 0.3}, 1.0),
    ("wrong_base_case",    "Recursion with an incorrect base case",
     {"S4": 1.0, "S9": 0.6, "S3": 0.3}, 1.2),
    ("nested_scan",        "Nested scan in place of a hash lookup",
     {"S5": 1.0, "S6": 0.6, "S12": 0.3}, 1.0),
    ("list_membership",    "List used for membership tests",
     {"S6": 1.0, "S5": 0.6, "S8": 0.3}, 1.2),
    ("omitted_step",       "Planned step omitted in implementation",
     {"S7": 1.0, "S10": 0.6, "S3": 0.3}, 1.4),
    ("int_division",       "Integer division for a real quantity",
     {"S8": 1.0, "S12": 0.6, "S9": 0.3}, 1.8),
    ("off_by_one_cmp",     "< in place of <=",
     {"S9": 1.0, "S10": 0.6, "S1": 0.3}, 1.8),
    ("shadowed_variable",  "Shadowed variable, silent wrong value",
     {"S10": 1.0, "S8": 0.6, "S9": 0.3}, 1.6),
    ("fails_on_duplicates", "Passes visible tests, fails on duplicates",
     {"S11": 1.0, "S2": 0.6, "S10": 0.3}, 1.4),
    ("wrong_return_type",  "Correct value, wrong return type",
     {"S12": 1.0, "S1": 0.3, "S8": 0.3}, 1.6),
    # extra classes that overlap the stages above
    ("mutate_while_iter",  "List mutated while iterating over it",
     {"S8": 1.0, "S10": 0.6}, 1.6),
    ("bad_accumulator_init", "Accumulator initialised to the wrong value",
     {"S9": 1.0, "S7": 0.6, "S11": 0.3}, 1.6),
    ("tle_on_large_input", "Correct but too slow on the largest inputs",
     {"S11": 1.0, "S5": 0.6, "S12": 0.3}, 1.2),
    ("inverted_condition", "Branch condition inverted",
     {"S7": 1.0, "S9": 0.6, "S4": 0.3}, 1.6),
]

BUG_IDS = [b[0] for b in BUG_CLASSES]
B = len(BUG_CLASSES)

Q = np.zeros((B, K))
for i, (_, _, loads, _) in enumerate(BUG_CLASSES):
    for s, v in loads.items():
        Q[i, STAGE_IDS.index(s)] = v
A = np.array([b[3] for b in BUG_CLASSES])          # a_beta
REQ = Q >= TAU_Q                                    # K(beta) as a mask [B, K]
assert REQ.any(axis=1).all(), "every class needs >=1 load-bearing stage"

# pre(k): direct prerequisites of each stage. Planning precedes coding;
# debugging/testing presuppose being able to write correct code.
PREREQS = {
    "S1": [], "S2": ["S1"], "S3": ["S2"], "S4": ["S3"], "S5": ["S3"],
    "S6": ["S4", "S5"], "S7": ["S3"], "S8": [], "S9": ["S7", "S8"],
    "S10": ["S9"], "S11": ["S2", "S9"], "S12": ["S11"],
}
PRE = np.zeros((K, K), dtype=bool)                  # PRE[k, j]: j in pre(k)
for s, ps in PREREQS.items():
    for p in ps:
        PRE[STAGE_IDS.index(s), STAGE_IDS.index(p)] = True
