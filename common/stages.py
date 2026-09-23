"""The 12-stage cognitive-flow taxonomy used as the latent skill space for the
proficiency-diagnosis experiment.

A student's latent state is a vector P: S1..S12 -> {1,2,3,4,5}. The simulated
student is *given* that vector and told to act it out; the tutor model has to
recover it from dialogue alone. Everything here is shared vocabulary between
the two sides: the tutor sees the stage names/descriptions and the level scale,
never a student's actual values.

The behavioral anchors matter more than the prose descriptions -- they are what
makes a level actable by the student model and inferable by the tutor. A level
that produces no observable difference in behavior is unmeasurable by
construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

LEVELS = [1, 2, 3, 4, 5]

LEVEL_DESCRIPTORS: Dict[int, str] = {
    1: "Absent. Does not perform this step at all, or performs it in a way that is "
       "actively wrong. Cannot do it even when prompted directly.",
    2: "Fragile. Attempts the step but usually gets it wrong. Needs heavy scaffolding; "
       "a direct hint is required before any progress.",
    3: "Inconsistent. Gets it right on familiar or simple cases, breaks down as soon as "
       "the problem varies. Can often self-correct after a pointed question.",
    4: "Solid. Usually correct. Minor slips only under time pressure or extra complexity, "
       "and notices most of them unprompted.",
    5: "Fluent. Reliably correct, self-checks without being asked, and can explain *why* "
       "the step is done that way.",
}


@dataclass(frozen=True)
class Stage:
    id: str
    name: str
    band: str  # "pre_coding" | "code_level"
    captures: str
    bug_class: str
    low_behavior: str   # what level 1-2 looks like in a transcript
    high_behavior: str  # what level 4-5 looks like in a transcript


STAGES: List[Stage] = [
    Stage(
        id="S1", name="Language Understanding", band="pre_coding",
        captures="Parse the statement, identify inputs/outputs and constraints.",
        bug_class="Inclusive bound read as exclusive.",
        low_behavior="Misreads or skips constraints; restates the problem as something "
                     "slightly different from what was asked; misses what the return value is.",
        high_behavior="Restates the problem precisely, names the input/output types and the "
                      "constraint bounds, and flags ambiguity in the wording.",
    ),
    Stage(
        id="S2", name="Problem Decomposition", band="pre_coding",
        captures="Split the problem into sub-problems and enumerate edge cases.",
        bug_class="Missing branch for empty input.",
        low_behavior="Treats the problem as one undifferentiated blob; lists no edge cases, "
                     "or only the ones already given in the examples.",
        high_behavior="Breaks the task into named sub-steps and volunteers edge cases "
                      "(empty, single element, duplicates, negatives) before being asked.",
    ),
    Stage(
        id="S3", name="Algorithm Formation", band="pre_coding",
        captures="Devise a solution strategy.",
        bug_class="Greedy choice where DP is required.",
        low_behavior="Has no strategy beyond 'loop through it'; jumps straight to code; "
                     "pattern-matches to a memorized template that does not fit.",
        high_behavior="Proposes a concrete strategy, can name it, and can sketch an "
                      "alternative approach when asked.",
    ),
    Stage(
        id="S4", name="Algorithm Correctness", band="pre_coding",
        captures="Verify that the chosen approach is actually sound.",
        bug_class="Recursion with an incorrect base case.",
        low_behavior="Asserts the approach works with no justification; cannot say what "
                     "would break it; accepts a counterexample without updating the plan.",
        high_behavior="Argues why the approach is correct (invariant, base case, exhaustive "
                      "cases) and engages properly with a proposed counterexample.",
    ),
    Stage(
        id="S5", name="Complexity Analysis", band="pre_coding",
        captures="Reason about time and space cost.",
        bug_class="Nested scan in place of a hash lookup.",
        low_behavior="Guesses big-O, or answers in vague terms ('fast', 'not too slow'); "
                     "miscounts nested loops; ignores space entirely.",
        high_behavior="States time and space complexity correctly, explains where the cost "
                      "comes from, and relates it to the input bounds in the statement.",
    ),
    Stage(
        id="S6", name="Approach Finalization", band="pre_coding",
        captures="Pick data structures and commit to a concrete plan.",
        bug_class="List used for membership tests.",
        low_behavior="Default to lists for everything; cannot justify the data-structure "
                     "choice; plan is still vague when coding starts.",
        high_behavior="Chooses data structures deliberately with a stated reason, and the "
                      "final plan is specific enough to code from directly.",
    ),
    Stage(
        id="S7", name="Code Translation", band="code_level",
        captures="Turn the stated algorithm into code.",
        bug_class="Planned step omitted in the implementation.",
        low_behavior="Code drifts away from the stated plan; steps described out loud never "
                     "appear in the code; structure does not match the sketch.",
        high_behavior="Code is a faithful, step-for-step realization of the stated plan.",
    ),
    Stage(
        id="S8", name="Syntax Proficiency", band="code_level",
        captures="Use language constructs correctly.",
        bug_class="Integer division used for a real quantity.",
        low_behavior="Frequent syntax slips; confuses language constructs (append vs +, "
                     "range semantics, / vs //, mutating while iterating).",
        high_behavior="Idiomatic, syntactically clean code; uses comprehensions, slicing and "
                      "standard library constructs correctly and knowingly.",
    ),
    Stage(
        id="S9", name="Code Correctness", band="code_level",
        captures="Code actually matches the intended logic.",
        bug_class="`<` written in place of `<=`.",
        low_behavior="Off-by-one errors, wrong variable used, conditions inverted; the code "
                     "does not do what the student says it does.",
        high_behavior="Code matches intent line by line; boundary conditions are right the "
                      "first time.",
    ),
    Stage(
        id="S10", name="Debugging", band="code_level",
        captures="Spot and fix bugs on the fly.",
        bug_class="Shadowed variable producing a silent wrong value.",
        low_behavior="When told the code is wrong, rewrites at random or re-reads without a "
                     "hypothesis; cannot localize a failure to a line.",
        high_behavior="Forms a hypothesis, traces or prints to isolate the fault, and fixes "
                      "the root cause rather than the symptom.",
    ),
    Stage(
        id="S11", name="Test-Case Checking", band="code_level",
        captures="Validate with tests, including edge cases.",
        bug_class="Passes the visible tests, fails on duplicates.",
        low_behavior="Only runs the examples given in the problem; declares success once "
                     "those pass; cannot invent a new test case when asked.",
        high_behavior="Generates its own tests, deliberately targeting edge cases and "
                      "adversarial inputs, before claiming the solution works.",
    ),
    Stage(
        id="S12", name="Verification", band="code_level",
        captures="Confirm the output, then refine or optimize.",
        bug_class="Correct value returned with the wrong return type.",
        low_behavior="Never re-checks the output against what the problem asked for; misses "
                     "format/type mismatches; no interest in refining a working solution.",
        high_behavior="Checks output shape/type against the spec, and proposes a concrete "
                      "refinement or optimization afterwards.",
    ),
]

STAGES_BY_ID: Dict[str, Stage] = {s.id: s for s in STAGES}
STAGE_IDS: List[str] = [s.id for s in STAGES]
PRE_CODING_IDS: List[str] = [s.id for s in STAGES if s.band == "pre_coding"]
CODE_LEVEL_IDS: List[str] = [s.id for s in STAGES if s.band == "code_level"]

Profile = Dict[str, int]


def validate_profile(profile: Profile) -> None:
    """Raise if a profile isn't a complete S1..S12 -> 1..5 mapping."""
    missing = [s for s in STAGE_IDS if s not in profile]
    if missing:
        raise ValueError(f"profile missing stages: {missing}")
    bad = {k: v for k, v in profile.items() if k in STAGE_IDS and v not in LEVELS}
    if bad:
        raise ValueError(f"profile has out-of-range levels (must be 1-5): {bad}")


def rubric_for_tutor() -> str:
    """Stage list + level scale as shown to the tutor. Deliberately contains no
    per-level behavioral anchors for individual stages -- the tutor gets the
    same generic scale a human rater would, and has to do the mapping itself."""
    lines = ["Stages (the 12-step cognitive flow a student goes through while solving a problem):"]
    for s in STAGES:
        band = "pre-coding" if s.band == "pre_coding" else "code-level"
        lines.append(f"  {s.id} - {s.name} [{band}]: {s.captures} "
                     f"Representative failure: {s.bug_class}")
    lines.append("")
    lines.append("Proficiency scale (the same 1-5 scale applies to every stage):")
    for lvl in LEVELS:
        lines.append(f"  {lvl} = {LEVEL_DESCRIPTORS[lvl]}")
    return "\n".join(lines)


def persona_spec_for_student(profile: Profile) -> str:
    """The per-stage script the simulated student must act out. Includes the
    behavioral anchors so the latent vector is actually expressible in
    dialogue; without these the student model regresses to a generic
    'average student' and the latent signal disappears."""
    validate_profile(profile)
    lines = []
    for s in STAGES:
        lvl = profile[s.id]
        if lvl <= 2:
            anchor = s.low_behavior
        elif lvl >= 4:
            anchor = s.high_behavior
        else:
            anchor = (f"On simple or familiar cases: {s.high_behavior} "
                      f"As soon as the case is unfamiliar, this breaks down instead: "
                      f"{s.low_behavior}")
        lines.append(
            f"{s.id} - {s.name} => LEVEL {lvl} ({LEVEL_DESCRIPTORS[lvl].split('.')[0]}).\n"
            f"    This step covers: {s.captures}\n"
            f"    This student: {anchor}"
        )
    return "\n".join(lines)
