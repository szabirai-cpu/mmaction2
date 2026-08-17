"""Offline check of the two-member split: the halves must be disjoint, cover the whole
2000, and the merge cell must refuse anything that is not two halves of one draw.

Runs the notebook's own config, slicing and merge code -- no GPU, no model.
"""
import io, json, os, random, re, shutil, sys, tempfile
from collections import Counter
from contextlib import redirect_stdout

nb = json.load(open("projectupdate_2.ipynb", encoding="utf-8"))
def cell(marker):
    """Locate a cell by its header comment rather than its index -- inserting a cell
    must not silently make these tests exercise the wrong code."""
    hits = [s for s in ("".join(c["source"]) for c in nb["cells"]) if marker in s]
    assert len(hits) == 1, f"{marker!r} matched {len(hits)} cells"
    return hits[0]


CONFIG = cell("= CONFIG =")
ANSWER = cell("ANSWER THE QUESTIONS")
MERGE = cell("MERGE THE SHARDS")

fails = 0


def check(name, cond, extra=""):
    global fails
    fails += not cond
    print(("FAIL " if not cond else "ok   ") + name + (f"  {extra}" if extra else ""))


def _set(text, name, value):
    """Rewrite an assignment at the start of a line. Anchoring matters: the same
    'SHARD = 1' text also appears in the explanatory comment above it, and a plain
    str.replace would edit the comment and leave the real setting untouched."""
    new, n = re.subn(rf"^{name} = .*$", f"{name} = {value}", text, count=1, flags=re.M)
    assert n == 1, f"could not find an assignment for {name}"
    return new


def run_config(shard, n=2000, n_shards=2, seed=42, quick=False):
    """Execute the real config cell with one setting changed."""
    text = CONFIG
    for name, value in (("SHARD", shard), ("N_SHARDS", n_shards),
                        ("N_EVAL_SAMPLES", n), ("SEED", seed),
                        ("QUICK_TEST", quick)):
        text = _set(text, name, value)
    G = {}
    with redirect_stdout(io.StringIO()) as out:
        exec(text, G)
    return G, out.getvalue()


# ---- 1. the split itself ----
POOL = [{"Question": f"q{i}", "Answer": f"a{i}", "Task": "Direct"} for i in range(2350)]


def slice_for(shard):
    """Run the notebook's own sampling + slicing code for one member."""
    G, _ = run_config(shard)
    ns = dict(G)
    ns.update(direct_questions=POOL, RESULTS_DIR=tempfile.mkdtemp(),
              json=json, os=os, random=random)
    body = ANSWER.split("def save_answers")[0]   # sampling + slicing only
    body = body.replace("from tqdm.auto import tqdm", "")
    with redirect_stdout(io.StringIO()):
        exec(body, ns)
    return ns["full"], ns["sample"]


full1, mine = slice_for(1)
full2, theirs = slice_for(2)

check("both members draw the identical 2000", [q["Question"] for q in full1]
      == [q["Question"] for q in full2])
check("each half is 1000", len(mine) == 1000 and len(theirs) == 1000,
      f"{len(mine)} / {len(theirs)}")
qa, qb = {q["Question"] for q in mine}, {q["Question"] for q in theirs}
check("halves do not overlap", not (qa & qb), f"{len(qa & qb)} shared")
check("halves cover the whole 2000", qa | qb == {q["Question"] for q in full1},
      f"union={len(qa | qb)}")
check("shard 1 is the first half", [q["Question"] for q in mine]
      == [q["Question"] for q in full1[:1000]])

# The trap the config comment warns about. What random.sample(pool, 1000) returns
# relative to a 2000-draw is a CPython implementation detail -- on current CPython both
# sizes take the same sequential path, so it returns shard 1's questions exactly. The
# guarantee that matters either way: it never gives you the second half, so a teammate
# who sets N_EVAL_SAMPLES = 1000 duplicates work instead of sharing it.
random.seed(42)
naive = {q["Question"] for q in random.sample(POOL, 1000)}
check("sampling 1000 directly never yields the second half",
      naive != qb, f"overlap with shard 2: {len(naive & qb)}")
print(f"     (fyi on this Python it overlaps shard 1 by {len(naive & qa)}/1000 "
      f"-- which is why the config says to keep 2000 and change SHARD)")

# ---- 2. uneven splits and edge cases ----
G, _ = run_config(1)
sb = G["shard_bounds"]
check("odd total splits without loss",
      [sb(2001, s, 2) for s in (1, 2)] == [(0, 1001), (1001, 2001)])
check("three-way split is exact and gapless",
      [sb(2000, s, 3) for s in (1, 2, 3)] == [(0, 667), (667, 1334), (1334, 2000)])
check("single shard takes everything", sb(2000, 1, 1) == (0, 2000))

_, quick_out = run_config(2, quick=True)
check("QUICK_TEST disables sharding", "no sharding" in quick_out, quick_out.strip())

try:
    run_config(3, n_shards=2)
    check("SHARD out of range is rejected", False)
except ValueError:
    check("SHARD out of range is rejected", True)

G1, out1 = run_config(1)
check("config states which questions are yours",
      "questions 1-1000" in out1 and "SHARD = 2" in out1)
check("shard tag lands in the filenames", G1["SHARD_TAG"] == "_shard1of2")
check("hours estimate is for your half, not the full 2000",
      "1000 questions at roughly" in out1 and "GPU hours" in out1
      and "2000 questions at" not in out1)


# ---- 3. the merge cell ----
def fake_records(questions, shard, seed=42, n_total=2000, n_shards=2,
                 corpus="all", mode="cot", dup=None):
    recs = [{"question": q, "reference": "ref " + q, "prediction": "pred " + q,
             "parsed_ok": True, "parse_reason": "ok", "reasoning": "r", "raw": "x",
             "sources": [], "mode": mode, "seed": seed, "n_total": n_total,
             "shard": shard, "n_shards": n_shards, "corpus": corpus}
            for q in questions]
    if dup:
        recs.append(dict(recs[0], question=dup))
    return recs


def run_merge(shards, expect_scored=True):
    """shards: {shard_number: list_of_records}. Returns (stdout, scored_or_None)."""
    tmp = tempfile.mkdtemp()
    G, _ = run_config(1)
    stem = f"gemma-3-4b-it_all_cot_n{G['N_EVAL_SAMPLES']}_seed{G['SEED']}"
    for s, recs in shards.items():
        with open(os.path.join(tmp, f"preds_{stem}_shard{s}of2.json"),
                  "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False)

    scored = {}

    def fake_evaluate(recs, label, save_as=None):
        scored["n"] = len(recs)
        scored["label"] = label
        return {"n": len(recs)}

    ns = dict(G)
    ns.update(RESULTS_DIR=tmp, STEM=stem, json=json, os=os, glob=__import__("glob"),
              evaluate=fake_evaluate, model_tag="gemma-3-4b-it", Counter=Counter)
    with redirect_stdout(io.StringIO()) as out:
        exec(MERGE, ns)
    shutil.rmtree(tmp, ignore_errors=True)
    return out.getvalue(), scored


A = [q["Question"] for q in mine]
B = [q["Question"] for q in theirs]

out, scored = run_merge({1: fake_records(A, 1), 2: fake_records(B, 2)})
check("merge scores all 2000 together", scored.get("n") == 2000, str(scored.get("n")))
check("merged label says COMBINED", "COMBINED" in scored.get("label", ""))
check("merge reports the row to use", "This is the row to report." in out)

out, scored = run_merge({1: fake_records(A, 1)})
check("missing half blocks the merge", not scored and "missing" in out)
check("missing half names the file to copy",
      "shard2of2.json" in out and "run this cell again" in out)

out, scored = run_merge({1: fake_records(A, 1), 2: fake_records(B, 2, seed=7)})
check("different seed blocks the merge", not scored and "seed is 7" in out)

out, scored = run_merge({1: fake_records(A, 1),
                         2: fake_records(B, 2, n_total=1000)})
check("different N_EVAL_SAMPLES blocks the merge",
      not scored and "n_total is 1000" in out)

out, scored = run_merge({1: fake_records(A, 1),
                         2: fake_records(B, 2, corpus="source")})
check("different corpus blocks the merge", not scored and "corpus is 'source'" in out)

out, scored = run_merge({1: fake_records(A, 1), 2: fake_records(B, 2, dup=A[0])})
check("a question answered by both is caught",
      not scored and "both shard" in out)

# An incomplete half must still merge (so you can see progress) but must warn loudly.
out, scored = run_merge({1: fake_records(A[:400], 1), 2: fake_records(B, 2)})
check("incomplete half warns but still scores",
      scored.get("n") == 1400 and "expected 2000" in out, str(scored.get("n")))

print("\nFAILED" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
