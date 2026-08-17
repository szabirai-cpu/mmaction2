"""Simulate losing power at the worst possible moments and check nothing is lost.

Runs the notebook's real save_answers / load_answers code against a temp directory.
"""
import io, json, os, sys, tempfile
from contextlib import redirect_stdout

nb = json.load(open("projectupdate_2.ipynb", encoding="utf-8"))
def cell(marker):
    """Locate a cell by its header comment rather than its index -- inserting a cell
    must not silently make these tests exercise the wrong code."""
    hits = [s for s in ("".join(c["source"]) for c in nb["cells"]) if marker in s]
    assert len(hits) == 1, f"{marker!r} matched {len(hits)} cells"
    return hits[0]


ANSWER = cell("ANSWER THE QUESTIONS")

fails = 0


def check(name, cond, extra=""):
    global fails
    fails += not cond
    print(("FAIL " if not cond else "ok   ") + name + (f"  {extra}" if extra else ""))


def fresh():
    """Load just the save/load helpers, pointed at a temp file."""
    body = ANSWER.split("done = {r[")[0].split("def save_answers")[1]
    body = "def save_answers" + body
    tmpdir = tempfile.mkdtemp()
    ns = {"os": os, "json": json, "path": os.path.join(tmpdir, "preds.json")}
    exec(body, ns)
    return ns


def recs(n):
    return [{"question": f"q{i}", "prediction": f"p{i}"} for i in range(n)]


# --- normal round trip ---
ns = fresh()
ns["save_answers"](recs(5))
check("saves and reloads", len(ns["load_answers"]()) == 5)

# --- 700 answers saved, then power dies while writing answer 701 ---
ns = fresh()
path = ns["path"]
ns["save_answers"](recs(700))
ns["save_answers"](recs(701))          # rotates 700 -> .bak, writes 701
with open(path, "r+b") as f:           # truncate mid-file = what a cut looks like
    data = f.read()
    f.seek(0)
    f.truncate()
    f.write(data[:len(data) // 3])

with redirect_stdout(io.StringIO()) as out:
    kept = ns["load_answers"]()
check("power cut mid-save keeps the previous 700", len(kept) == 700, f"{len(kept)}")
check("and says it recovered from the backup", "recovered" in out.getvalue())

# --- the temp file must never be mistaken for the real one ---
ns = fresh()
ns["save_answers"](recs(3))
check("no .tmp file left behind", not os.path.exists(ns["path"] + ".tmp"))

# --- power dies during the very first save, no backup exists yet ---
ns = fresh()
open(ns["path"], "w").write('[{"question": "q0",')      # truncated JSON
with redirect_stdout(io.StringIO()) as out:
    kept = ns["load_answers"]()
check("unreadable file with no backup starts clean instead of crashing", kept == [])
check("and warns rather than failing silently", "unreadable" in out.getvalue())

# --- nothing saved yet ---
ns = fresh()
check("missing file is not an error", ns["load_answers"]() == [])

# --- repeated saves: after every one, at least one file must be readable ---
def readable(p):
    try:
        json.load(open(p, encoding="utf-8"))
        return True
    except Exception:
        return False


ns = fresh()
worst = None
for n in range(1, 40):
    ns["save_answers"](recs(n))
    if not any(os.path.exists(c) and readable(c)
               for c in (ns["path"], ns["path"] + ".bak")):
        worst = n
        break
check("every save leaves at least one readable file", worst is None,
      "" if worst is None else f"failed at n={worst}")

print("\nFAILED" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
