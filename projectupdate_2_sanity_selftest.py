"""Check the sanity cell: it must confirm a complete corpus, measure grounding
honestly (Arabic spelling variants must not count as ungrounded), and catch each way
the run can be wrong."""
import io, json, sys, types
from collections import Counter
from contextlib import redirect_stdout

nb = json.load(open("projectupdate_2.ipynb", encoding="utf-8"))
def cell(marker):
    """Locate a cell by its header comment rather than its index -- inserting a cell
    must not silently make these tests exercise the wrong code."""
    hits = [s for s in ("".join(c["source"]) for c in nb["cells"]) if marker in s]
    assert len(hits) == 1, f"{marker!r} matched {len(hits)} cells"
    return hits[0]


SANITY = cell("SANITY CHECKS")

fails = 0


def check(name, cond, extra=""):
    global fails
    fails += not cond
    print(("FAIL " if not cond else "ok   ") + name + (f"  {extra}" if extra else ""))


BOOKS = ["aashoor", "alaloosi", "almawirdee", "almuyassar", "alrazi", "altasheel",
         "aysaraAltafasir", "fathAlqadeer", "fathaAlbayan", "katheer", "mukhtasar",
         "qurtubi", "saadi", "tabari", "zadAlmaseer"]


def make_chunks(n_books=15, total=55471):
    chunks, per = [], total // n_books
    for bi, b in enumerate(BOOKS[:n_books]):
        for i in range(per + (total % n_books if bi == 0 else 0)):
            chunks.append({"source": b, "surah_number": bi + 1, "start_ayah": i,
                           "end_ayah": i, "chunk_text": f"التفسير {b} آية {i} "
                                                        "البسمله هي قول العبد"})
    return chunks


def run(chunks, records, ntotal=None, corpus="all", n_eval=1000, n_shards=1):
    ns = {"chunks": chunks, "records": records, "Counter": Counter,
          "CORPUS": corpus, "N_EVAL_SAMPLES": n_eval, "N_SHARDS": n_shards,
          "sample": records,
          "index": types.SimpleNamespace(ntotal=len(chunks) if ntotal is None
                                         else ntotal)}
    with redirect_stdout(io.StringIO()) as out:
        exec(SANITY, ns)
    return out.getvalue(), ns["problems"]


full = make_chunks()
ids = [f"{c['source']}:{c['surah_number']}:{c['start_ayah']}-{c['end_ayah']}"
       for c in full[:6]]


def rec(q, pred, sources=ids, ref="مرجع"):
    return {"question": q, "prediction": pred, "reference": ref, "sources": sources}


# --- a healthy run ---
good = [rec(f"سؤال {i}", "البسملة هي قول العبد بسم الله", ids) for i in range(20)]
out, probs = run(full, good)
check("clean run reports no problems", not probs, str(probs))
check("prints the corpus size", "15 books, 55471 chunks" in out)
check("prints per-book counts", "aysaraAltafasir" in out)

# --- grounding must survive Arabic spelling differences ---
# 'البسملة' vs 'البسمله', diacritics, and alef variants all mean the same thing.
out, _ = run(full, [rec("س", "البِسْمَلَةُ هي قَولُ العَبْدِ", ids)])
pct = int(out.split("on average ")[1].split("%")[0])
check("diacritics and ta-marbuta do not count as ungrounded", pct >= 66, f"{pct}%")

# --- an invented answer must score low and be listed ---
out, _ = run(full, [rec("س", "كوكب المشتري يدور حول الشمس بسرعه كبيره جدا", ids)])
pct = int(out.split("on average ")[1].split("%")[0])
check("invented wording scores low", pct <= 34, f"{pct}%")
check("weakest answers are listed for review", "Weakest five" in out)

# --- each way the corpus can be wrong ---
_, probs = run(make_chunks(n_books=14, total=50000), good)
check("missing book is caught", any("14 books" in p for p in probs), str(probs))

_, probs = run(make_chunks(total=54181), good)
check("wrong chunk total is caught", any("54181 chunks" in p for p in probs))

_, probs = run(full, good, ntotal=55470)
check("index/chunks mismatch is caught",
      any("wrong passages" in p for p in probs), str(probs))

_, probs = run(full, [rec("س", "جواب", ["nonexistent:9:9-9"])])
check("untraceable retrieved passage is caught",
      any("not found in the current chunks" in p for p in probs))

# --- integrity of the answer set ---
_, probs = run(full, good + [good[0]])
check("duplicate question is caught", any("duplicated" in p for p in probs))

_, probs = run(full, [rec("س", "جواب", ids, ref="")])
check("missing reference is caught", any("no reference" in p for p in probs))

_, probs = run(full, [rec("س", "   ", ids)])
check("empty prediction is caught", any("empty prediction" in p for p in probs))

out, probs = run(full, good, n_eval=1000)
check("incomplete run is noted but not called a problem",
      not probs and "run is incomplete" in out)

# --- 'source' corpus must not be judged against the 15-book expectation ---
_, probs = run(make_chunks(n_books=2, total=7487), good, corpus="source")
check("source-only corpus is not flagged", not probs, str(probs))

print("\nFAILED" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
