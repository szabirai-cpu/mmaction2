"""Offline check of the notebook's CoT prompt/parse/generate logic with stubs.

torch, numpy, transformers etc. are not installed here, so they are faked. This only
exercises the pure-Python control flow: prompt assembly, marker parsing, the
truncation retry and the fallbacks.
"""
import json, sys, types

# ---- stub the heavy deps ----
torch = types.ModuleType("torch")


class _OOM(Exception):
    pass


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
torch.OutOfMemoryError = _OOM
torch.cuda.OutOfMemoryError = _OOM
torch.no_grad = _NoGrad
sys.modules["torch"] = torch

np = types.ModuleType("numpy")
np.array = lambda x, dtype=None: x
sys.modules["numpy"] = np

nb = json.load(open("projectupdate_2.ipynb", encoding="utf-8"))
src = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"])}

G = {"__name__": "nbtest", "torch": torch, "np": np}

# ---- config knobs the prompt cell needs ----
G.update(USE_COT=True, CONCISE_FINAL_ANSWER=True, TOP_K=6,
         MAX_CHARS_PER_CHUNK=2000, MAX_PROMPT_TOKENS=8192,
         MAX_NEW_TOKENS=1024, COT_RETRY_NEW_TOKENS=1536,
         SUPPORTS_SYSTEM_ROLE=True)

exec(src[8], G)                      # prompts + split_cot
split_cot = G["split_cot"]

FINAL = "الإجابة النهائية:"
fails = 0


def case(name, raw, want_answer, want_ok, want_reason, truncated=False):
    """Check one generation splits into the expected scored answer."""
    global fails
    reasoning, answer, ok, reason = split_cot(raw, truncated)
    bad = (answer != want_answer) or (ok != want_ok) or (reason != want_reason)
    fails += bad
    print(("FAIL " if bad else "ok   ") + f"{name:36} {reason:26} {answer[:44]!r}")
    if bad:
        print(f"       expected ok={want_ok} reason={want_reason} answer={want_answer!r}")
    # These must hold in every case: no heading text and no step numbering may ever
    # leak into the string that gets scored.
    for leak in ("الإجابة النهائية", "التفكير", "الخلاصة:"):
        assert leak not in answer, f"{name}: {leak!r} leaked into the scored answer"


# --- the format we asked for, and the variants Gemma actually produces ---
case("clean headings",
     "التفكير:\n١. السؤال عن البسملة.\n\n" + FINAL + "\nالبسملة هي قول: بسم الله.",
     "البسملة هي قول: بسم الله.", True, "ok")
case("markdown bold",
     "**التفكير:**\n- المقطع ٢ يذكر الآية.\n\n**الإجابة النهائية:**\nالجواب هنا.",
     "الجواب هنا.", True, "ok")
case("markdown heading",
     "## التفكير\nنص\n\n## الإجابة النهائية\nالجواب الثاني.",
     "الجواب الثاني.", True, "ok")
case("hamza-less spelling",
     "التفكير:\nنص\n\nالاجابة النهائية : جواب ثالث.",
     "جواب ثالث.", True, "ok")
case("english heading",
     "Reasoning:\nstep one\n\nFinal Answer: هذه هي الإجابة.",
     "هذه هي الإجابة.", True, "ok")
case("heading restated in reasoning",
     "التفكير:\nسأكتب الإجابة النهائية: لاحقًا.\n\n" + FINAL + "\nالجواب الصحيح.",
     "الجواب الصحيح.", True, "ok")

# --- the three real defects found in the n=20 run ---
case("shortened to 'الإجابة:'",
     "التفكير:\n١. نص\n\n**الإجابة:**\nالجواب الصحيح هنا.",
     "الجواب الصحيح هنا.", True, "short_heading")
case("shortened to 'الجواب:'",
     "التفكير:\n١. نص\n\nالجواب:\nالجواب الصحيح هنا.",
     "الجواب الصحيح هنا.", True, "short_heading")
case("shortened to 'الخلاصة:'",
     "التفكير:\n١. نص\n\nالخلاصة:\nالجواب الصحيح هنا.",
     "الجواب الصحيح هنا.", True, "short_heading")
case("no headings, plain answer kept whole",
     "البسملة هي قول العبد بسم الله.\n\nوهي مشروعة عند قراءة كل سورة.",
     "البسملة هي قول العبد بسم الله.\n\nوهي مشروعة عند قراءة كل سورة.",
     True, "answered_without_headings")

# --- genuine failures, correctly flagged ---
case("steps but no headings",
     "١. السؤال عن الآية.\n٢. المقطع ٣ يذكرها.\n\nالجواب هو كذا.",
     "الجواب هو كذا.", False, "steps_without_headings")
case("cut off mid-reasoning",
     "التفكير:\n١. السؤال عن الآية.\n\n٢. المقطع ٣ يذكر أن",
     "٢. المقطع ٣ يذكر أن", False, "ran_out_of_room", truncated=True)
case("finished with no answer section",
     "التفكير:\n١. السؤال عن الآية.\n\n٢. المقطع ٣ يذكر ذلك.",
     "٢. المقطع ٣ يذكر ذلك.", False, "no_answer_section")
case("heading then nothing",
     "التفكير:\nنص التفكير\n\n" + FINAL,
     "", False, "cut_off_at_heading")

# A sentence mentioning الجواب must NOT be mistaken for a heading (colon required).
case("word 'الجواب' inside prose",
     "التفكير:\nنص\n\n" + FINAL + "\nالجواب على ذلك مذكور في الآية.",
     "الجواب على ذلك مذكور في الآية.", True, "ok")

# ---- generation control flow ----
calls = []


class FakeIds(list):
    """len() is the real token count (that is what _fit checks); .shape[1] is pinned
    to 0 so the fake generation string is not sliced away in _run."""

    @property
    def shape(self):
        return (1, 0)


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTok:
    def __call__(self, text, return_tensors=None):
        return FakeBatch(input_ids=FakeIds([0] * (len(text) // 4)))

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "\n".join(m["content"] for m in msgs)

    def decode(self, ids, skip_special_tokens=True):
        return ids


class FakeModel:
    """scripted: list of (text, n_tokens). n_tokens >= the cap means 'was truncated'."""
    device = "cpu"

    def __init__(self, scripted):
        self.scripted = list(scripted)

    def generate(self, **kw):
        cap = kw["max_new_tokens"]
        calls.append(cap)
        out, n = self.scripted.pop(0)
        if out == "OOM":
            raise _OOM("fake")
        return [FakeStr(out, n if n is not None else cap)]


class FakeStr(str):
    """Stands in for a token tensor: slicing returns the text, len() the token count."""

    def __new__(cls, text, n):
        s = super().__new__(cls, text)
        s.n = n
        return s

    def __getitem__(self, k):
        return self if isinstance(k, slice) else str.__getitem__(self, k)

    def __len__(self):
        return self.n


G["tokenizer"] = FakeTok()
gen_src = src[9].split("direct_questions = ")[0]      # drop the live smoke test
exec(gen_src, G)

chunk = {"chunk_text": "س" * 400, "source": "x", "surah_name": "y",
         "surah_number": 1, "start_ayah": 1, "end_ayah": 1}
top = [dict(chunk) for _ in range(6)]
GOOD = "التفكير:\nخطوة\n\n" + FINAL + "\nجواب."


def check(name, cond, extra=""):
    global fails
    fails += not cond
    print(("FAIL " if not cond else "ok   ") + name + (f"  {extra}" if extra else ""))


# 1) clean CoT answer, one generate call
G["model"] = FakeModel([(GOOD, 200)])
calls.clear()
res = G["generate"](top, "سؤال؟", cot=True)
check("generate: clean CoT, single call",
      res["prediction"] == "جواب." and res["parsed_ok"] and calls == [1024], str(calls))

# 2) truncated (hit the cap) -> retried once with the bigger budget
G["model"] = FakeModel([("التفكير:\nمقطوع", 1024), (GOOD, 300)])
calls.clear()
res = G["generate"](top, "سؤال؟", cot=True)
check("generate: hit the cap -> retry at COT_RETRY_NEW_TOKENS",
      res["prediction"] == "جواب." and res["parsed_ok"] and calls == [1024, 1536],
      str(calls))

# 3) THE FIX: model stopped on its own with no heading -> NO retry.
#    Greedy decoding is deterministic, so a retry would return the same text.
G["model"] = FakeModel([("التفكير:\n١. نص\n\n٢. نص آخر.", 400), (GOOD, 300)])
calls.clear()
res = G["generate"](top, "سؤال؟", cot=True)
check("generate: stopped early -> no wasted retry",
      calls == [1024] and res["parse_reason"] == "no_answer_section", str(calls))

# 4) both attempts truncated -> fallback answer, flagged, never crashes
G["model"] = FakeModel([("التفكير:\nمقطوع أ", 1024), ("التفكير:\nمقطوع ب", 1536)])
calls.clear()
res = G["generate"](top, "سؤال؟", cot=True)
check("generate: double truncation falls back and flags it",
      bool(res["prediction"]) and not res["parsed_ok"], res["parse_reason"])

# 5) OOM halves the budget and succeeds
G["model"] = FakeModel([("OOM", None), (GOOD, 200)])
calls.clear()
res = G["generate"](top, "سؤال؟", cot=True)
check("generate: OOM halves max_new_tokens and retries",
      res["prediction"] == "جواب." and calls == [1024, 512], str(calls))

# 6) baseline mode returns the raw text untouched and never parses
G["model"] = FakeModel([("نص الإجابة كما هو.", 100)])
res = G["generate"](top, "سؤال؟", cot=False)
check("generate: baseline mode unchanged",
      res["prediction"] == "نص الإجابة كما هو." and res["reasoning"] == ""
      and res["parse_reason"] == "baseline")

# 7) prompts differ exactly as intended
cot_prompt = G["_assemble"](top, "سؤال؟", 2000, True)
base_prompt = G["_assemble"](top, "سؤال؟", 2000, False)
check("prompts: passages numbered 'i of n' only under CoT",
      "[المقطع 1 من 6]" in cot_prompt and "[المقطع 6 من 6]" in cot_prompt
      and "[المقطع" not in base_prompt)
check("prompts: paper constraints kept verbatim in both",
      G["SYSTEM_PROMPT"] in base_prompt and G["SYSTEM_PROMPT"] in cot_prompt
      and "التفكير:" not in base_prompt)
check("prompts: question is last so truncation cannot delete it",
      base_prompt.endswith("Question: سؤال؟\nAnswer:"))

# 8) oversized context is shrunk, question survives
G["MAX_PROMPT_TOKENS"] = 200
text, chars = G["_fit"]([{"chunk_text": "ط" * 9000}] * 6, "سؤال طويل؟", True)
check("_fit: context shrunk, question intact",
      text.endswith("Question: سؤال طويل؟\nAnswer:") and chars < 2000,
      f"chars={chars}")

print("\nFAILED" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
