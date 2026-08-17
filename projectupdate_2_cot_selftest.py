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


torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
torch.OutOfMemoryError = _OOM
torch.cuda.OutOfMemoryError = _OOM
class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


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
         MAX_NEW_TOKENS=768, COT_RETRY_NEW_TOKENS=1152,
         SUPPORTS_SYSTEM_ROLE=True)

exec(src[8], G)                      # prompts + split_cot
split_cot = G["split_cot"]

FINAL = "الإجابة النهائية:"
cases = [
    ("plain",
     "التفكير:\n١. السؤال عن البسملة.\n٢. المقطع 1 يذكرها.\n\n"
     + FINAL + "\nالبسملة هي قول: بسم الله الرحمن الرحيم.",
     "البسملة هي قول: بسم الله الرحمن الرحيم.", True),
    ("markdown bold",
     "**التفكير:**\n- المقطع 2 يذكر الآية.\n\n**الإجابة النهائية:**\nالجواب هنا.",
     "الجواب هنا.", True),
    ("heading style",
     "## التفكير\nنص\n\n## الإجابة النهائية\nالجواب الثاني.",
     "الجواب الثاني.", True),
    ("hamza-less spelling",
     "التفكير:\nنص\n\nالاجابة النهائية : جواب ثالث.",
     "جواب ثالث.", True),
    ("english fallback marker",
     "Reasoning:\nstep one\n\nFinal Answer: هذه هي الإجابة.",
     "هذه هي الإجابة.", True),
    ("marker restated twice",
     "التفكير:\nسأكتب الإجابة النهائية: لاحقًا.\n\nالإجابة النهائية:\nالجواب الصحيح.",
     "الجواب الصحيح.", True),
    ("truncated mid-reasoning (no marker)",
     "التفكير:\n١. السؤال عن الآية.\n\n٢. المقطع 3 يذكر أن",
     "٢. المقطع 3 يذكر أن", False),
    ("marker with nothing after it",
     "التفكير:\nنص التفكير\n\nالإجابة النهائية:",
     "", False),
    ("no structure at all",
     "الجواب مباشرة بدون أي عناوين.",
     "الجواب مباشرة بدون أي عناوين.", False),
]

fails = 0
for name, raw, want_answer, want_ok in cases:
    reasoning, answer, ok = split_cot(raw)
    bad = (answer != want_answer) or (ok != want_ok)
    fails += bad
    print(("FAIL " if bad else "ok   ") + f"{name:34} ok={ok} answer={answer!r}")
    if bad:
        print(f"       expected ok={want_ok} answer={want_answer!r}")
    assert FINAL not in answer, f"{name}: marker leaked into the scored answer"
    assert "التفكير" not in answer, f"{name}: reasoning heading leaked into the answer"

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
    device = "cpu"

    def __init__(self, scripted):
        self.scripted = list(scripted)

    def generate(self, **kw):
        calls.append(kw["max_new_tokens"])
        out = self.scripted.pop(0)
        if out == "OOM":
            raise _OOM("fake")
        return [out]


G["tokenizer"] = FakeTok()
gen_src = src[9].split("direct_questions = ")[0]      # drop the live smoke test
exec(gen_src, G)

chunk = {"chunk_text": "س" * 400, "source": "x", "surah_name": "y",
         "surah_number": 1, "start_ayah": 1, "end_ayah": 1}
top = [dict(chunk) for _ in range(6)]

# 1) clean CoT answer, one generate call
G["model"] = FakeModel(["التفكير:\nخطوة\n\n" + FINAL + "\nجواب."])
calls.clear()
res = G["generate"](top, "سؤال؟", cot=True)
assert res["prediction"] == "جواب." and res["parsed_ok"] and calls == [768], (res, calls)
print("ok   generate: clean CoT, single call")

# 2) truncated first attempt -> retried once with the bigger budget
G["model"] = FakeModel(["التفكير:\nمقطوع",
                        "التفكير:\nخطوة\n\n" + FINAL + "\nجواب بعد الإعادة."])
calls.clear()
res = G["generate"](top, "سؤال؟", cot=True)
assert res["prediction"] == "جواب بعد الإعادة." and res["parsed_ok"], res
assert calls == [768, 1152], calls
print("ok   generate: truncation retry uses COT_RETRY_NEW_TOKENS")

# 3) both attempts truncated -> fallback answer, flagged, never crashes
G["model"] = FakeModel(["التفكير:\nمقطوع أ", "التفكير:\nمقطوع ب"])
calls.clear()
res = G["generate"](top, "سؤال؟", cot=True)
assert res["prediction"] and not res["parsed_ok"], res
print("ok   generate: double truncation falls back and flags parsed_ok=False")

# 4) OOM on the first try halves the budget and succeeds
G["model"] = FakeModel(["OOM", "التفكير:\nخطوة\n\n" + FINAL + "\nجواب."])
calls.clear()
res = G["generate"](top, "سؤال؟", cot=True)
assert res["prediction"] == "جواب." and calls == [768, 384], (res, calls)
print("ok   generate: OOM halves max_new_tokens and retries")

# 5) baseline mode returns the raw text untouched and never parses
G["model"] = FakeModel(["نص الإجابة كما هو."])
res = G["generate"](top, "سؤال؟", cot=False)
assert res["prediction"] == "نص الإجابة كما هو." and res["reasoning"] == ""
print("ok   generate: baseline mode unchanged")

# 6) prompts differ exactly as intended
cot_prompt = G["_assemble"](top, "سؤال؟", 2000, True)
base_prompt = G["_assemble"](top, "سؤال؟", 2000, False)
assert "[المقطع 1]" in cot_prompt and "[المقطع 6]" in cot_prompt
assert "[المقطع" not in base_prompt
assert G["SYSTEM_PROMPT"] in base_prompt and "التفكير:" not in base_prompt
assert G["SYSTEM_PROMPT"] in cot_prompt          # baseline constraints kept verbatim
assert base_prompt.endswith("Question: سؤال؟\nAnswer:")
print("ok   prompts: numbering only in CoT, paper constraints kept in both")

# 7) oversized context is shrunk, question survives
G["MAX_PROMPT_TOKENS"] = 200
text, chars = G["_fit"]([{"chunk_text": "ط" * 9000}] * 6, "سؤال طويل؟", True)
assert text.endswith("Question: سؤال طويل؟\nAnswer:") and chars < 2000
print(f"ok   _fit: context shrunk to {chars} chars, question intact")

print("\nFAILED" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
