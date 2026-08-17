"""Check the download retry logic: retries on transient errors, fails fast on
permanent ones, never leaves a partial file, and does not sleep for real in tests."""
import io, json, os, sys, tempfile, time
from contextlib import redirect_stdout
from unittest import mock

nb = json.load(open("projectupdate_2.ipynb", encoding="utf-8"))


def cell(marker):
    hits = [s for s in ("".join(c["source"]) for c in nb["cells"]) if marker in s]
    assert len(hits) == 1, f"{marker!r} matched {len(hits)} cells"
    return hits[0]


DATA = cell("DOWNLOAD THE DATA")
body = "import time\n\n\ndef fetch" + DATA.split("def fetch")[1].split("os.makedirs(\"books\"")[0]

fails = 0


def check(name, cond, extra=""):
    global fails
    fails += not cond
    print(("FAIL " if not cond else "ok   ") + name + (f"  {extra}" if extra else ""))


class FakeResp:
    def __init__(self, status, content=b"data"):
        self.status_code = status
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code), response=self)


def env():
    import requests
    tmpdir = tempfile.mkdtemp()
    ns = {"os": os, "requests": requests, "time": mock.Mock(sleep=lambda s: None)}
    exec(body, ns)
    return ns, os.path.join(tmpdir, "book.xlsx")


# 1) succeeds first try
ns, dest = env()
with mock.patch("requests.get", return_value=FakeResp(200, b"hello")):
    ns["fetch"]("http://x", dest)
check("saves the file on success", open(dest, "rb").read() == b"hello")
check("no leftover .tmp file", not os.path.exists(dest + ".tmp"))

# 2) 429 then success -- must retry, not fail
ns, dest = env()
calls = [FakeResp(429), FakeResp(429), FakeResp(200, b"ok")]
with mock.patch("requests.get", side_effect=lambda *a, **k: calls.pop(0)):
    with redirect_stdout(io.StringIO()) as out:
        ns["fetch"]("http://x", dest)
check("recovers after two 429s", open(dest, "rb").read() == b"ok")
check("prints that it is retrying", "retrying" in out.getvalue())

# 3) 500 is also treated as transient
ns, dest = env()
calls = [FakeResp(503), FakeResp(200, b"ok")]
with mock.patch("requests.get", side_effect=lambda *a, **k: calls.pop(0)):
    ns["fetch"]("http://x", dest)
check("503 is retried like 429", open(dest, "rb").read() == b"ok")

# 4) 404 must NOT retry -- it is never going to succeed
ns, dest = env()
attempts = []


def count_404(*a, **k):
    attempts.append(1)
    return FakeResp(404)


with mock.patch("requests.get", side_effect=count_404):
    try:
        ns["fetch"]("http://x", dest)
        check("404 raises instead of returning", False)
    except Exception:
        check("404 raises instead of returning", True)
check("404 fails on the first attempt, no wasted retries", len(attempts) == 1,
      f"{len(attempts)} attempts")
check("no file written after a 404", not os.path.exists(dest))

# 5) always 429 -- must give up eventually, not loop forever
ns, dest = env()
with mock.patch("requests.get", return_value=FakeResp(429)):
    try:
        ns["fetch"]("http://x", dest)
        check("gives up after repeated 429s", False)
    except Exception:
        check("gives up after repeated 429s", True)

# 6) a network exception (not an HTTP status) is also retried
ns, dest = env()
import requests as _requests
calls = [_requests.ConnectionError("dns fail"), FakeResp(200, b"ok")]


def flaky(*a, **k):
    c = calls.pop(0)
    if isinstance(c, Exception):
        raise c
    return c


with mock.patch("requests.get", side_effect=flaky):
    ns["fetch"]("http://x", dest)
check("connection errors are retried too", open(dest, "rb").read() == b"ok")

print("\nFAILED" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
