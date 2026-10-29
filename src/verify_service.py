"""Acceptance check for the Sichuan speech service.

Run from any machine that can reach the service (default http://192.168.3.178:8100):

    python3 src/verify_service.py [base_url]

It synthesizes one Sichuan sentence with each voice, sends the audio back to /asr, and
prints the round-trip CER. Exit code 0 means every check passed.
"""
import sys, json, io, wave, urllib.request, urllib.error, uuid

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://192.168.3.178:8100").rstrip("/")

CASES = [
    ("sichuan", "今天天气好安逸哦，我们出去耍一哈嘛。"),
    ("sichuan_avg", "莫慌，等我把这个事情搞归一了再说。"),
    ("mandarin", "今天天气很好，我们出去走一走吧。"),
]


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


def post_form(path, fields, timeout=180):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                    % (boundary, k, v)).encode())
    body.write(("--%s--\r\n" % boundary).encode())
    req = urllib.request.Request(BASE + path, data=body.getvalue(), method="POST",
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), {k.lower(): v for k, v in r.headers.items()}


def post_file(path, filename, data, timeout=120, model="ft"):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    body.write(("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
                "Content-Type: audio/wav\r\n\r\n" % (boundary, filename)).encode())
    body.write(data)
    body.write(("--%s\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n%s\r\n"
                % (boundary, model)).encode())
    body.write(("--%s--\r\n" % boundary).encode())
    req = urllib.request.Request(BASE + path, data=body.getvalue(), method="POST",
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def norm(t):
    import re
    return re.sub(r"[\s，。、？！；：,.?!;:\"'“”‘’（）()《》<>·—\-]", "", t)


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(1, len(r))


ok = True
try:
    h = get("/health")
    print("health:", h)
    ok = ok and h.get("ok") and h.get("gpu") == "1"
except Exception as e:
    print("health FAILED:", repr(e))
    sys.exit(1)

v = get("/voices")
print("voices:", [x["id"] for x in v["voices"]])
ok = ok and {x["id"] for x in v["voices"]} >= {"sichuan", "mandarin"}

print("\n%-14s %-8s %-8s %-6s %-7s %s" % ("voice", "dur(s)", "rtf", "CER", "best", "transcript"))
for voice, text in CASES:
    try:
        wav, hdr = post_form("/tts", {"text": text, "voice": voice, "n_samples": "3"})
        assert wav[:4] == b"RIFF", "not a wav"
        asr = post_file("/asr", "t.wav", wav)
        base = post_file("/asr", "t.wav", wav, model="base")
        c = cer(text, asr["text"])
        print("%-14s %-8s %-8s %-6.3f %-7s %s" % (voice, hdr.get("x-duration", "?"),
                                                  hdr.get("x-rtf", "?"), c,
                                                  hdr.get("x-best-cer", "?"), asr["text"]))
        if voice in ("sichuan", "sichuan_avg") and c > 0.25:
            print("  ^ CER too high for the dialect voice")
            ok = False
        if cer(text, base["text"]) > 0.25:
            print("  ^ stock Mandarin ASR cannot follow this clip either (synthesis may be broken)")
            ok = False
    except Exception as e:
        print("%-14s FAILED %r" % (voice, e))
        ok = False

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
