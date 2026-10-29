"""Measure sampling variance: generate N candidates per sentence and report the CER
distribution plus the best-of-N CER. Used to decide whether observed misreads are
systematic or stochastic.

    python3 work/variance_probe.py [n]
"""
import sys, os, io, json, re, uuid, urllib.request

BASE = "http://192.168.3.178:8100"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "variance")
os.makedirs(OUT, exist_ok=True)

CASES = [
    ("sichuan", "老板儿，来二两小面，多放点海椒。"),
    ("sichuan", "他屋头那个猫儿又跑到房顶上去了。"),
    ("sichuan_avg", "天都快黑球了，你还不搞快点儿收摊摊？"),
    ("sichuan", "今天天气好安逸哦，我们出去耍一哈嘛。"),
    ("sichuan", "这个事情我晓得，你莫着急，慢慢来。"),
]


def post_form(path, fields, timeout=180):
    b = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (b, k, v)).encode())
    body.write(("--%s--\r\n" % b).encode())
    req = urllib.request.Request(BASE + path, data=body.getvalue(), method="POST",
                                 headers={"Content-Type": "multipart/form-data; boundary=" + b})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def post_file(path, filename, data, timeout=120):
    b = uuid.uuid4().hex
    body = io.BytesIO()
    body.write(("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
                "Content-Type: audio/wav\r\n\r\n" % (b, filename)).encode())
    body.write(data)
    body.write(("\r\n--%s--\r\n" % b).encode())
    req = urllib.request.Request(BASE + path, data=body.getvalue(), method="POST",
                                 headers={"Content-Type": "multipart/form-data; boundary=" + b})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def norm(t):
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


res = []
for ci, (voice, text) in enumerate(CASES, 1):
    cers, hyps = [], []
    for k in range(N):
        data = post_form("/tts", {"text": text, "voice": voice})
        p = os.path.join(OUT, "%d_%s_%d.wav" % (ci, voice, k))
        with open(p, "wb") as f:
            f.write(data)
        hyp = post_file("/asr", os.path.basename(p), data).get("text", "")
        hyps.append(hyp)
        cers.append(round(cer(text, hyp), 4))
    print("\n[%d] %s  %s" % (ci, voice, text))
    for k in range(N):
        print("   %d  cer=%.3f  %s" % (k, cers[k], hyps[k]))
    print("   -> best=%.3f  mean=%.3f  exact_hits=%d/%d"
          % (min(cers), sum(cers) / N, sum(1 for c in cers if c == 0), N))
    res.append({"text": text, "voice": voice, "cers": cers, "hyps": hyps,
                "best": min(cers), "mean": round(sum(cers) / N, 4)})

with open(os.path.join(OUT, "variance.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("\nmean-of-best = %.4f ; mean-of-mean = %.4f"
      % (sum(r["best"] for r in res) / len(res), sum(r["mean"] for r in res) / len(res)))
