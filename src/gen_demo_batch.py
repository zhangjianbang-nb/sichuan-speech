"""Generate a listenable batch of Sichuan speech from the live service, then verify each
clip with BOTH ASRs (dialect fine-tune + stock Mandarin) and compute CER.

    python3 work/gen_demo_batch.py [out_dir] [n_samples]

Writes <out_dir>/NN_<voice>.wav, a concatenated <out_dir>/all.wav for one-shot listening,
and <out_dir>/report.json / report.md.
"""
import sys, os, io, json, re, wave, uuid, urllib.request

BASE = "http://192.168.3.178:8100"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
os.makedirs(OUT, exist_ok=True)

# (voice, text) -- Sichuan colloquial first, Mandarin contrast last
CASES = [
    ("sichuan", "今天天气好安逸哦，我们出去耍一哈嘛。"),
    ("sichuan", "你吃饭没有得？没得的话我们一起切吃火锅。"),
    ("sichuan", "这个事情我晓得，你莫着急，慢慢来。"),
    ("sichuan", "老板儿，来二两小面，多放点海椒。"),
    ("sichuan", "来嘛，坐下喝口茶，慢慢摆龙门阵。"),
    ("sichuan", "他屋头那个猫儿又跑到房顶上去了。"),
    ("sichuan", "我跟你说哈，昨天我在春熙路碰到个老朋友，摆了半天龙门阵，巴适得很。"),
    ("sichuan", "这个娃儿成绩好得很，考起了川大，他爸妈高兴惨了。"),
    ("sichuan_avg", "莫慌，等我把这个事情搞归一了再说。"),
    ("sichuan_avg", "天都快黑球了，你还不搞快点儿收摊摊？"),
    ("sichuan_zeroshot", "走嘛，周末我们切青城山耍，顺便吃碗担担面。"),
    ("mandarin", "今天天气很好，我们出去走一走吧。"),
]


def post_form(path, fields, timeout=600):
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


def post_file(path, filename, data, timeout=180, model="ft"):
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


rows = []
clips = []
for i, (voice, text) in enumerate(CASES, 1):
    name = "%02d_%s" % (i, voice)
    data, hdr = post_form("/tts", {"text": text, "voice": voice, "n_samples": str(NS)})
    p = os.path.join(OUT, name + ".wav")
    with open(p, "wb") as f:
        f.write(data)
    hyp_ft = post_file("/asr", name + ".wav", data).get("text", "")
    hyp_base = post_file("/asr", name + ".wav", data, model="base").get("text", "")
    c_ft, c_base = cer(text, hyp_ft), cer(text, hyp_base)
    rows.append({"file": name + ".wav", "voice": voice, "text": text,
                 "asr_ft": hyp_ft, "asr_base": hyp_base,
                 "cer_ft": round(c_ft, 4), "cer_base": round(c_base, 4),
                 "dur": float(hdr.get("x-duration", 0)), "rtf": float(hdr.get("x-rtf", 0)),
                 "best_cer": hdr.get("x-best-cer")})
    clips.append(p)
    print("%-22s dur=%.2fs rtf=%4.1f cer_ft=%.3f cer_base=%.3f\n   ft  : %s\n   base: %s"
          % (name, float(hdr.get("x-duration", 0)), float(hdr.get("x-rtf", 0)), c_ft, c_base,
             hyp_ft, hyp_base))

# concatenate with 0.35 s silence for one-shot listening
frames, rate = [], None
for p in clips:
    with wave.open(p, "rb") as w:
        if rate is None:
            rate = w.getframerate()
        frames.append(w.readframes(w.getnframes()))
        frames.append(b"\x00\x00" * int(rate * 0.35))
with wave.open(os.path.join(OUT, "all.wav"), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    for f in frames:
        w.writeframes(f)

n = len(rows)
m_ft = sum(r["cer_ft"] for r in rows) / n
m_base = sum(r["cer_base"] for r in rows) / n
sw = [r for r in rows if r["voice"].startswith("sichuan")]
print("\nn=%d  mean cer_ft=%.4f  mean cer_base=%.4f" % (n, m_ft, m_base))
print("Sichuan-voice clips: cer_ft=%.4f  cer_base=%.4f  exact(ft)=%d/%d"
      % (sum(r["cer_ft"] for r in sw) / len(sw), sum(r["cer_base"] for r in sw) / len(sw),
         sum(1 for r in sw if r["cer_ft"] == 0), len(sw)))

with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump({"n_samples": NS, "mean_cer_ft": round(m_ft, 4),
               "mean_cer_base": round(m_base, 4), "rows": rows},
              f, ensure_ascii=False, indent=1)

with open(os.path.join(OUT, "report.md"), "w") as f:
    f.write("# 四川话批量合成（%d 句，best-of-%d）\n\n" % (n, NS))
    f.write("全部由 `http://192.168.3.178:8100` 现场合成；每句合成 %d 个候选，"
            "按川话 ASR 回读 CER 取最好的一个。每段再用两个识别器回读：\n"
            "`ft` = 川话微调 SenseVoice，`base` = 原版 SenseVoiceSmall。\n\n" % NS)
    f.write("| 文件 | 声音 | 文本 | 川话回读 | 普通话回读 | CER(ft) | CER(base) |\n"
            "|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write("| `%s` | %s | %s | %s | %s | %s | %s |\n"
                % (r["file"], r["voice"], r["text"], r["asr_ft"], r["asr_base"],
                   r["cer_ft"], r["cer_base"]))
    f.write("\n平均 CER：川话识别器 %.4f，普通话识别器 %.4f。\n" % (m_ft, m_base))
    f.write("\n两个识别器都 0 错 = 模型确实读对了；只有川话识别器 0 错 = 发音是方言读法，"
            "普通话识别器按普通话听写才出错（这是口音的正常表现，不是缺陷）。\n")
