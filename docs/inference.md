# 推理与部署

一个 FastAPI 进程同时提供合成与识别，前端页面随包发布，不依赖任何 CDN。

## 1. 安装

```bash
# 只跑推理（服务端）
pip install "sichuan-speech[serve]"      # numpy + soundfile + fastapi + uvicorn

# 需要客户端 / 开发
pip install "sichuan-speech[serve,dev]"
```

CosyVoice 源码不在 wheel 里，需要单独放好并把 `SICHUAN_COSYVOICE_HOME` 指过去。
模型目录与数据目录同样通过环境变量给出（见 §2）。

## 2. 配置

所有配置都有默认值，全部可用环境变量覆盖。`sichuan-speech info` 打印解析后的结果，
并标出哪些路径不存在。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SICHUAN_GPU`（或 `CUDA_VISIBLE_DEVICES`） | `1` | 用哪张卡；GPU0 留给 ComfyUI |
| `SICHUAN_HOST` / `SICHUAN_PORT` | `0.0.0.0` / `8100` | 监听地址与端口 |
| `SICHUAN_LOG_LEVEL` | `INFO` | 日志级别 |
| `SICHUAN_COSYVOICE_HOME` | `~/CosyVoice` | CosyVoice 源码根目录 |
| `SICHUAN_BASE_DIR`（或 `BASE_DIR`） | `~/models/CosyVoice2-0.5B` | 原版模型 |
| `SICHUAN_SFT_DIR`（或 `SFT_DIR`） | `~/chuan/models/cosyvoice2-sichuan2` | 微调模型 |
| `SICHUAN_ASR_BASE_DIR` | `~/models/SenseVoiceSmall` | 原版 ASR |
| `SICHUAN_ASR_CKPT`（或 `ASR_CKPT`） | `~/chuan/models/sensevoice-sichuan.pt` | 微调 ASR |
| `SICHUAN_DATA_ROOT` | `~/chuan/data` | 数据根目录 |
| `SICHUAN_WORK_ROOT` | `~/chuan` | 实验/配置根目录 |
| `SICHUAN_EVAL_ROOT` | `~/chuan/eval` | 评测输出 |
| `SICHUAN_PROMPT_WAV` / `_TEXT` | MagicHub 成都女声 | 川话提示音（零样本克隆用） |
| `SICHUAN_OFFICIAL_PROMPT_WAV` / `_TEXT` | CosyVoice 自带 | 普通话提示音 |
| `SICHUAN_TTS_FP16` | `1` | 合成用 fp16 |
| `SICHUAN_ASR_DEVICE` | `cpu` | ASR 跑在 CPU 上，省显存 |
| `SICHUAN_MAX_TEXT_CHARS` | `500` | 单次合成文本上限 |
| `SICHUAN_MAX_UPLOAD_MB` | `20` | 上传音频上限 |
| `SICHUAN_DEFAULT_N_SAMPLES` / `SICHUAN_MAX_N_SAMPLES` | `5` / `5` | best-of-N 默认与上限 |
| `SICHUAN_SPEED_MIN` / `SICHUAN_SPEED_MAX` | `0.5` / `2.0` | 语速范围 |
| `SICHUAN_REQUEST_TIMEOUT` | `180` | 客户端超时 |
| `SICHUAN_CORS_ORIGINS` | `*` | 逗号分隔 |

## 3. 启动

```bash
sichuan-speech serve                       # 0.0.0.0:8100, GPU 1
sichuan-speech serve --port 8200 --gpu 1 --log-level DEBUG
```

打开 `http://<host>:8100/` 就是界面。首次请求会加载模型（几十秒），之后常驻显存。

### 界面能做什么

- **合成**：输入文本、选声音（4 个）、调语速、选候选条数（1–5）、生成；
  播放器 + 波形 + 时长/采样率/候选数/回读字错率等徽标；一键下载 WAV。
- **回读校验**：把刚合成的音频送回方言识别器，前端自己算字错率，和参考文本并排显示。
- **识别**：拖拽/选择文件，或浏览器录音（自动转成 16-bit PCM WAV）；可选只跑微调版、
  或两个模型都跑一遍做对照；结果可复制。
- **状态**：右上角状态胶囊读 `/health`，显示版本与 GPU。

界面不引用任何外部资源（无 CDN、无外部字体），断网也能用。

## 4. HTTP 接口

| 方法 | 路径 | 入参 | 返回 |
|---|---|---|---|
| GET | `/health` | — | `{ok, version, gpu, sft_dir, asr_ckpt, voices, loaded}` |
| GET | `/voices` | — | `{default, voices:[{id,label,description,engine,available,reason}]}` |
| GET | `/api` | — | 端点清单 + 限制 + 解析后的配置 |
| POST | `/tts` | form: `text, voice, speed, n_samples` | `audio/wav`，响应头带 `X-Duration / X-RTF / X-Samples / X-Best-Cer / X-Voice / X-Sample-Rate` |
| POST | `/asr` | multipart: `file, model=ft|base` | `{text, model, elapsed_s}` |
| GET | `/` `/static/*` | — | 前端页面与静态资源 |

错误约定：参数问题 400（`{"detail": "..."}`），音频解码失败 415，引擎异常 503。

`n_samples > 1` 时服务端渲染 N 条候选，用方言识别器逐条回读，取 CER 最低的一条；
遇到 CER=0 立即停止。`X-Samples` 返回**实际渲染条数**（提前停止时会小于请求值）。

## 5. 命令行

```bash
sichuan-speech info                       # 配置与路径体检
sichuan-speech voices                     # 本地声音注册表
sichuan-speech voices --url http://127.0.0.1:8100
sichuan-speech tts "今天天气好安逸哦" --voice sichuan --out demo.wav --n-samples 5
sichuan-speech asr demo.wav --model ft
sichuan-speech verify                     # 端到端验收，3 个声音各一句
sichuan-speech batch demo/batch --n-samples 3   # 12 句试听包 + report.json/md
sichuan-speech variance 6                 # 采样方差与 best-of-N 实测
```

`verify` 的判定：方言声音的川话 ASR CER ≤ `--max-cer`（默认 0.25），普通话声音的普通话
CER ≤ 同值；GPU 不符只记 warning，不判失败。退出码 0 通过、1 有失败项、2 连不上服务。

## 6. Python 客户端

```python
from sichuan_speech import SpeechClient

client = SpeechClient("http://192.168.3.178:8100", timeout=180)
print(client.health())

result = client.tts("老板儿，来二两小面。", voice="sichuan", n_samples=5)
result.save("noodle.wav")
print(result.duration_s, result.rtf, result.best_cer)

print(client.asr(open("noodle.wav", "rb").read(), "noodle.wav", model="ft").text)
```

客户端只用标准库 `urllib`，没有额外依赖；`SpeechApiError` 带 `status` 与服务端 `detail`。

## 7. 部署为 systemd 用户服务

仓库自带 `deploy/sichuan-audio.service`：

```ini
[Unit]
Description=Sichuan speech API (TTS + ASR)
After=network-online.target

[Service]
Type=simple
Environment=SICHUAN_GPU=1
Environment=SICHUAN_PORT=8100
ExecStart=%h/audio/venv/bin/sichuan-speech serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
mkdir -p ~/.config/systemd/user
cp deploy/sichuan-audio.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now sichuan-audio.service
loginctl enable-linger "$USER"       # 退出登录后仍运行
```

## 8. 排障

| 症状 | 先查什么 |
|---|---|
| 语音是乱码/无意义音节 | transformers 版本；是否打了 `scripts/patch_transformers5.sh` |
| `KeyError: 'embedding'` | 微调模型目录的 `spk2info.pt` 是否用 `build_sft_model.py` 生成 |
| 首次请求很慢、之后正常 | 正常：模型懒加载，首次几十秒 |
| 503 `TTS failed: CUDA out of memory` | GPU 被别的进程占着；`nvidia-smi` 看显存 |
| 上传 mp3 报 415 | 装 `ffmpeg`；`which ffmpeg` |
| 声音列表全是「不可用」 | `sichuan-speech info` 看哪个路径 MISSING |
| 页面打不开但接口正常 | 是否用旧包（`pip show -f sichuan-speech` 看 `web/index.html` 在不在） |
| 识别结果全是标点 | 音频太短或全是静音；先 `ffprobe` 看时长 |

日志默认打到 stdout；`--log-level DEBUG` 会打印每个引擎的加载与每次请求的耗时。
