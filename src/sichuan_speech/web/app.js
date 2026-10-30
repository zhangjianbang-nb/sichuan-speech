/* Sichuan speech workshop — vanilla JS, no build step, no external assets. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var state = {
    voices: [],
    voiceId: "sichuan",
    ttsBlob: null,
    ttsText: "",
    ttsUrl: null,
    asrFile: null,
    audioCtx: null,
    recorder: null,
    chunks: [],
    micStartedAt: 0,
    micTimer: 0,
  };

  var MAX_TEXT = 500;
  var MAX_UPLOAD_MB = 20;

  /* ------------------------------------------------------------- utilities */

  function toast(message, isError) {
    var el = $("toast");
    el.textContent = message;
    el.classList.toggle("is-error", !!isError);
    el.hidden = false;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(function () { el.hidden = true; }, isError ? 6000 : 3200);
  }

  function setStatus(text, kind) {
    var box = $("status");
    $("status-text").textContent = text;
    box.classList.toggle("is-ok", kind === "ok");
    box.classList.toggle("is-bad", kind === "bad");
  }

  function setInline(id, text, kind) {
    var el = $(id);
    el.textContent = text || "";
    el.classList.toggle("is-error", kind === "error");
    el.classList.toggle("is-ok", kind === "ok");
  }

  function busy(buttonId, spinnerId, isBusy, label) {
    var button = $(buttonId);
    $(spinnerId).hidden = !isBusy;
    button.disabled = isBusy;
    button.classList.toggle("is-busy", isBusy);
    if (label) button.querySelector(".btn-label").textContent = label;
  }

  async function readError(response) {
    var detail = "";
    try {
      var payload = await response.json();
      if (payload && payload.detail) detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
    } catch (err) {
      try { detail = await response.text(); } catch (err2) { detail = ""; }
    }
    return (detail || response.statusText || "请求失败").slice(0, 300);
  }

  function humanDuration(seconds) {
    if (!isFinite(seconds)) return "-";
    return seconds >= 60 ? Math.floor(seconds / 60) + " 分 " + Math.round(seconds % 60) + " 秒" : seconds.toFixed(1) + " 秒";
  }

  /* ---------------------------------------------------------- text / CER */

  function normalizeText(text) {
    return String(text || "")
      .replace(/<\|[^|]*\|>/g, "")
      .replace(/[\s，。、？！；：,.?!;:\"'“”‘’（）()《》<>·—\-\u200b\u200c\u200d\ufeff]/g, "");
  }

  function editDistance(reference, hypothesis) {
    var r = Array.from(reference);
    var h = Array.from(hypothesis);
    var previous = new Array(h.length + 1);
    var current = new Array(h.length + 1);
    for (var j = 0; j <= h.length; j += 1) previous[j] = j;
    for (var i = 1; i <= r.length; i += 1) {
      current[0] = i;
      for (var k = 1; k <= h.length; k += 1) {
        var cost = r[i - 1] === h[k - 1] ? 0 : 1;
        current[k] = Math.min(previous[k] + 1, current[k - 1] + 1, previous[k - 1] + cost);
      }
      var swap = previous;
      previous = current;
      current = swap;
    }
    return previous[h.length];
  }

  function cer(reference, hypothesis) {
    var ref = Array.from(normalizeText(reference));
    if (!ref.length) return normalizeText(hypothesis).length ? 1 : 0;
    return editDistance(ref, normalizeText(hypothesis)) / ref.length;
  }

  /* -------------------------------------------------------------- waveform */

  function audioContext() {
    if (!state.audioCtx) {
      var Ctx = window.AudioContext || window.webkitAudioContext;
      state.audioCtx = new Ctx();
    }
    return state.audioCtx;
  }

  function sizeCanvas(canvas) {
    var ratio = window.devicePixelRatio || 1;
    var width = canvas.clientWidth || canvas.width;
    var height = canvas.clientHeight || canvas.height;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    return { ratio: ratio, width: width, height: height };
  }

  function drawWave(buffer) {
    var canvas = $("wave");
    var dims = sizeCanvas(canvas);
    var ctx = canvas.getContext("2d");
    var data = buffer.getChannelData(0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    var bars = Math.max(80, Math.floor(dims.width / 3));
    var step = Math.max(1, Math.floor(data.length / bars));
    var mid = canvas.height / 2;
    var maxHeight = canvas.height * 0.42;

    ctx.strokeStyle = "rgba(222, 211, 191, 0.9)";
    ctx.lineWidth = dims.ratio;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(canvas.width, mid);
    ctx.stroke();

    var gradient = ctx.createLinearGradient(0, mid - maxHeight, 0, mid + maxHeight);
    gradient.addColorStop(0, "rgba(193, 63, 44, 0.85)");
    gradient.addColorStop(1, "rgba(156, 47, 32, 0.55)");
    ctx.fillStyle = gradient;

    var barWidth = Math.max(1, (canvas.width / bars) * 0.62);
    for (var i = 0; i < bars; i += 1) {
      var peak = 0;
      var start = i * step;
      for (var s = start; s < Math.min(start + step, data.length); s += 1) {
        var value = Math.abs(data[s]);
        if (value > peak) peak = value;
      }
      var height = Math.max(1, Math.min(maxHeight, peak * maxHeight * 1.6));
      var x = (i / bars) * canvas.width;
      ctx.fillRect(x, mid - height, barWidth, height * 2);
    }
  }

  async function renderWave(blob) {
    try {
      var arrayBuffer = await blob.arrayBuffer();
      var buffer = await audioContext().decodeAudioData(arrayBuffer);
      drawWave(buffer);
      return buffer.duration;
    } catch (err) {
      toast("波形绘制失败：" + (err.message || err), true);
      return null;
    }
  }

  /* ----------------------------------------------------------------- voices */

  function renderVoices(payload) {
    var container = $("voices");
    container.innerHTML = "";
    state.voices = (payload && payload.voices) || [];
    var available = state.voices.filter(function (voice) { return voice.available; });
    if (!state.voices.length) {
      container.innerHTML = '<p class="placeholder">服务没有返回声音列表。</p>';
      return;
    }
    var preferred = available.some(function (voice) { return voice.id === "sichuan"; })
      ? "sichuan"
      : (available[0] ? available[0].id : state.voices[0].id);
    state.voiceId = preferred;

    state.voices.forEach(function (voice) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "voice";
      button.setAttribute("role", "radio");
      button.setAttribute("aria-checked", String(voice.id === state.voiceId));
      button.dataset.voice = voice.id;
      if (!voice.available) {
        button.disabled = true;
        button.title = voice.reason || "不可用";
      }
      var top = document.createElement("div");
      top.className = "voice-top";
      var name = document.createElement("span");
      name.className = "voice-name";
      name.textContent = voice.label || voice.id;
      var tag = document.createElement("span");
      tag.className = "voice-tag";
      tag.textContent = voice.engine === "sft" ? "sft" : "base";
      top.appendChild(name);
      top.appendChild(tag);
      var desc = document.createElement("span");
      desc.className = "voice-desc";
      desc.textContent = voice.description || "";
      button.appendChild(top);
      button.appendChild(desc);
      if (!voice.available) {
        var why = document.createElement("span");
        why.className = "voice-why";
        why.textContent = "不可用：" + (voice.reason || "");
        button.appendChild(why);
      }
      button.addEventListener("click", function () {
        state.voiceId = voice.id;
        Array.prototype.forEach.call(container.children, function (child) {
          if (child.classList && child.classList.contains("voice")) {
            child.setAttribute("aria-checked", String(child.dataset.voice === voice.id));
          }
        });
      });
      container.appendChild(button);
    });
  }

  async function loadServiceInfo() {
    try {
      var health = await fetch("/health", { headers: { Accept: "application/json" } });
      if (!health.ok) throw new Error("HTTP " + health.status);
      var info = await health.json();
      setStatus("服务在线 · v" + info.version + " · GPU " + info.gpu, "ok");
      $("version").textContent = "v" + info.version;
      var voices = await fetch("/voices", { headers: { Accept: "application/json" } });
      if (!voices.ok) throw new Error("HTTP " + voices.status);
      renderVoices(await voices.json());
    } catch (err) {
      setStatus("服务未连接", "bad");
      $("voices").innerHTML = '<p class="placeholder">读不到声音列表：' + (err.message || err) + "</p>";
    }
  }

  /* -------------------------------------------------------------------- TTS */

  function ttsPayload() {
    var text = $("tts-text").value.trim();
    if (!text) {
      toast("请先输入要合成的文本", true);
      $("tts-text").focus();
      return null;
    }
    if (Array.from(text).length > MAX_TEXT) {
      toast("文本超过 " + MAX_TEXT + " 字", true);
      return null;
    }
    var checked = document.querySelector('#tts-nsamples input:checked');
    var body = new FormData();
    body.append("text", text);
    body.append("voice", state.voiceId);
    body.append("speed", String(parseFloat($("tts-speed").value)));
    body.append("n_samples", checked ? checked.value : "3");
    return { text: text, body: body };
  }

  function badge(label, value, kind) {
    var span = document.createElement("span");
    span.className = "badge" + (kind ? " " + kind : "");
    span.innerHTML = label + " <strong>" + value + "</strong>";
    return span;
  }

  async function runTts() {
    var payload = ttsPayload();
    if (!payload) return;
    var started = performance.now();
    busy("tts-run", "tts-spinner", true, "合成中…");
    setInline("tts-status", "正在合成，首次调用需要加载模型，可能要几十秒。");
    try {
      var response = await fetch("/tts", { method: "POST", body: payload.body });
      if (!response.ok) throw new Error(await readError(response));
      var blob = await response.blob();
      var elapsed = (performance.now() - started) / 1000;

      if (state.ttsUrl) URL.revokeObjectURL(state.ttsUrl);
      state.ttsBlob = blob;
      state.ttsText = payload.text;
      state.ttsUrl = URL.createObjectURL(blob);
      $("player").src = state.ttsUrl;
      $("tts-result").hidden = false;
      $("asr-use-tts").disabled = false;
      $("selfcheck-out").hidden = true;
      $("tts-download").disabled = false;

      var meta = $("tts-meta");
      meta.innerHTML = "";
      var duration = parseFloat(response.headers.get("X-Duration") || "0");
      var rtf = response.headers.get("X-RTF");
      var samples = response.headers.get("X-Samples");
      var bestCer = response.headers.get("X-Best-Cer");
      var sampleRate = response.headers.get("X-Sample-Rate");
      var voiceId = response.headers.get("X-Voice");
      var voice = state.voices.filter(function (item) { return item.id === voiceId; })[0];

      meta.appendChild(badge("时长", duration.toFixed(2) + " s"));
      meta.appendChild(badge("采样率", (sampleRate || "-") + " Hz"));
      meta.appendChild(badge("候选", (samples || "1") + " 条"));
      meta.appendChild(badge("声音", voice ? voice.label : (voiceId || "-")));
      if (rtf) meta.appendChild(badge("实时率", parseFloat(rtf).toFixed(2)));
      if (bestCer !== null) {
        var value = parseFloat(bestCer);
        meta.appendChild(badge("回读字错率", (value * 100).toFixed(1) + "%", value === 0 ? "is-good" : (value > 0.2 ? "is-warn" : "")));
      } else {
        meta.appendChild(badge("回读字错率", "未回读"));
      }
      meta.appendChild(badge("请求耗时", elapsed.toFixed(1) + " s"));

      renderWave(blob);
      setInline("tts-status", "完成：" + duration.toFixed(2) + " 秒音频，用时 " + elapsed.toFixed(1) + " 秒。", "ok");
    } catch (err) {
      setInline("tts-status", "", "");
      toast("合成失败：" + (err.message || err), true);
    } finally {
      busy("tts-run", "tts-spinner", false, "合成语音");
    }
  }

  async function selfCheck() {
    if (!state.ttsBlob) {
      toast("先合成一段音频", true);
      return;
    }
    var started = performance.now();
    setInline("selfcheck-status", "回读中…");
    var body = new FormData();
    body.append("file", state.ttsBlob, "tts.wav");
    body.append("model", "ft");
    try {
      var response = await fetch("/asr", { method: "POST", body: body });
      if (!response.ok) throw new Error(await readError(response));
      var payload = await response.json();
      var elapsed = (performance.now() - started) / 1000;
      var value = cer(state.ttsText, payload.text);
      var out = $("selfcheck-out");
      out.hidden = false;
      out.innerHTML = "";
      var header = document.createElement("header");
      var title = document.createElement("h3");
      title.textContent = "方言识别器回读";
      var meta = document.createElement("span");
      meta.className = "transcript-meta";
      meta.textContent = "字错率 " + (value * 100).toFixed(1) + "% · 用时 " + elapsed.toFixed(1) + " 秒";
      header.appendChild(title);
      header.appendChild(meta);
      var text = document.createElement("p");
      text.className = "transcript-text";
      text.textContent = payload.text || "（空）";
      var note = document.createElement("p");
      note.className = "hint";
      note.textContent = "参考文本：" + state.ttsText;
      out.appendChild(header);
      out.appendChild(text);
      out.appendChild(note);
      setInline("selfcheck-status", value === 0 ? "回读完全一致" : "有 " + (value * 100).toFixed(1) + "% 的字不同", value === 0 ? "ok" : "error");
    } catch (err) {
      setInline("selfcheck-status", "", "");
      toast("回读失败：" + (err.message || err), true);
    }
  }

  function downloadTts() {
    if (!state.ttsUrl) return;
    var anchor = document.createElement("a");
    anchor.href = state.ttsUrl;
    anchor.download = "sichuan-tts-" + Date.now() + ".wav";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  }

  /* -------------------------------------------------------------------- ASR */

  function setAsrFile(file, label) {
    if (!file) return;
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      toast("文件超过 " + MAX_UPLOAD_MB + " MB", true);
      return;
    }
    state.asrFile = file;
    var drop = $("drop");
    drop.classList.add("has-file");
    $("drop-title").textContent = label || file.name;
    $("drop-sub").textContent = (file.size / 1024).toFixed(0) + " KB · 已就绪";
    $("asr-run").disabled = false;
  }

  function clearAsrResult() {
    $("asr-result").hidden = true;
    $("asr-ft-block").hidden = true;
    $("asr-base-block").hidden = true;
  }

  function renderTranscript(prefix, title, payload, elapsedMs) {
    var block = $(prefix + "-block");
    block.hidden = false;
    block.classList.toggle("is-base", prefix === "asr-base");
    block.querySelector("h3").textContent = title;
    $(prefix + "-meta").textContent =
      "用时 " + (payload.elapsed_s !== undefined ? payload.elapsed_s : (elapsedMs / 1000).toFixed(1)) + " 秒 · 模型 " + payload.model;
    $(prefix + "-text").textContent = payload.text || "（空）";
  }

  async function asrRequest(file, model) {
    var body = new FormData();
    body.append("file", file, file.name || "audio.wav");
    body.append("model", model);
    var response = await fetch("/asr", { method: "POST", body: body });
    if (!response.ok) throw new Error(await readError(response));
    return response.json();
  }

  async function runAsr() {
    if (!state.asrFile) {
      toast("先选择一个音频文件", true);
      return;
    }
    var compare = $("compare").checked;
    var selected = document.querySelector('#asr-model input:checked');
    var model = selected ? selected.value : "ft";
    busy("asr-run", "asr-spinner", true, "识别中…");
    clearAsrResult();
    setInline("asr-status", compare ? "两个模型依次识别，串行执行。" : "识别中…");
    var started = performance.now();
    try {
      if (compare) {
        var ft = await asrRequest(state.asrFile, "ft");
        var base = await asrRequest(state.asrFile, "base");
        renderTranscript("asr-ft", "方言微调版", ft, 0);
        renderTranscript("asr-base", "原版普通话", base, 0);
        setInline("asr-status", "完成，两种模型共 " + ((performance.now() - started) / 1000).toFixed(1) + " 秒。", "ok");
      } else {
        var result = await asrRequest(state.asrFile, model);
        renderTranscript("asr-ft", model === "ft" ? "方言微调版" : "原版普通话", result, 0);
        $("asr-ft-block").classList.toggle("is-base", model === "base");
        setInline("asr-status", "完成，用时 " + ((performance.now() - started) / 1000).toFixed(1) + " 秒。", "ok");
      }
      $("asr-result").hidden = false;
    } catch (err) {
      setInline("asr-status", "", "");
      toast("识别失败：" + (err.message || err), true);
    } finally {
      busy("asr-run", "asr-spinner", false, "开始识别");
      $("asr-run").disabled = !state.asrFile;
    }
  }

  /* --------------------------------------------------------------- recording */

  function encodeWav(buffer) {
    var channels = Math.min(1, buffer.numberOfChannels);
    var data = buffer.getChannelData(0);
    var bytesPerSample = 2;
    var blockAlign = channels * bytesPerSample;
    var dataSize = data.length * blockAlign;
    var arrayBuffer = new ArrayBuffer(44 + dataSize);
    var view = new DataView(arrayBuffer);
    var writeString = function (offset, text) {
      for (var i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
    };
    writeString(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channels, true);
    view.setUint32(24, buffer.sampleRate, true);
    view.setUint32(28, buffer.sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, dataSize, true);
    var offset = 44;
    for (var s = 0; s < data.length; s += 1) {
      var sample = Math.max(-1, Math.min(1, data[s]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
    return new Blob([view], { type: "audio/wav" });
  }

  async function toggleRecording() {
    var button = $("mic");
    if (state.recorder && state.recorder.state === "recording") {
      state.recorder.stop();
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast("当前浏览器不支持录音，请改用上传文件", true);
      return;
    }
    try {
      var stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      var mime = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg"].filter(function (type) {
        return window.MediaRecorder && MediaRecorder.isTypeSupported(type);
      })[0];
      state.recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      state.chunks = [];
      state.recorder.addEventListener("dataavailable", function (event) {
        if (event.data && event.data.size) state.chunks.push(event.data);
      });
      state.recorder.addEventListener("stop", async function () {
        clearInterval(state.micTimer);
        stream.getTracks().forEach(function (track) { track.stop(); });
        button.classList.remove("is-recording");
        $("mic-label").textContent = "按住说话";
        var seconds = (Date.now() - state.micStartedAt) / 1000;
        setInline("mic-status", "录了 " + humanDuration(seconds) + "，正在转码…");
        var recorded = new Blob(state.chunks, { type: (state.recorder.mimeType || "audio/webm") });
        try {
          var arrayBuffer = await recorded.arrayBuffer();
          var buffer = await audioContext().decodeAudioData(arrayBuffer);
          var wav = encodeWav(buffer);
          setAsrFile(new File([wav], "recording.wav", { type: "audio/wav" }), "刚录的一段（" + humanDuration(buffer.duration) + "）");
          setInline("mic-status", "已转成 WAV，可以识别了。", "ok");
        } catch (err) {
          setInline("mic-status", "", "");
          toast("录音转码失败：" + (err.message || err), true);
        }
      });
      state.recorder.start();
      state.micStartedAt = Date.now();
      button.classList.add("is-recording");
      $("mic-label").textContent = "停止录音";
      setInline("mic-status", "录音中…");
      state.micTimer = setInterval(function () {
        setInline("mic-status", "录音中 " + humanDuration((Date.now() - state.micStartedAt) / 1000));
      }, 500);
    } catch (err) {
      toast("无法打开麦克风：" + (err.message || err), true);
    }
  }

  /* ------------------------------------------------------------------- wiring */

  function bind() {
    $("tts-text").addEventListener("input", function () {
      var length = Array.from($("tts-text").value).length;
      $("tts-count").textContent = length + " / " + MAX_TEXT;
      $("tts-count").style.color = length > MAX_TEXT * 0.95 ? "var(--vermilion-dark)" : "";
    });

    $("tts-samples").addEventListener("click", function (event) {
      var chip = event.target.closest(".chip");
      if (!chip) return;
      $("tts-text").value = chip.dataset.text;
      $("tts-text").dispatchEvent(new Event("input"));
      $("tts-text").focus();
    });

    $("tts-speed").addEventListener("input", function () {
      $("speed-value").textContent = parseFloat($("tts-speed").value).toFixed(2) + "×";
    });

    $("tts-run").addEventListener("click", runTts);
    $("tts-selfcheck").addEventListener("click", selfCheck);
    $("tts-download").addEventListener("click", downloadTts);
    $("tts-clear").addEventListener("click", function () {
      $("tts-text").value = "";
      $("tts-text").dispatchEvent(new Event("input"));
      $("tts-result").hidden = true;
      $("tts-status").textContent = "";
      $("selfcheck-out").hidden = true;
      state.ttsBlob = null;
      state.ttsText = "";
      $("asr-use-tts").disabled = true;
      if (state.ttsUrl) { URL.revokeObjectURL(state.ttsUrl); state.ttsUrl = null; }
      $("player").removeAttribute("src");
    });

    $("file").addEventListener("change", function () {
      if (this.files && this.files[0]) setAsrFile(this.files[0]);
      this.value = "";
    });

    var drop = $("drop");
    drop.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        $("file").click();
      }
    });
    ["dragenter", "dragover"].forEach(function (name) {
      drop.addEventListener(name, function (event) {
        event.preventDefault();
        drop.classList.add("is-over");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      drop.addEventListener(name, function (event) {
        event.preventDefault();
        drop.classList.remove("is-over");
      });
    });
    drop.addEventListener("drop", function (event) {
      var file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) setAsrFile(file);
    });

    $("mic").addEventListener("click", toggleRecording);
    $("asr-run").addEventListener("click", runAsr);

    $("asr-use-tts").addEventListener("click", function () {
      if (!state.ttsBlob) return;
      setAsrFile(new File([state.ttsBlob], "synthesized.wav", { type: "audio/wav" }), "上一步合成的音频");
      toast("已把合成音频放进识别区");
    });

    document.addEventListener("click", function (event) {
      var button = event.target.closest("[data-copy]");
      if (!button) return;
      var text = $(button.dataset.copy).textContent;
      navigator.clipboard.writeText(text).then(function () {
        toast("已复制");
      }, function () {
        toast("复制失败，请手动选择文本", true);
      });
    });

    $("tts-text").dispatchEvent(new Event("input"));
    window.addEventListener("resize", function () {
      if (state.ttsBlob) renderWave(state.ttsBlob);
    });
  }

  bind();
  loadServiceInfo();
})();
