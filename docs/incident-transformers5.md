# 事故复盘：transformers 5.17 让 CosyVoice2 说出乱码

## 现象

基础（未微调）CosyVoice2-0.5B 的零样本合成直接输出乱码：

```
REF : 收到好友从远方寄来的生日礼物，那份意外的惊喜让她开心得像个孩子。
HYP : （空）
dur : 25.60s
```

33 个字的句子生成了 25.6 秒，而且**恰好等于 `max_token_text_ratio=20` 的上限**。
换几句都是这样：长度顶格、ASR 转写全是"咕噜""火萝咕咕噜"这类音节串。

## 排查路径（每一步都排除了一个假设）

| # | 假设 | 检查 | 结论 |
|---|---|---|---|
| 1 | `spk2info` 缺 `embedding` 键 | `KeyError: 'embedding'` at `frontend.py:164` | 真 bug，但不是乱码的原因 |
| 2 | fp16 数值问题 | fp32 重跑 | 更糟：生成 ~1 个 token，hift 直接报 `Kernel size can't be greater than actual input size` |
| 3 | 我给的提示音频有问题 | 换官方 `asset/zero_shot_prompt.wav` + 官方提示文本 | 一样乱码 → 排除提示音频 |
| 4 | 权重加载错位 | 对比 `llm.pt` 与模型 `state_dict` 的键 | 295 vs 295，缺失 0、多余 0 → 排除 |
| 5 | 语音 tokenizer 特征提取不对 | 读 `whisper.log_mel_spectrogram` 源码 | 与 20231117 版本逐行一致 → 排除 |
| 6 | LLM 前向本身有问题 | 教师强制：用提示音的 speech token 做目标，看模型能否复现 | **准确率只有 9.3%** → 模型连提示音都预测不出来 |
| 7 | 解码时的 mask 语义 | 读 `cosyvoice/llm/llm.py` 的 `forward_one_step` | **找到根因** |

## 根因

`Qwen2Encoder.forward_one_step` 这样传 mask：

```python
input_masks = masks[:, -1, :]     # masks 是调用方造的 tril(ones((1, T, T)))
outs = self.model(inputs_embeds=xs, attention_mask=input_masks,
                  use_cache=True, past_key_values=cache)
```

调用方在解码循环里每步把 `lm_input` 换成 `(1, 1, d)`，于是 `masks` 变成 `(1,1,1)`，
`input_masks` 只有 **长度 1**，而 KV cache 里已经有 T 个位置。

- **transformers 4.x**：`_prepare_4d_causal_attention_mask_with_cache_position` 只对
  `causal_mask[..., :mask_length]` 做屏蔽，`mask_length=1` 时第 1 位之后的位置保持不屏蔽，
  实际效果等于全 1 → 恰好能跑。
- **transformers 5.x**：按 mask 字面语义展开，长度 1 的 mask 意味着**只允许看第 0 个 key**，
  其余历史全被屏蔽 → 模型只看得到 SOS，于是要么立刻吐 EOS（fp32），
  要么一路乱生成到上限（fp16）。

这就是为什么"基础模型"也坏——不是我的数据或微调的问题，是环境版本问题。

## 修复

在 `cosyvoice/llm/llm.py` 的 `forward_one_step` 里，按 cache 长度重建全长 mask：

```python
input_masks = masks[:, -1, :]
past = 0
if cache is not None:
    if hasattr(cache, "get_seq_length"):
        past = cache.get_seq_length()
    elif isinstance(cache, (tuple, list)) and len(cache) and cache[0] is not None:
        past = cache[0][0].shape[-2]
total = past + xs.shape[1]
if input_masks.shape[-1] != total:
    input_masks = torch.ones((xs.shape[0], total), dtype=torch.bool, device=xs.device)
```

备份在 `cosyvoice/llm/llm.py.bak-chuan`。修完在 transformers 5.17.0 下：

```
official_man    dur 6.92s  hyp=收到好友从远方寄来的生日礼物，那份意外的惊喜，让他开心的像个孩子。
official_man2   dur 3.96s  hyp=今天天气很好，我们出去走一走吧。
official_sic    dur 3.60s  hyp=今天天气好安逸哟，我们出去耍一哈嘛。
```

同时也验证过：把 `transformers==4.51.3` 装到隔离目录走 `PYTHONPATH`，不装 4.x 也能正常出音，
两种做法都可行；最终选择打补丁，因为不用维护第二套依赖。

## 教训

1. **大版本升级的 `attention_mask` 语义变化不会报错，只会静默降质。** 这类问题用 loss 看不出来
   （训练时用的是完整 2D padding mask，路径正确，所以 SFT 训练一直是好的），只有推理才暴露。
2. **环境健康检查要放在最前面**：拿到一个 TTS 模型，先用仓库自带 asset 跑一句最短的话，
   用 ASR 回读，确认能对上再开始训练。否则会在错误的基线上浪费几个小时。
3. **"基础模型也坏"是最有价值的信号**。它把问题从"我的训练/数据"直接切到"环境/代码"，
   省掉了大量无效调参。
