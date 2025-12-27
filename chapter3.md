# Chapter 3: 数据与标注：采集、清洗、切分、对齐与许可

## 1. 开篇段落

在 ASR 与 Diarization 的开发中，数据往往占据了 80% 的工作量。从 RNN 时代的声学模型，到 Transformer 时代的端到端模型，再到如今的 MLLM（多模态大模型），数据处理的范式发生了巨大的迁移：从**“特征工程”**变成了**“数据工程”**。

早期模型（GMM-HMM）需要极其精确的音素级对齐；端到端模型（E2E）开始容忍较粗糙的对齐；而 MLLM 则引入了“指令微调（Instruction Tuning）”和“上下文学习（In-Context Learning）”，要求数据不仅要有音频和文本，还要有**意图（Intent）和知识（Knowledge）**。

本章将作为“数据工程”的实战册，指导你如何从杂乱的原始录音中提炼出高质量的训练燃料。我们将深入探讨合规性红线、物理音频规范、防泄漏的切分策略、强制对齐（Forced Alignment）流水线，以及如何为下一代语音模型构建 JSONL 格式的指令数据。

**本章学习目标**：

1. **合规风控**：建立数据许可（License）审查机制，识别“有毒”数据。
2. **音频工程**：掌握 `ffmpeg`/`sox` 的标准化处理，理解采样率与编码的深层影响。
3. **切分架构**：设计严密的 Train/Dev/Test 切分方案，杜绝 Session/Speaker 泄漏。
4. **自动流水线**：利用 MFA/CTC-Segmentation 实现大规模数据的自动清洗与对齐。
5. **MLLM 适配**：学会构造包含 Instruction、Prompt 和 Context 的新型训练数据。

---

## 2. 文字论述

### 3.1 数据许可与“可用性分级”

数据污染是不可逆的。一旦商业模型在训练中见过了 NC（非商用）数据，整个模型的法律地位岌岌可危。

#### 3.1.1 许可红绿灯

* 🟢 **Green (安全商用)**:
* **CC-0 (Public Domain)**: 如 LibriSpeech（部分）。
* **CC-BY (Attribution)**: 需保留署名文件。如 Common Voice (部分版本，需仔细检查)。
* **Apache 2.0 / MIT**: 常见于代码库附带的小型数据集。


* 🟡 **Yellow (需购买/授权)**:
* **LDC (Linguistic Data Consortium)**: 质量极高，价格昂贵，通常购买后拥有商用权。
* **Data Tang / MagicData (付费版)**: 商业数据供应商。


* 🔴 **Red (绝对禁止商用)**:
* **CC-BY-NC (Non-Commercial)**: 学术界最常用的许可（如 AISHELL-3, BigFish 等）。**企业研发人员请务必在下载脚本中设置黑名单过滤此类数据。**
* **YouTube/Podcast 爬取**: 除非不仅是 Public Domain 且确认内容未侵犯第三方版权，否则默认视为**高风险**。



#### 3.1.2 隐私与 PII (Personally Identifiable Information)

在欧盟 GDPR 和中国《个人信息保护法》下，语音生物特征属于敏感个人信息。

* **音频脱敏**: 对变声处理通常会破坏 ASR 特征，因此重点在于**授权书**。
* **文本脱敏**: 训练前必须替换姓名、身份证号、电话号码、银行卡号。
* *替换策略*: `13800000000`  `<PHONE_NUMBER>` 或随机生成的假号码（保持韵律特征）。



### 3.2 音频基础规范：工程视角的参数选择

统一的输入格式能显著降低 DataLoader 的 CPU 开销。

#### 3.2.1 采样率 (Sample Rate)

* **16kHz (Wideband)**: ASR 黄金标准。根据奈奎斯特采样定理，能覆盖 8kHz 以下频率，包含了人类语音绝大部分的可懂度信息（辅音的高频部分）。
* **8kHz (Narrowband)**: 电话系统标准。如果业务场景是电话客服，**必须**包含 8kHz 数据，或者将 16kHz 下采样训练。
* **Up-sampling 的陷阱**: 绝对不要把 8kHz 插值强转为 16kHz 混入训练，这会产生“空频带”，导致模型在真实 16kHz 场景下对高频噪声过敏。

#### 3.2.2 编码与容器

* **PCM WAV (signed 16-bit)**: 训练首选。解码速度最快，无压缩损耗。
* **FLAC**: 存储首选。无损压缩（节省 ~40% 空间），解码开销适中。
* **MP3/AAC**: **特征杀手**。心理声学模型会切除“人耳听不见但模型可能需要”的频谱细节。
* *Rule-of-Thumb*: 如果原始数据是 MP3，训练时**不要**转为 WAV 存储（浪费空间且无法找回音质），直接解码读取，但需在 Data Augmentation 中加入 Codec Augmentation（模拟压缩伪影）以增强鲁棒性。



#### 3.2.3 多通道与阵列

* **Diarization/Meeting 场景**: 尽量保留多通道原始音频。
* **波束形成 (Beamforming)**: 建议在训练时**在线**随机选取一个通道，或者做前端增强合成单通道。**不要只用增强后的音频训练**，否则模型会丧失对混响和噪声的鲁棒性。

### 3.3 标注粒度与格式

#### 3.3.1 ASR 标注

* **Verbatim (逐字)**: 包含 "um", "uh", 重复, 结巴。适合训练转模型。
* **Cleaned (整理后)**: 去除口癖。适合字幕生成。**注意：如果音频有 "um" 但文本没有，会导致 CTC/Transducer 训练时的对齐混乱。建议训练由 Verbatim 文本驱动，下游任务再做 ITN。**

#### 3.3.2 Diarization 标注 (RTTM 标准)

RTTM (Rich Transcription Time Marked) 是 NIST 评测的标准格式。

```text
Type  File  Ch  Start   Dur     Ortho  SType  Name     Conf
SPEAKER file1 1   0.50    3.25    <NA>   <NA>   spk_01   <NA>
SPEAKER file1 1   4.00    2.10    <NA>   <NA>   spk_02   <NA>

```

* **Overlap (重叠)**: 高级 Diarization 系统的关键。如 0.5s-3.75s 是 spk_01，3.5s-5.0s 是 spk_02，则 3.5s-3.75s 为重叠区。标注必须精确覆盖重叠，否则 EEND (End-to-End Neural Diarization) 模型无法收敛。

### 3.4 切分策略：防泄漏的艺术

除了 Chapter 1 提到的 Speaker/Session 泄漏，还有更隐蔽的泄漏方式。

**[ASCII Diagram: 数据集切分层级]**

```text
Level 1: Source Split (最安全)
[  Conference A (Train)  ]   [  Conference B (Dev)  ]   [  Conference C (Test)  ]
        |                           |                           |
        V                           V                           V
   包含 spk_1 ~ spk_50        包含 spk_51 ~ spk_60       包含 spk_61 ~ spk_70

Level 2: Speaker Split (次级安全 - 适用于单人朗读)
[  Spk 1 (Train) ] ... [ Spk 80 (Train) ]  ||  [ Spk 81 (Dev) ] ... [ Spk 100 (Test) ]

Level 3: Utterance Split (Random Shuffle) -> ☠️ 灾难级错误
[ Spk1_utt1 (Train) ] ... [ Spk1_utt2 (Test) ] ...
Result: 测试集 WER 1%，上线 WER 30%。模型记住了 Spk1 的麦克风底噪。

```

* **Stratified Sampling (分层采样)**: 确保 Test Set 覆盖：
* **性别比例**: 1:1
* **时长分布**: 短句 (1s) 到长句 (15s) 都要有。
* **信噪比 (SNR)**: 安静环境与嘈杂环境。
* **口音**: 如果做中文 ASR，测试集必须包含非标准普通话（川普、广普）。



### 3.5 强制对齐 (Forced Alignment) 与自动化清洗

面对 10,000 小时的原始录音（如会议录音、电视剧），人工切分是不可能的。必须构建自动流水线。

**推荐工具链**:

1. **Montreal Forced Aligner (MFA)**: 基于 Kaldi GMM-HMM。安装略繁琐，但边界极其精准。
2. **CTC-Segmentation**: 基于 PyTorch 和预训练 CTC 模型。对噪声鲁棒性更好，更适合现代流程。

**[ASCII Diagram: 自动切分清洗流水线]**

```text
Raw Audio (1 Hour) ------------------------+
      |                                    |
      V                                    V
+-------------+                    +--------------+
|     VAD     | (粗切分)            |  ASR Model   | (辅助)
+-------------+                    +--------------+
      |                                    |
      V                                    V
[Active Segments]                  [Loose Alignment Logits]
      |                                    |
      +---------------+--------------------+
                      |
                      V
             +------------------+
             | CTC-Segmentation | (核心算法)
             +------------------+
                      |
        +-------------+-------------+
        |                           |
[Aligned Segments]          [Confidence Score]
(Start: 10.5s, End: 15.2s)  (Score: -0.2)
        |                           |
        V                           V
   +----------+             +----------------+
   |  Slicer  | <---------- | Quality Filter | (If score < threshold: DROP)
   +----------+             +----------------+
        |
        V
Final Dataset (Chunks < 30s)

```

**工程细节 (Rule-of-Thumb)**:

* **切分时长**: 训练 ASR 的最佳 chunk 时长为 **10s - 30s**。太短导致 Transformer 没上下文，太长导致 OOM (Out of Memory)。
* **边界扩充**: 在 VAD 切分点前后各加 **0.1s - 0.2s** 的 padding，防止切掉首尾辅音。
* **丢弃标准**:
* 时长 < 0.5s（无意义）。
* 时长 > 30s（可能含大段静音或 VAD 失败）
* 字符/时长比 (CPS) > 10 (语速过快，通常是对齐错误) 或 < 1 (全是噪音)。



### 3.6 面向 MLLM 的数据构造

MLLM (如 Qwen-Audio, GPT-4o) 的输入不再是单纯的 `(wav, text)` 对，而是结构化的 Instruction Data。

#### 3.6.1 数据结构示例 (JSONL)

```json
{
  "id": "train_001",
  "audio": "path/to/audio.wav",
  "duration": 5.2,
  "conversations": [
    {
      "role": "user",
      "content": "<|audio_bos|><|AUDIO|><|audio_eos|>请将这段语音转写为文本，并自动纠正其中的口语错误（如重复、结巴）。"
    },
    {
      "role": "assistant",
      "content": "今天天气真不错，我们一起去公园吧。"
    }
  ],
  "context": {
    "hotwords": ["公园", "天气"],
    "speaker_profile": "Young female, Beijing accent"
  }
}

```

#### 3.6.2 Context Injection (上下文注入)

为了训练模型支持 RAG（检索增强生成）或热词（Biasing），需要在训练数据中**动态合成**上下文。

* **正例构造**: 从 Ground Truth 中随机抽取实体词放入 Context 字段，要求模型在输出时予以关注。
* **负例构造 (Hard Negatives)**: 在 Context 中故意放入音频中*没有*出现的相似词（如音频是“张三”，Context 给“张山”），训练模型**不被误导**的能力。

---

## 3. 本章小结

1. **数据决定上限**: 任何模型架构的改进都无法弥补 Session 泄漏导致的评测虚高。
2. **对齐即正义**: 掌握 CTC-Segmentation 或 MFA 是处理大规模非监督/弱监督数据的核心能力。
3. **物理一致性**: 严格统一采样率（16kHz）和位深，慎用 MP3。
4. **MLLM 范式**: 数据准备需从单纯的“转写”转向“指令-响应”对，包含上下文注入与多任务 Prompt。

---

## 4. 练习题

### 基础题

**Q1: 在处理多通道会议录音（例如 8 个麦克风阵列）进行单通道 ASR 训练时，简单地将 8 个通道平均（Average）成一个通道会有什么物理问题？**

<details>
<summary>点查看答案与提示</summary>

* **提示**: 考虑声波的相位（Phase）和延迟（Delay）。
* **答案**:
* 由于声源到达不同麦克风的距离不同，存在时间延迟（Time Delay）。
* 直接平均会导致**梳状滤波效应（Comb Filtering）**。某些频率因为相位相反相消（Destructive Interference），导致信号失真、高频丢失，声音听起来像是在水管里。
* **正确做法**: 随机选一个通道（Random Selection）或使用专业的波束形成（Beamforming）算法（如 MVDR）合成。



</details>

**Q2: 为什么在计算 ASR 数据集的时长时，要区分“音频总时长”和“有效语音时长”？差异通常有多大？**

<details>
<summary>点击查看答案与提示</summary>

* **提示**: VAD (Voice Activity Detection)。
* **答案**:
* 原始录音（特别是会议或对话）包含大量静音、思考停顿和背景噪声。
* **差异**: 在自然对话中，有效语音（Speech）通常只占总时长的 40%-60%。
* **影响**: 如果按总时长计算 Epoch，会导致模型在大量的 Silence 上浪费算力，甚至过拟合静音模式。训练时应仅计算有效 Speech 帧。



</details>

**Q3: 使用 `sox` 将 48kHz 音频下采样到 16kHz 时，如果不加低通滤波器（Low-pass filter）会发生什么？**

<details>
<summary>点击查看答案与提示</summary>

* **提示**: 混叠（Aliasing）。
* **答案**:
* 会发生**混叠效应**。高于 8kHz（新奈奎斯特频率）的频率成分会“折叠”回低频段（0-8kHz）。
* 例如，原始音频中的 10kHz 信号，在 16kHz 采样率下会表现为  的伪影噪声。
* **注**: 现代工具如 `sox` 或 `ffmpeg` 默认会在下采样前自动应用低通滤波器，但手写 DSP 代码时需极度小心。



</details>

### 挑战题

**Q4 (场景设计): 你正在为一家医院开发“医生查房录音”ASR 系统。数据极其敏感（不能出内网），且包含大量医学术语。你只有 10 小时的医生真录音，但有 10,000 小时的通用开源数据（AISHELL 等）。请设计数据混合与训练策略。**

<details>
<summary>点击查看答案与提示</summary>

* **提示**: Domain Adaptation, Mixing Ratio, Lexicon。
* **答案**:
1. **数据混合 (Mixing)**: 通用数据用于学习声学特征（发音），医疗数据用于适应领域。训练时每个 Batch 保持一定比例（如 1:1 或 1:2），即使医疗数据很少，也要通过 **Over-sampling (重采样)** 保证模型每一轮都能反复看到。
2. **词表增强 (Lexicon)**: 必须构建医疗术语表。在 Chapter 4 中会讲到，利用术语表生成合成文本（TTS）或做 Text-only 的 LM 训练来增强对生僻药名的识别。
3. **切分**: 10 小时真实数据极其宝贵，建议拿出 1-2 小时做 Dev/Test，剩下 8 小时全部混入 Train。**绝对不要**只用通用数据训练，然后指望在医疗 Dev 上调参。



</details>

**Q5 (MLLM): 你想训练一个能根据语音语调判断情感高兴/愤怒/中性）的 MLLM。你现有的数据只有 ASR 转写文本。如何利用开源工具低成本地“伪造”情感标签来启动训练？**

<details>
<summary>点击查看答案与提示</summary>

* **提示**: SER (Speech Emotion Recognition) 预训练模型。
* **答案**:
1. **Teacher Model**: 下载一个开源的语音情感识别模型（如基于 Wav2Vec2-Emotion 或 distil-wav2vec2-audio-better-com）。
2. **Pseudo-labeling**: 用该模型跑一遍你的 ASR 数据集，获取每个片段的情感 Logits 或分类结果。
3. **清洗**: 设定高阈值（如 Conf > 0.9），只保留置信度极高的样本作为“伪真值”。
4. **构造指令**: Instruction: "Recognize the speech and identify the speaker's emotion." -> Output: "(Angry) Get out of here!"



</details>

---

## 5. 常见陷阱与错误 (Gotchas)

### 5.1 OOV (Out-of-Vocabulary) 爆炸

* **现象**: 训练 CER 很低，但测试时只要遇到人名地名就全错。
* **原因**: 英文 BPE 或中文 Char 词表没覆盖生僻字。
* **Gotcha**: 很多开源中文词表只有 3000-5000 常用字。对于人名（如“李**鱏**”），如果词表中没有，模型只能输出 `<UNK>` 或读音相近的错字。
* **对策**: 检查训练集覆盖率。如果做通用 ASR，中文字表应在 6000-8000 字左右；如果做 MLLM，通常沿用 LLM 的大词表（50k-100k token）。

### 5.2 错误的 sox/ffmpeg 管道

* **错误代码**: `ffmpeg -i input.mp3 -ar 16000 output.wav` (看似没问题)
* **陷阱**: 如果原音频是立体声（Stereo），ffmpeg 默认会合并通道或只取左声道（取决于版本和参数），可能导致相位抵消（见练习题 Q1）。
* **修正**: 显式指定通道处理策略。例如 `ffmpeg -i input.mp3 -ac 1 -ar 16000 output.wav` (混合为单声道，需确认无相位问题) 或 `-map_channel` 提取特定通道。

### 5.3 忽视 Text Normalization 对齐的影响

* **现象**: 音频里说 "The price is $5"，文本标注是 "The price is five dollars"。
* **Gotcha**: 强制对齐工具会因为 "" 符号上。
* **对策**: 在强制对齐**前**，必须将文本做 **Verbalization**（口语化展开），即 `$5` -> `five dollars`。这是 Chapter 4 的核心内容，但在 Chapter 3 的对齐阶段就得预处理。

### 5.4 验证集 Loss 很低，WER 很高

* **原因**: Teacher Forcing 的陷阱。
* **Gotcha**: 训练时模型总是能看到上一个正确的 Ground Truth Token，所以 Loss 降得很快。但推理（Inference）时模型只能靠自己生成的历史，一步错步步错（Exposure Bias）。
* **对策**: 尽早跑完整的 Decode 评测（Beam Search），不要只看 Loss。Loss 和 WER 在训练后期往往不成线性关系。
