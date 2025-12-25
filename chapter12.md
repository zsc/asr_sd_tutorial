# Chapter 12: 多语种与混语训练 (Multilingual & Code-Switching)

> **本章定位**：从单语种迈向“巴别塔”的工程指南。
> **前置知识**：建议完成 **Chapter 3**（数据基础）与 **Chapter 4**（文本规范化）。

## 1. 开篇：通天塔的构建艺术

在 ASR 发展的早期，多语种（Multilingual）意味着“拼盘”：为英语训练一个模型，为中文训练另一个，前端加一个语种分类器来路由。这种方法维护成本极高，且无法利用语种间的**正迁移（Positive Transfer）**。

今天，无论是传统的 Conformer-Transducer 还是最新的 MLLM（如 Whisper, Qwen-Audio, Gemini Audio），目标都是**通用语音模型（Universal Speech Model）**：一个模型，同一套参数，处理 100+ 种语言，甚至处理句内混语（Code-Switching, CS）。

### 1.1 核心矛盾

训练多语种模型本质上是在解决三个冲突：

1. **容量冲突（Capacity）**：模型参数有限，英语学得太好，可能挤占斯瓦希里语的参数空间（Curse of Multilinguality）。
2. **符号冲突（Script）**：中文是表意字，英文是表音字母，日文是音节假名。如何让它们在一个词表中和平共处？
3. **数据不平衡（Imbalance）**：英语数据可能有 50,000 小时，而某些低资源语种（Low-resource language）只有 10 小时。直接混训会导致低资源语种被“淹没”。

---

## 2. 词表与建模单位：统一的基石

在单语种模型中，你只需要关心 "Character vs BPE"。在多语种模型中，词表（Vocabulary）设计决定了模型的上限。

### 2.1 常见策略对比

| 策略 | 说明 | 适用场景 | 致命缺陷 |
| --- | --- | --- | --- |
| **Union of Characters** | 将所有语言的字符取集 | 早期 E2E 模型 | 加上中文/日文/韩文后，输出层（Softmax）巨大（>10k），且拉丁语系序列过长，导致 Loss 难以平衡。 |
| **Unified Subword (BPE/SPM)** | **工业界主流**。在混合语料上训练 SentencePiece | Conformer / Transducer /常规 ASR | 需要精心设计的采样策略，否则低资源语言会被切碎成字符甚至 Unknown。 |
| **Byte-level** | 直接对 UTF-8 字节建模（256 类） | Whisper / MLLM | 序列长度膨胀（中文 1 字 = 3 Bytes）。需要强力的 Context 建模能力（Transformer）。 |
| **Phoneme (IPA)** | 统一映射到国际音标 | 学术研究 / 语言学分析 | 需要所有语种的高质量 G2P（Grapheme-to-Phoneme），工程落地极难。 |

### 2.2 详解：Unified SentencePiece 的构建配方

这是目前最“稳”的方案。构建一个多语种 Tokenizer 并不是简单地 `cat all_text.txt | spm_train`。

#### 步骤 1：数据重采样（Resampling for Tokenizer）

**不要**直接使用训练数据的自然分布来训练 Tokenizer。否则，你的词表里全是英文单词（如 "the", "ing"），而泰语或印地语全被打散成单字符。

* **做法**：为 Tokenizer 训练准备一份独立的文本文件。在这份文件中，**强制让每种语言的句子数量大致相等**。
* **目的**：确保低资源语种的常见词根（Subword）也能进入词表，保证所有语言的编码效率（Compression Rate）接近。

#### 步骤 2：强制字符覆盖（Character Coverage）

SentencePiece 默认覆盖率可能是 0.9995，这会丢弃极低频字符。

* **做法**：设置 `character_coverage=1.0`，或者手动提取所有语种的**基础字符集（Alphabet/Syllabary/Kanji）**，作为 `user_defined_symbols` 或强制包含列表传入。
* **注意**：对于中文/日文汉字，通常需要根据频次截断（如保留前 3000-5000 常用字），非常用字允许回退到 `<UNK>` 或拆解。

#### 步骤 3：处理“共享”与“独占”

* **Latin Script**：英、法、德、西语共享字母。这是好事，有助于迁移（Transfer）。
* **Han Script**：中、日、韩（部分）共享汉字。
* *陷阱*：日文汉字“骨”（Bone）与简体中文“骨”写法微殊（Unicode 码点可能相同也可能不同，取决于 CJK 归并标准）。
* *建议*：如果数据量足够，**不要**刻意区分语言标签（即不要变成 `zh_骨` 和 `ja_骨`），让模型通过上下文去学读音差异。



#### 步骤 4：字节级回退（Byte Fallback）

这是 SentencePiece 的一个重要特性。当遇到 OOV 时，不输出 `<UNK>`，而是输出该字符的 UTF-8 字节序列。这对于多语种系统的鲁棒性至关重要。

---

## 3. 数据平衡：多语种训练的调节阀

一旦数据进入 DataLoader，核心问题就是：**在一个 Batch 里，各种语言该占多少比例？**

### 3.1 温度采样 (Temperature Sampling) —— 行业标准

假设有  个语种，第  个语种的数据量占比为 。我们训练时重采样的概率  计算如下：

其中  是温度（Temperature）：

* ** (Natural)**：原始分布。高资源语种（如英文 10k 小时）主导梯度，低资源语种（10小时）几乎不更新参数。
* ** (Uniform)**：均匀分布。所有语种概率相等。
* *风险*：10 小时的数据会被重复采样 1000 次，导致严重的过拟合（Overfitting）；10k 小时的数据没学完。


* ** (Heuristic)**：这是一个经验上的“甜蜜点”（Sweet Spot），常用于 Multilingual BERT 和 ASR。它提升了低资源语种的可见度，同时保留了高资源语种的多样性。

> **Rule of Thumb (动态调整策略)**
> 不要整个训练周期只用一个 。
> 1. **Warmup 阶段**：使用  或 。先利用高资源语种把声学特征提取器（Encoder）训练稳定。
> 2. **Main 阶段**：切换到 。开始拉升低资源语种性能。
> 3. **Finetune 阶段**：如果某些语种仍不达标，可以对其进行专项 Upsampling。
> 
> 

### 3.2 批次构建策略 (Batch Construction)

* **Mixed Batch**：一个 Batch 内包含不同语种的样本。
* *优点*：梯度的方向是多语种平均的，训练更稳定。
* *缺点*：需要处理不同语种的 Padding 浪费（如果语种间平均时长差异大）。


* **Homogeneous Batch**：一个 Batch 只包含一种语言，但 Batch 之间按  轮换。
* *优点*：实现简单，便于针对特定语种加 Adapter。
* *缺点*：梯度方向可能震荡。



---

## 4. 架构演进：LID 与容量扩张

### 4.1 语言识别 (LID) 的处理

模型如何知道当前是哪种语言？

1. **One-hot ID (Input)**：在声学特征（Mel-spectrogram）后拼接一个全时序的 Language Embedding。
2. **Start Token (Output)**：Decoder 的第一个 Token 预测（或强制输入）`<|zh|>` 或 `<|en|>`。
* *Whisper 模式*：这是目前最流行的做法。它允许用户通过 Prompt 强制指定输出语言（例如做语音翻译）。


3. **End-to-End LID**：模型不仅输出文本，还作为一个分类任务输出 LID。这种多任务学习（Multi-task Learning）有助于分离语种特征。

### 4.2 适配器 (Adapters) —— 解决容量瓶颈

当语种超过 50 种，单一模型的参数容量（Capacity）开始捉襟见肘。

* **Language-Specific Adapters**：
保持主干（Backbone）冻结或共享，为每个语种插入小的 Adapter 模块（通常是 bottleneck linear layers）。
* 位置：通常在 Feed-Forward Network (FFN) 后，或 Self-Attention 旁。
* *优点*：新增一种语言只需要微调 Adapter，不影响其他语言。



### 4.3 Mixture of Experts (MoE) —— 终极方案

对于超大规模模型（MLLM），MoE 是标配。

* **原理**：将 FFN 层替换为多个“专家”网络。对于每个 Token（或每一帧），通过一个 Gating Network 决定激活哪几个专家。
* **在多语种中的意义**：模型会自动学会将“中文专家”、“罗曼语族专家”分离开。推理时，处理中文帧只激活中文相关的参数，**计算量不增加，但模型总容量增加了数十倍**。

---

## 5. 混语 (Code-Switching) 专项：最难的骨头

“帮我 *check* 一下这个 *bug* 怎么 *fix*。” —— 这种句内混语（Intra-sentential CS）是亚洲职场和生活中的常态，也是 ASR 的噩梦。

### 5.1 难点剖析

1. **声学边界模糊**：发音人说英文单词时往往带有母语口音（Chinglish, Japanglish），音素发生形变。
2. **语言模型困惑**：中文后面接英文单词，打破了单语种 LM 的概率分布。
3. **标注数据匮乏**：高质量的 CS 数据（如 SEAME, TAL-CS）非常少。

### 5.2 数据合成策略 (Data Synthesis)

既然真数据少，就必须造假数据。

* **Text-only Synthesis (for LM/MLLM)**:
* 利用规则或 LLM，将大量纯中文文本中的实体词（Entity）替换为英文。
* *Prompt*: "将这句话里的技术名词替换为英文：'由于内存溢出导致程序崩溃' -> '由于 OOM 导致 App Crash'。"


* **TTS Augmentation (for Audio)**:
* 使用支持多语种的 TTS 引擎（如 VITS, Bark, Tortoise），输入混语文本生成音频。这是目前提升 CS 性能**最有效**的手段。


* **Audio Splicing (慎用)**:
* 强行拼接中文音频和英文音频。效果通常很差，因为缺乏自然的协同发音（Co-articulation）和韵律过渡。



### 5.3 混语评测：MER (Mixed Error Rate)

**千万不要用单纯的 CER 或 WER 来评测混语模型。**

* **问题**：如果用 CER，英文单词 "hello" 算 5 个字；如果用 WER，中文 "你好" 可能被当做一个词（这就依赖分词器的质量）。
* **解决方案**：MER (Mixed Error Rate)。
* **计算算法**：
1. **预处理**：扫描 Ref 和 Hyp 文本。
2. **分类切分**：
* 遇到 CJK 字符：按**字**切分（Character-level）。
* 遇到 Latin/Numeric 字符：按**词**切分（Word-level，通常以空格为界）。


3. **对齐计算**：基于切分后的 List 计算 Levenshtein Distance
4. **示例**：
* Ref: `我 需要 verify 账号` (Len=4: 我, 需要, verify, 账号)
* Hyp: `我 需要 very fast 账号` (Len=5: 我, 需要, very, fast, 账号)
* Error: 1 sub (verify->very), 1 ins (fast)。





---

## 6. MLLM 时代的借鉴与新问题

在 MLLM（如 GPT-4o, Qwen-Audio）时代，多语种训练依然遵循上述物理规律，但出现了一些新特性。

### 6.1 语言漂移 (Language Drift / Hallucination)

MLLM 有时会在长语音转写中“忘记”当前的任务是 ASR，开始做翻译或续写。

* **现象**：音频是中文，模型写着写着变成了英文翻译。
* **原因**：预训练数据中包含大量翻译对，或者 Instruction Tuning 时的指令不够明确。
* **对策**：
* **Prompt Engineering**：明确指令 `Transcribe the following audio verbatim in its original language.`
* **Negative Constraint**：在 Loss 中惩罚非目标语言的 Token（如果已知语种）。



### 6.2 脚本/文字系统规范化

MLLM 的输出极其自，它可能将粤语语音转写为书面语（Standard Chinese），或者将英文数字 "one hundred" 写成 "100"。

* 这在传统 ASR 叫“错误”，在 MLLM 叫“特性”。
* **回扣 Chapter 4**：必须建立严格的 ITN（Inverse Text Normalization）评测标准，否则无法衡量 MLLM 的真实准确率。

---

## 7. 本章小结

1. **词表是地基**：使用 Unified SentencePiece，必须做**数据重采样**来平衡各语种的 Subword 粒度。
2. **采样是杠杆**：使用  的温度采样策略，动态平衡高/低资源语种的训练权重。
3. **混语需特制**：Code-Switching 依赖合成数据（TTS/LLM Rewrite）和专门的 MER 评测指标。
4. **架构看规模**：小模型用 Shared Encoder，中模型加 Adapter，大模型上 MoE。

---

## 8. 练习题

### 基础题

1. **采样计算**：假设英文数据 10,000 小时，泰语 100 小时。
* 计算在自然分布（T=1）下，泰语样本出现的概率。
* 计算在 T=5 时，泰语样本出的概率。
* （答案中需体现“为什么 T=5 能救泰语”）。


2. **Tokenizer 陷阱**：在没有重采样的情况下，直接把 90% 的英文和 10% 的中文丢给 SentencePiece 训练，得到的词表会有什么特征？这对中文识别有什么坏处？

### 挑战题

3. **混语指标设计**：编写一个 Python 函数伪代码 `calculate_mer(ref, hyp)`，能够正确处理中英混合文本。要求处理英文大小写不敏感，且忽略多余空格。
4. **架构思考**：你正在设计一个流式会议转写系统，支持中、英、日、韩。为了降低推理延迟，你不能使用大型 MoE。请设计一个基于 Adapter 的方案，并详细说明在**推理阶段**如何决定使用哪个 Adapter（提示：需要一个快速的 LID 模块）。

### 答案与提示

<details>
<summary>点击展开答案</summary>

**1. 采样计算**

* 总时长：10100 小时。
* **T=1 (自然)**: 泰语概率 .
* *后果*：泰语在一个 Epoch 里仅出现极少次，梯度几乎被英文主导。


* **T=5**:
* Weight_En = 
* Weight_Th = 
* Total_Weight = 
* 泰语概率 .
* *结论*：泰语的可见度提升了约 28 倍，模型能有效学习其特征。



**2. Tokenizer 陷阱**

* **特征**：词表会被英文常见 Subword（如 "tion", "ing", "the"）占满。中文由于频率相对低（虽然总数多但单字频率被英文单词稀释），大部分汉字无法合并成词组，甚至很多生僻字被丢弃。
* **坏处**：
1. 中文序列变长（全变成单字），解码慢。
2. 语义建模弱（无法将“人工”+“智能”作为一个整体建模）。
3. 出现 `<UNK>` 的概率极高。



**3. MER 伪代码**

```python
import unicodedata

def is_cjk(char):
    # 简化的 CJK 判断范围
    return '\u4e00' <= char <= '\u9fff'

def tokenize_mixed(text):
    tokens = []
    current_eng_word = []
    
    for char in text:
        if is_cjk(char):
            # 如果之前有英文单词，先flush
            if current_eng_word:
                tokens.append("".join(current_eng_word).lower())
                current_eng_word = []
            tokens.append(char) # 中文按字加
        elif char.strip() == "":
            if current_eng_word:
                tokens.append("".join(current_eng_word).lower())
                current_eng_word = []
        else:
            current_eng_word.append(char)
            
    if current_eng_word:
        tokens.append("".join(current_eng_word).lower())
    return tokens

def calculate_mer(ref, hyp):
    ref_toks = tokenize_mixed(ref)
    hyp_toks = tokenize_mixed(hyp)
    # 此处调用标准的 edit_distance 函数
    ed = levenshtein(ref_toks, hyp_toks)
    return ed / len(ref_toks)

```

**4. Adapter 架构设计**

* **架构**：Shared Conformer Encoder + Language Adapters (in FFN)。
* **推理流程**：
1. **LID 分支**：模型前 5 层 Encoder 输出一个特征，接一个小型的 LID Classifier。
2. **Lookahead**：使用 VAD 切出的 Chunk 的前 0.5 秒音频先过 LID 分。
3. **路由**：
* 若 LID=ZH，加载/激活 ZH-Adapter。
* 若 LID 置信度低，或检测到频繁切换，激活 "General/Mixed Adapter"（训练时需专门训练一个混语 Adapter）。


4. **解码**：后续层使用选定的 Adapter 进行计算。


* **关键点**：LID 必须极快且轻量，否则流式延迟不达标。

</details>

---

## 9. 常见陷阱与错误 (Gotchas)

### 9.1 "脚本泄漏" (Script Leaking)

* **现象**：在训练日文 ASR 时，模型偶尔会输出简体中文特有的汉字（如“发”而不是“發”/“髪”），或者在训练中文 ASR 时输出日文汉字。
* **原因**：词表中混杂了不同来源的汉字，且 Unicode 码点在某些字体下看起来一样，但实际不同。或者训练数据中有脏数据。
* **调试**：使用 `OpenCC` 或 Unicode Range 过滤器清洗训练集，确保日文数据里没有纯简体字，中文数据里没有平假名。

### 9.2 "语种不平衡的灾难性遗忘"

* **现象**：Finetune 一个多语种底座模型到某个小语种上，结果英文能力完全丧失。
* **对策**：始终保留一部分（如 10%）的高资源语种数据（Replay Buffer）在 Finetune 阶段混合训练，或者使用 Adapter 技术冻结底座。

### 9.3 混语合成数据的音色单一

* **现象**：使用单一 TTS 引擎合成大量 CS 数据，模型在真实场景（多人、多口音）下效果依然差。
* **原因**：模型过拟合了 TTS 的特定音色和韵律。
* **对策**：
1. 使用多款 TTS 引擎。
2. 对合成音频做激进的 SpecAugment 和噪声增强。
3. **Code-Switching 文本注入**：只在文本层面做替换（Text injection），强迫模型学会“看到中文字输出英文词”的语言模型概率转移，而不完全依赖声学。
