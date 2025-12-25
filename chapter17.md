# Chapter 17: MLLM 新内容：RAG 热词识别、上下文增强与说话人知识注入

## 17.1 开篇与学习目标

在 ASR 发展的历史长河中（如 Chapter 7-9 所述），"Contextual Biasing"（上下文偏置）一直是一个核心难题。在 WFST 时代，我们通过修改解码图（H-Level 或 G-Level）来硬性插入热词；在端到端（E2E）时代，我们使用 Shallow Fusion 或 Neural Contextual Biasing 模块。

到了 MLLM（Multimodal Large Language Model）时代，游戏规则变了。模型不再仅仅是一个声学概率计算器，而是一个具备推理能力的智能体。ASR 任务逐渐演变成了**“基于音频证据的指令遵循任务”**。我们不再需要费力地修改底层解码图，而通过 **RAG（Retrieval-Augmented Generation）** 将外部知识动态注入到 Prompt 中。

然而，MLLM 也带来了新的风险：它可能因为“过度聪明”而根据上下文编造音频中未出现的内容（幻觉）。本章将深入探讨如何驾驭这股力量，构建既精准又可控的下一代 ASR 系统。

**本章学习目标**：

1. **深入理解 ASR-RAG 的特殊性**：为什么文本 RAG 的“语义检索”在语音热词任务中往往失效？如何构建“声学检索器”？
2. **掌握热词注入的三种范式**：从单纯的 Prompting 到 Logit Bias，再到两阶段（2-pass）修正。
3. **学会利用 Diarization 进行个性化识别**：如何将 Speaker Embedding 转化为 RAG 的检索键。
4. **构建防御性解码策略**：如何通过置信度校验和局部重写（Partial Rewriting）抑制幻觉。
5. **建立新的评测指标**：学习 Bias-WER (B-WER) 和 Unbiased-WER (U-WER) 以量化 RAG 的收益与代价。

---

## 17.2 核心机制：ASR 专用的 RAG 流水线

与传统的文本问答 RAG 不同，ASR 的输入是**模糊的声学信号**或**包含错误的初步转写**。因此，直接使用 BERT/Embedding 进行语义检索通常效果不佳（例如：用户说了“买**也是**”，ASR 识别为“买**椰氏**”，语义向量天差地别）。

我们需要构建一个 **ASR 专用的 RAG 流水线**：

### 17.2.1 检索策略：声学相似度 > 语义相似度

在 ASR 场景下，**“听起来像”**比“意思相近”更重要。

1. **索引构建 (Indexing)**：
* **Key**: 热词本身（如 "DeepSeek"）。
* **Phonetic Key**: 热词的音素序列（如 `/d i p s i k/`）或拼音（`di pu xi ke`）。
* **Fuzzy Key**: 使用 Metaphone, Soundex 或简单的字符 N-gram 生成模糊键。


2. **查询生成 (Query Generation)**：
* **源**: ASR 的初步解码结果（1-best 或 N-best hypothesis）。
* **变换**: 将初步结果转换为音素/拼音序列。


3. **检索匹配 (Retrieval)**：
* 使 **加权编辑距离 (Weighted Edit Distance)** 计算 Query 音素序列与知识库 Key 音素序列的相似度。
* **Rule of Thumb**: 对于缩写词（如 "AI", "App"），字符级匹配权重更高；对于长实体（如 "阿达木单抗"），音素级匹配权重更高。



### 17.2.2 流程架构图

```ascii
[知识库 Preparation]
实体: "Zweihander" -> G2P -> /z w aɪ h æ n d ər/ -> 存入向量库/倒排索引

[Runtime Process]
1. Audio Input  ----(Speech Encoder)----> Acoustic Embeddings
                                              |
2. First Pass   ----(Fast Decoder)------> "Please call two hander team" (ASR Error)
                                              |
3. Phonetic Search (关键步骤)
   Query: "two hander" -> /t u h æ n d ər/
   Match: /t u h æ n d ər/ vs /z w aɪ h æ n d ər/ (Distance < Threshold)
   Retrieve: "Zweihander"
                                              |
4. Context Construction
   Prompt: "The user might be mentioning: ['Zweihander'].
            Audio evidence suggests specific terms. 
            Transcribe faithfully."
                                              |
5. Second Pass (MLLM)
   Input: Acoustic Embeddings + Prompt
   Output: "Please call Zweihander team"

```

---

## 17.3 热词注入与偏置方法演进

在 MLLM 框架下，我们有多种手段让模型“注意到”这些检索回来的热词。

### 17.3.1 Prompting (指令引导)

这是最直接的方法，利用 LLM 的 In-context Learning 能力。

* **Naive Prompt**: "Possible hotwords: [A, B, C]. Transcribe the audio."
* **Structure-Aware Prompt** (推荐):
```markdown
# Instruction
You are an expert ASR system. Below is a list of specialized vocabulary that MAY appear in the audio.

# Vocabulary
- "Retin-A" (Medical drug)
- "Tretinoin" (Generic name)

# Constraints
- Only use the vocabulary if the audio strictly matches the pronunciation.
- Do NOT hallucinate words just because they are in the list.

# Audio Context
[Audio Embeddings inserted here]

```



### 17.3.2 Biased Decoding (Logit 偏置)

如果 Prompting 效果不稳定（模型视而不见），可以在解码层面施加硬约束。这类似于 Shallow Fusion 的 MLLM 版本。

* **原理**: 在 LLM 生成每一个 Token 时，检查该 Token 是否属于检索到的热词的前缀。如果是，人为增加该 Token 的 Logit 值。
* **挑战**: MLLM 的 Tokenizer（如 Tiktoken, SentencePiece）可能将一个热词切碎（如 `TensorFlow` -> `Ten`, `sor`, `Flow`）。必须构建 **Trie (前缀树)** 来通过 Token 边界进行匹配。

### 17.3.3 Tool-Use / Agentic ASR (工具调用模式)

这是 MLLM 的高级用法。将“查词典”作为一个 Tool。

* **流程**:
1. MLLM 遇到不确定的发音，输出特殊 Token `<lookup> sound_like_xxx </lookup>`。
2. 外部程序捕获该 Token，执行模糊检索。
3. 将检索结果填回 Prompt，MLLM 继续生成。


* **优点**: 极大地减少了 Prompt 长度，只在需要时检索。

---

## 17.4 说话人知识注入 (Speaker-aware RAG)

利用 **Chapter 10/11** 的 Diarization 结果，我们可以将“说话人身份”作为最强的上下文线索。

### 17.4.1 说话人画像 (Speaker Profile)

我们可以为每个 Speaker ID 维护一个动态 Profile：

* **静态属性**: 姓名、职位、部门（决定了专用术语表）。
* **动态历史**: 本次会议中该人已经说过的专有名词（Cache）。

### 17.4.2 实现方案：Role-based Prompting

假设 Diarization 告诉我们：`00:00 - 00:15` 是 `Speaker_A` (Doctor)，`00:16 - 00:30` 是 `Speaker_B` (Patient)。

**Prompt 模板**:

```text
[Role Definition]
Speaker_A is a Cardiologist. Expect medical terminology (drugs, procedures).
Speaker_B is a Patient. Expect colloquial language, symptom descriptions.

[Transcription Task]
Turn 1 (Speaker_A): "Have you been taking your [MASK]?" -> Bias towards "Warfarin", "Aspirin".
Turn 2 (Speaker_B): "Yes, I take the [MASK] one." -> Bias towards "red", "small" (common words).

```

这种方法能极大地解决**同词消歧**问题（例如：医生说的 "MS" 可能是 "Mitral Stenosis" 二尖瓣狭窄，而 IT 人员说的 "MS" 是 "Microsoft"）。

---

## 17.5 常见陷阱与幻觉控制 (Gotchas & Anti-Hallucination)

MLLM ASR 最大的噩梦是：知识库里有 "Apple"，用户说了 "Apply"，模型强行转写为 "Apple"。

### 17.5.1 幻觉成因

1. **Over-trusting context**: 模型在预训练阶段学会了“尽可能利用 Prompt 信息”，导致忽略底层的声学证据。
2. **Tokenization artifact**: 某些罕见热词被切分成极短的 token，导致概率分布极其平坦，容易被 Logit Bias 干扰。

### 17.5.2 防御工程 (Defense Engineering)

1. **Confidence-based Selection (置信度门控)**:
* 不要盲目接受 MLLM 的修正。
* 计算 `Score(RAG_Result)` 和 `Score(Original_Result)`。只有当 `Score(RAG_Result) - Score(Original_Result) > Threshold` 时才采纳。


2. **Negative Prompting (负向提示)**:
* 明确告诉模型什么**不要做**。
* Prompt: "If the audio sounds like 'Apply' (/ə p l aɪ/), do NOT output 'Apple' (/æ p l/) even if 'Apple' is in the list."


3. **Anchor Constraint (锚点约束)**:
* 要求 MLLM 输出时必须携带时间戳。如果热词的时间戳与原音频段严重不符（例如时长差异过大），则丢弃该热词。


4. **The "None of the Above" Option**:
* 在热词列表中始终加入一个 `<NO_MATCH>` 选项，训练模型在声学不匹配时主动选择它。



---

## 17.6 评测体系：不仅仅是 WER

传统的 WER (Word Error Rate) 无法准确衡量 RAG 的价值。我们需要更细粒度的指标。

### 17.6.1 专项指标

1. **R-WER (Reference-WER)**: 仅计算热词列表中的词的错误率。
2. **B-WER (Biased-WER)**: 给定热词提示时，热词的召回率（Recall）。目标：越高越好。
3. **U-WER (Unbiased-WER)**: 给定热词提示时，**非热词**部分的错误率。目标：不要升高（即不发生“灾难性遗忘”或“误伤”）。
4. **FPR (False Positive Rate)**: 这是一个关键指标。当 Prompt 中包含热词 X，但音频中**完全没说** X 时，模型幻觉出 X 的概率。

### 17.6.2 对比实验设计

| 实验组 | Prompt 设置 | 预期结果 | 风险 |
| --- | --- | --- | --- |
| **Baseline** | 无热词列表 | 高 B-WER | 漏识别专名 |
| **Oracle** | 仅包含音频中真实出现的热词 | 理论上限 (Upper Bound) | 无法落地 |
| **RAG-TopK** | 包含 Top-K 检索结果 (含干扰项) | 真实性能 | 可能触发幻觉 |
| **Distractor** | 仅包含干扰项 (音频中未出现的词) | 评估 FPR | 测试抗干扰能力 |

---

## 17.7 练习题

### 基础题

<details>
<summary><b>1. 为什么直接把几千个员工名字全部塞进 MLLM 的 System Prompt 是不可行的？</b> (点击展开)</summary>

* **提示**：考虑上下文窗口（Context Window）和注意力分散（Attention Dilution）。
* **答案**：
1. **成本与延迟**：Prompt 越长，首 token 延迟（TTFT）越高，推理成本越贵。
2. **注意稀释**：即 "Lost in the Middle" 现象。当列表过长时，模型往往只能记住开头和结尾的词，中间的词会被忽略。
3. **误报率激增**：候选词越多，撞上“音似词”导致幻觉的概率就越大。



</details>

<details>
<summary><b>2. 在构建 ASR 专用的 RAG 检索器时，为什么 "Soundex" 或 "Metaphone" 算法比 "BERT Embedding" 更有效？</b> (点击展开)</summary>

* **提示**：ASR 的错误通常是基于什么的？
* **答案**：ASR 的初步错误通常是**发音驱动**的（如 "two hander" vs "Zweihander"）。Soundex/Metaphone 是专门将发音编码为代码的算法，能捕捉声学相似性。而 BERT Embedding 基于语义，"two hander" 和 "Zweihander" 在语义空间距离极远，无法被召回。

</details>

### 挑战题

<details>
<summary><b>3. [系统设计] 设计一个“流式（Streaming）”的 MLLM RAG 方案。由于 MLLM 推理慢，如何避免用户等待太久？</b> (点击展开)</summary>

* **提示**考虑“推测解码”（Speculative Decoding）或“异步修正”架构。
* **答案**：
* **双流架构 (Dual-Stream)**：
1. **Fast Stream**: 使用小模型（如 Conformer-Transducer）实时上屏，延迟低，但可能有错。
2. **Slow Stream (Async)**: 后台异步运行 MLLM + RAG。每隔 5-10 秒（或一个句子结束），对 Fast Stream 的结果进行“回溯修正”。


* **UI 呈现**: 也就是常说的 "Finalization" 机制。用户先看到灰色的初步结果（变动中），几秒后文字变为黑色（MLLM 修正定稿）。



</details>

<details>
<summary><b>4. [Prompt 工程] 编写一个用于“粤语-英语混读”会议记录的 RAG Prompt 模板，要求模型修正专有名词，但保留粤语语气词（如“嘅”、“啫”）。</b> (点击展开)</summary>

* **提示**：利用 Chapter 4 的混语处理知识，强调“Style Preservation”。
* **答案**：
```markdown
# Role
You are a Cantonese-English code-switching ASR assistant.

# Context (Hotwords)
- "Docker"
- "Kubernetes"

# Rules
1. **Correction**: If the audio sounds like "Dock 㗎", and context is technical, correct "Dock" to "Docker".
2. **Preservation**: STRICTLY PRESERVE Cantonese sentence-final particles (SFPs) like "嘅", "啫", "啦". Do NOT translate them or remove them.
3. **Output**: Output the transcribed text directly.

# Example
Input Audio: "我地用 Kubernete 嘅"
Output: "我地用 Kubernetes 嘅"

```



</details>

<details>
<summary><b>5. [诊断] 你的 RAG ASR 系统上线后，用户投诉：“我没说这个药名，它自己加上去了”。请列出 3 种排查思路。</b> (点击展开)</summary>

* **提示**：检查检索环节、Prompt 环节和音频质量。
* **答案**：
1. **检查检索阈值**：是否检索模块的编辑距离阈值设得太宽？导致用户说个普通词（如 "aspirational"）召回了热词（"Aspirin"）。
2. **检查 Prompt 强弱**：是否使用了过于强硬的指令（如 "You MUST use words from the list"）？应改为 "Only use if audio matches"。
3. **检查 Logit Bias**：如果使用了 Logit Bias，是否不仅给热词首 Token 加分了，还给后续 Token 加分过高？导致一旦触发前缀就“刹不住车”。



</details>

---

## 17.8 本章小结：迈向可控的智能语音

Chapter 17 标志着我们从“训练模型”转向“使用模型”。

* **ASR + RAG** 不仅仅是查漏补缺，它是通向**领域自适应（Domain Adaptation）**的捷径，无需重新训练模型即可适配医疗、法律等垂直场景。
* **Prompt Engineering** 是新的 Feature Engineering。你需要像对待代码一样对待 Prompt，进行版本管理和回归测试。
* **核心矛盾** 依然是 **召回率 vs. 幻觉**。所有的工程技巧（声学检索、置信度校验、负向提示）都是为了在这个权衡中寻找最优解。
