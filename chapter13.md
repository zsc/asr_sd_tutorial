# Chapter 13: 评测与误差分析——ASR (WER/CER/MER) 与 Diarization (DER/JER) 的细节陷阱

## 13.1 开篇与学习目标

在语音识别（ASR）和说话人日志（Diarization）的开发周期中，评测（Evaluation）往往是最容易被低估，却最容易导致项目失败的环节。**“跑出一个指标”很容易，但“跑出一个可信、可比、能指导优化的指标”很难**。

初学者常犯的错误包括：被文本规范化（Normalization）差异误导，误以为模型提升了；或者在 Diarization 中忽视了 Collar（容差）设置，导致无法与 SOTA 论文对齐。而在 MLLM 时代，传统的“字面匹配”更面临着“语义正确但字面不同”的巨大挑战

**本章学习目标**：

1. **掌握核心算法**：不仅是公式，更要理解 WER/CER/MER 的对齐逻辑（Levenshtein）与 DER 的最佳映射（Hungarian Algorithm）。
2. **攻克文本规范化（TN）**：建立一套工业级的 Eval Pipeline，解决大小写、标点、数字（1 vs one）、简繁体对指标的干扰。
3. **多语种与混语（Code-Switching）**：掌握混合错误率（MER）的计算逻辑，处理中英日混杂场景。
4. **Diarization 的“黑魔法”**：透彻理解 Collar、Overlap、VAD 对 DER 的蝴蝶效应。
5. **MLLM 与 RAG 新评测**：学习如何评测语义一致性、幻觉率（Hallucination Rate）以及热词召回率。

---

## 13.2 ASR 核心指标：从编辑距离到混合策略

### 13.2.1 基础：Levenshtein Distance 与 WER

ASR 评测的核心是计算**假设文本（Hypothesis, Hyp）**与**参考文本（Reference, Ref）**之间的差异。这本质上是一个动态规划问题，即寻找最小编辑距离。

* **S (Substitution, 替换)**：把 `cat` 认成了 `cap`。这是声学模型最常见的错误。
* **D (Deletion, 删除)**：漏识别了某个词。通常发生在语速极快或声音极小处。
* **I (Insertion, 插入)**：多识别了不存在的词。通常由背景噪声或长时间静音（导致模型强行解码）引起。
* ****：参考文本的总词数。
* **关键点**：分母必须是 **Reference** 的长度，绝不能是 Hypothesis 的长度。
* **推论**：WER 可以超过 100%。如果 Ref 是 "No"，Hyp 是 "No no no no no"，则 。



### 13.2.2 粒度选择：Word (WER) vs Character (CER)

| 语种/场景 | 指标 | 原因 | 示例 |
| --- | --- | --- | --- |
| **英语/印欧语系** | **WER** | 单词是最小语义单位，空格天然分词。 | `apple` 错成 `apply` 算 1 个错，不算 2 个字符错。 |
| **中文** | **CER** | 汉字之间无空格，分词标准（结巴/HanLP）不统一会导致指标波动。 | “南京市长”若分词为“南京/市长”或“京市/长”，会影响 WER，但 CER 恒定。 |
| **日语** | **CER** | 混合了假名和汉字，通常按字符（Character）计算。 | `ご飯` (2 chars) vs `ごはん` (3 chars) 的对齐需特殊处理（见下文）。 |
| **韩语** | **CER/WER** | 视业务而定。韩语有空格（Eojeol），但形态变化复杂，通常推荐 CER 或字/词素级 WER。 | - |

### 13.2.3 进阶：混合错误率 (MER, Mixed Error Rate)

在 **Code-Switching（如中英混杂）** 场景下，单纯用 WER 或 CER 都不公平：

* 用 WER：中文分词错误（如“人工智能”切成“人工 智能”）会被误判为 ASR 错误。
* 用 CER：英文单词 `Constitution` 被拆成 12 个字母，权重是中文汉字 `宪` 的 12 倍，导致英文错误主导指标。

**MER 计算逻辑：**

> **Rule of Thumb**: “见到 CJK 字符按字切，见到 Latin 字符串按词切。”

**算法步骤演示**：

```python
Ref: "我 在 Office 开 会"
Hyp: "我 在 office 开会"

Step 1: Tokenization (智能切分)
Ref Tokens: ['我', '在', 'Office', '开', '会']  (5 tokens)
Hyp Tokens: ['我', '在', 'office', '开', '会']  (5 tokens, "开会"被拆开)

Step 2: Normalization (规范化)
Ref: ['我', '在', 'office', '开', '会'] (Lowercased)
Hyp: ['我', '在', 'office', '开', '会']

Step 3: Levenshtein Alignment
Result: Match=5, Error=0. MER=0.0%

```

---

## 13.3 文本规范化（TN）：评测的“隐形杀手”

**如果评测脚本没有统一的 TN Pipeline，任何 WER 的改进都是不可信的。**

### 13.3.1 必须处理的规范化层级

建议构建一个标准化的 `normalize_for_eval(text)` 函数，包含以下层级：

1. **Unicode 规范化 (NFKC)**：
* 解决全角/半角问题：`Ａ` → `A`, `１` → `1`。
* 解决组合字符问题：`e` + `´` → `é`。


2. **大小写 (Casing)**：
* 通常转小写（Lowercasing）。
* *例外*：若评测实体识别（如 `Apple` 公司 vs `apple` 水果），则需保留。


3. **标点符号 (Punctuation)**：
* **删除策略**：大多数 ASR 评测会移除 `,.?!;:"` 等符号。
* **保留策略**：如果模型是 "Rich Transcription"（含标点预测），则标点应作为独立 Token 参与 WER 计算。
* **陷阱**：中文的顿号 `、` 和英文逗号 `,` 在混语中极易混淆，建议全部映射为空格。


4. **空白符整理**：
* `strip()` 去首尾，`re.sub(r'\s+', ' ', text)` 将连续空格合并为一个。



### 13.3.2 棘手的“数字与多义性”问题

| 原始文本 | ASR 输出可能 | 评测策略 |
| --- | --- | --- |
| **Ref**: `I have 2 apples` | `I have two apples` | **策略 A (Verbalization)**: 将 Ref 中的 `2` 转写为 `two`，再对比。<br>

<br>**策略 B (ITN)**: 将 Hyp 中的 `two` 转为 `2`，再对比。 |
| **Ref**: `1990年` | `一九九零年`<br>

<br>`一千九百九十年` | **困境**：年份读法不唯一。<br>

<br>**解决**：通常使用策略 B（ITN），将中文数字转回阿拉伯数字进行评测最稳健。 |
| **Ref**: `ok` | `OK`<br>

<br>`Okay`<br>

<br>`okay` | **解决**：建立同义词表（Synonym Map）或在规范化阶段统一替换。 |

**工业界最佳实践**：

* **声学模型迭代期**：偏向 **Verbalization**（Ref 转文字）。因为我们要测模型“听”得准不准，而不是由 ITN 模块负责的格式对不对。
* **产品交付期**：偏向 **ITN**（Hyp 转数字）。因为用户最终看到的是“2025年”，而非“二零二五年”。

---

## 13.4 Speaker Diarization 关键指标：DER 与 JER

### 13.4.1 DER (Diarization Error Rate) 深度解析

DER 是时间维度的错误率，不关心说话人是谁，只关心“这段时间谁在说话”这个标签对不对。

1. **Missed Speech (漏检)**：Ref 说有人，Hyp 说没人（被判为静音）。原因：VAD 阈值过高、声音太小。
2. **False Alarm (FA, 误检)**：Ref 说没人，Hyp 说有人（把噪声当语音）。原因：VAD 阈值过低、呼吸声/键盘声没过滤。
3. **Speaker Confusion (混淆)**：Ref 是 Spk A，Hyp 是 Spk B。这是 Diarization 模型的核心错误。

### 13.4.2 关键参数：Collar（容差/缓冲带）

由于人工标注很难精确到毫秒级，且模型平滑处理会导致边界抖动，直接评测会产生大量无意义的 Miss/FA。

* **无 Collar (0ms)**：极其严苛，适用于合成数据。
* **标准 Collar (250ms)**：NIST / DIHARD 标准。**即：参考边界前后 250ms 内的误差不计入 DER。**
* *Gotcha*：在计算 Confusion 时，Collar 区域内的混淆通常也被移除，只看 Collar 之外的稳定区域。
* **实现细节**：如果不设置 Collar，你的 DER 可能比论文高 10%~20%。



### 13.4.3 最佳映射 (Optimal Mapping)

模型输出是匿名的 `Cluster_0`, `Cluster_1`，参考是 `Bob`, `Alice`。如何对应？
评测脚本会构建一个二分图，边的权重是两个说话人的重叠时长，使用 **匈牙利算法 (Hungarian Algorithm)** 求最大权匹配。

### 13.4.4 JER (Jaccard Error Rate) —— 为“少言”伸张正义

**DER 的缺陷**：分母是总时长。如果会议里 Leader 说了 50 分钟，Intern 只说了 1 分钟。哪怕模型把 Intern 的 1 分钟全部分错，对总 DER 的影响也微乎其微。

**JER (Jaccard Error Rate)**：

1. 对每个 Reference Speaker，找到与其重叠最大的 Hypothesis Speaker。
2. 计算该配对的 Jaccard Index（交并比）。
3. 计算所有 Reference Speakers 的 Jaccard 错误率 JER_i 的**平均值**。
4. **特点**：每个说话人（无论话多话少）权重相等。

---

## 13.5 MLLM 时代的 ASR/Diarization 评测新视角

对于 Whisper、Qwen-Audio、Gemini 等模型，传统的 WER/DER 开始失效。

### 13.5.1 语义一致性 (Semantic Similarity)

MLLM 往往会进行“隐式 ITN”或“润色”。

* Ref: "I, uh... want to um, go."
* Hyp: "I want to go."
* **WER**: 很高（因为删除了 uh, um）。
* **真实体验**：很好。

**新指标：BERT-Score / Semantic Embedding Distance**
使用 Text Embedding 模型（如 `e5` 或 `openai-ada-002`）提取 Ref 和 Hyp 的向量，计算余弦相似度。如果相似度 > 0.95，即使 WER 高，也视为正确。

### 13.5.2 幻觉率 (Hallucination Rate)

MLLM 在长时间静音或音乐段容易“发疯”，输出重复的“Thank you for watching”或无关文本。

**检测算法**：

1. **长度比异常**：（且 Ref 不为空）。
2. **重复检测**：检测 Hyp 中 n-gram 的重复频率（如 "ok ok ok ok"）。
3. **特定词监控**：监控训练数据中常见的无关词（Subtitle credits, YouTube descriptions）。

### 13.5.3 RAG 热词召回评测 (Contextual Biasing Eval)

当使用 RAG 注入热词（如人名、药名）时，不能只看整体 WER（可能因为整体数据量大而被稀释）。

需定义 **Keyword-Specific Metrics**：

* **Recall (召回率)**：Ref 中的热词，Hyp 中出现了多少？
* **Precision (准确率)**：Hyp 中预测出的热词，有多少是对的？
* **Bias Effect (偏置效应)**：热词注入是否导了周围词的 WER 升高（例如为了强行匹配“张三”，把“涨散”错认成“张三”）。

---

## 13.6 本章小结

1. **ASR 核心**：WER = (S + D + I) / N。必须注意分母是 Ref。
2. **TN 是基石**：评测前必须统一全半角、标点、大小写。中文用 CER，英文用 WER，混语用 MER。
3. **Diarization 三巨头**：DER（时间错误）、JER（说话人公平）、Collar（容差）。不要在未声明 Collar 的情况下对比 DER。
4. **MLLM 挑战**：从“字面匹配”转向“语义匹配”与“幻觉控制”。

---

## 13.7 练习题

### 基础题 (50%)

<details>
<summary><strong>Q1: 基础 WER 手算</strong></summary>

**题目**：
Ref: "The cat sat on the mat"
Hyp: "The cat on the mat"

1. 计算 S, D, I 分别是多少？
2. 计算 WER。

> **Hint**: 对齐两者，找出缺少的词。注意 N 是 Ref 的长度。

**答案**：

1. **对齐**：
Ref: The cat **sat** on the mat
Hyp: The cat ******* on the mat
差异：sat 被删除了。
S=0, D=1 (sat), I=0.
2. **计算**：
 (Ref 的词数: The, cat, sat, on, the, mat).
.

</details>

<details>
<summary><strong>Q2: 中文 CER vs WER 的陷阱</strong></summary>

**题目**：
Ref: "南京市长" (Nanjing Mayor)
Hyp: "南京市长江" (Nanjing City Yangtze River - 典型断句错误导致的识别)

假设分词器 A 将 Ref 切分为 `['南京', '市长']`。
假设分词器 B 将 Hyp 切分为 `['南京市', '长江']`。

1. 计算 WER（基于上述分词）。
2. 计算 CER（基于字符）。
3. 哪个指标更能反映声学模型的错误？

**答案**：

1. **WER**:
Ref: [南京] [市长]
Hyp: [南京市] [长江]
对齐后完全不匹配。S=2 (或 S=1, I=1 等，取决于距离计算，但通常是完全错误)。
WER = 2/2 = 1.0。
2. **CER**:
Ref: 南 京 市 长
Hyp: 南 京 市 长 江
Match: 南, 京, 市, 长。
Insertion: 江。
S=0, D=0, I=1, N=4。
CER = 1/4 = 0.25。
3. **结论**：CER (25%) 更合理。声学模型其实只多听了一个音（江），前面都对了。WER (100%) 夸大了错误，因它受到了分词歧义的影响。

</details>

<details>
<summary><strong>Q3: Diarization 映射与 DER</strong></summary>

**题目**：
Ref: Spk_A (0-10s), Spk_B (10-20s)
Hyp: Spk_1 (0-10s), Spk_2 (10-20s)

系统会如何判定 Speaker 对应关系？DER 是多少？

> **Hint**: 匈牙利算法寻找最大重叠时长。

**答案**：

1. **重叠矩阵**：
Spk_1 vs Spk_A: 10s 重叠。
Spk_2 vs Spk_B: 10s 重叠。
2. **映射**：Spk_1 -> Spk_A, Spk_2 -> Spk_B。
3. **错误计算**：
Miss = 0, FA = 0, Conf = 0。
DER = 0.0%。

</details>

<details>
<summary><strong>Q4: 文本规范化的影响</strong></summary>

**题目**：
Ref: "It's 10:00 p.m."
Hyp: "it is ten pm"

如果不做任何 Normalization，WER 是多少？
如果做了完善的 Normalization，WER 是多少？

**答案**：

1. **无 Normalization**:
Ref tokens: "It's", "10:00", "p.m." (3 tokens)
Hyp tokens: "it", "is", "ten", "pm" (4 tokens)
对齐：It's(S)->it, (I)->is, 10:00(S)->ten, p.m.(S)->pm.
WER 约 133%（可能 >100%，视 tokenizer 而定）。
2. **有 Normalization**:
Ref -> expansion -> "it is ten pm"
Hyp -> expansion -> "it is ten pm"
WER = 0%.

</details>

### 挑战题 (50%)

<details>
<summary><strong>Q5: 混语 MER Tokenizer 设计</strong></summary>

**题目**：
请设计一个 Python 函数逻辑（伪代码），实现 MER 的分词标准：CJK 按字，英文按词。
输入：`"I love 中国"`
期望输出：`['I', 'love', '中', '国']`

> **Hint**: 遍历字符，检查 Unicode 范围。

**答案**：

```python
def mixed_tokenize(text):
    tokens = []
    buffer = "" # 用于累积英文字符
    for char in text:
        if is_cjk(char): # 判断是否为中日韩字符
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append(char) # CJK 字符单独成词
        elif char.strip() == "": # 空格
            if buffer:
                tokens.append(buffer)
                buffer = ""
        else: # 英文或其他字符
            buffer += char
    if buffer:
        tokens.append(buffer)
    return tokens

```

</details>

<details>
<summary><strong>Q6: RAG 热词评测的准确率陷阱</strong></summary>

**题目**：
你为一个医疗 ASR 系统加入了一个含有 10,000 个生僻药名的热词表。
评测结果显示：

* 热词 Recall 从 20% 提升到了 80%。
* 总体 WER 保持不变 (10%)。
* 但医生投诉“很多常用词被识别成了奇怪的药名”。

请解释为什么 WER 没变但体验变差了？如何量化这个问题？

**答案**：

1. **原因**：这是典型的 **False Positive (误报)** 问题。
* 模型确实找回了更多的药名 (Recall 提升)。
* 但模型变得过于敏感（Biased），把发音相近的常用词（如“要吃饭”）强行识别成了生僻药名（如“药赤凡”）。
* 由于生僻药名在测试集中出现频率低，而常用词出现频率高，少量的常用词错误被“召回率提升带来的正确字数”抵消了，导致总体 WER 看起来没变。


2. **量化方法**：
* 计算 **热词 Precision**：Hyp 中出现的药名，有多少是真的？
* 计算 **非热词区域 WER**：只计算 Reference 中不包含药名的句子的 WER。如果这个指标上升，说明热词注入破坏了通用模型的性能。



</details>

<details>
<summary><strong>Q7: 幻觉检测算法设计</strong></summary>

**题目**：
Whisper 模型在一段 30秒 的静音音频中，输出了重复 50 次的 "You"。
WER 计算为 Insertion Error。
请设计一个简单的规则过滤器，自动标记这类样本。

**答案**：
**规则组合**：

1. **Compression Ratio (压缩比)**：计算 `gzip(text) / len(text)`。如果文本包含大量重复内容（如 "You You You"），压缩比会非常高（体积变得很小）。
2. **Distinct N-gram Fraction**：计算 `len(set(ngrams)) / len(ngrams)`。如果该值极低（例如 < 0.1），说明词汇多样性极低，极大概率是重复幻觉。
3. **Log-Probability**：检查模型输出的 `avg_logprob`。幻通常伴随着较低或异常分布的置信度（虽然不总是）。

</details>

<details>
<summary><strong>Q8: Overlap 对 DER 的理论上限</strong></summary>

**题目**：
一段 10分钟的会议音频，其中有 10% 的时间是两人同时说话（Overlap）。
你的 Diarization 系统不支持 Overlap 检测（即同一时刻只能输出一个说话人）。
请问你的 DER 理论下限（最好结果）是多少？（假设 VAD 和 Speaker Clustering 完美）。

> **Hint**: 在 Overlap 区域，Ref 有 2 个人，Hyp 只有 1 个人。

**答案**：

1. **分析**：
* 总时长 600s。
* Overlap 时长 60s。
* 在 Overlap 区域，Ref 人数 = 2，Hyp 人数 = 1。
* 每一时刻，模型都漏掉了一个人 (Missed Detection)。
* Missed Speech Duration = 60s。


2. **分母计算**：
* Total Speech in Ref = (单人说话时长) + (双人说话时长 × 2)
* 假设剩余 90% 是单人说话（不考虑静音以便简化，或者假设全是语音）：
* Total Speech = 540s + 120s = 660s。


3. **DER 计**：
* DER = 60 / 660 ≈ 9.1%。


4. **结论**：即使其他部分完美，不支持 Overlap 的系统在包含 10% 重叠的数据上，DER 最好也只能达到 ~9.1%。

</details>

---

## 13.8 常见陷阱与错误 (Gotchas)

### 1. `sclite` vs `compute-wer.py`

* **现象**：你自己写的 Python 脚本算出的 WER 和 NIST 的 `sclite` 工具算出的不一样。
* **原因**：`sclite` 支持处理 Reference 中的**可选分支**（如 `(ok|okay)`）和更复杂的对齐逻辑。
* **建议**：学术论文对比务必使用 `sclite` 或标准的 `jiwer` 库，不要依赖手写脚本。

### 2. 空 Reference 导致的 Crash

* **现象**：VAD 切分出的某些片段完全是噪音，Reference 为空字符串。计算 WER 时分母为 0。
* **修复**：
* **单句级别**：如果 N=0 且 Hyp 非空，则 WER 无法定义（或视为无穷大/100%）。
* **全局级别**：始终使用 `Sum(Errors) / Sum(N_ref)` 来计算整个测试集的 WER，而**不是**对每句话的 WER 求平均。这样可以自动处理空 Ref 的问题（只要总 Ref 不为空）。



### 3. 时间戳漂移 (Timestamp Shift)

* **现象**：DER 异常高，但看 RTTM 可视化图，模型切分点形状是对的，只是整体晚了 0.5秒。
* **原因**：前端特征提取（STFT）的窗移、流式系统的缓冲延迟未被扣除。
* **技巧**：在评测前，尝试对 Hyp 的所有时间戳减去一个固定的 offset（如 200ms），看 DER 是否显著下降，以此校准系统延迟。

### 4. 繁简转换的“不可逆性”

* **现象**：Ref 是“头发”（简），Hyp 是“頭髮”（繁）。
* **陷阱**：简单的 `opencc` 转换可能把“头发”转回繁体时变成“頭發”（如果词典不全）。
* **建议**：**全部转简体**进行评测通常更稳健，因为简体字符集较小，歧义较少。
