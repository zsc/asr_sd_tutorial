# Chapter 10: Speaker Diarization 经典流水线：SAD + Embedding + Clustering + Resegmentation

## 1. 开篇段落：解构“谁在说话”

**Speaker Diarization（说话人日志/区分）** 是语音处理中回答“**Who spoke when?**（谁在什么时间说了话？）”的关键技术。

在 ASR 解决了内容的转写后，Diarization 赋予了这些文字身份属性。在会议记录、医疗问诊、法律取证以及现代的“AI 助手”场景中，区分不同说话人是理解对话逻辑的前提。

尽管 EEND（端到端神经 Diarization）等新架构正在学术界兴起，但**模块化级联系统（Modular Cascade System）** 依然是工业界最稳健、最可控、且被广泛部署的方案（如 Kaldi, SpeechBrain, Pyannote 的基础版本）。这种方案将复杂任务解耦为四个可独立优化的子模块。

本章将带你像“搭积木”一样构建一个高精度的 Diarization 系统，并深入探讨每个模块背后的工程陷阱。

---

## 2. 文字论述

### 10.1 任务全景与流水线架构

输入是一段长音频 X，输出是若干个元组 (speaker_id, start, end)。

经典的流水线是一个“漏斗模型”，信息逐步压缩并结构化：

```ascii
[Raw Audio Stream]
       |
       v
+-----------------------------+
| 1. SAD / VAD                |  <-- 过滤非人声
| (Signal Processing / DNN)   |      目标：只保留有效语音，避免对噪声聚类
+-----------------------------+
       | Segments (e.g. 0.0-5.2s)
       v
+-----------------------------+
| 2. Segmentation & Embedding |  <-- 连续特征离散化
| (Sliding Window -> x-vector)|      目标：将变长音频切片映射为固定维度的声纹向量
+-----------------------------+
       | Sequence of Vectors
       v
+-----------------------------+
| 3. Clustering               |  <-- 无监督分组
| (AHC / Spectral / k-means)  |      目标：确定有多少人(k)，并赋予粗略标签
+-----------------------------+
       | Rough Labels
       v
+-----------------------------+
| 4. Resegmentation           |  <-- 边界微调
| (VBx / HMM)                 |      目标：修正聚类错误，恢复精确的时间边界
+-----------------------------+
       |
       v
[Final RTTM: Spk1 0.0-5.2s]

```

### 10.2 SAD / VAD：第一道防线

**Voice Activity Detection (VAD)** 或 **Speech Activity Detection (SAD)** 决定了系统的下限。

* **误杀（Miss）**：说话人声音被切掉，后续所有模块都无法挽回。
* **虚警（False Alarm）**：呼吸声、键盘声、关门声被放入流水线。这些噪声产生的 Embedding 会在特征空间中形成不可预测的“噪声簇”，或被随机分配给某个说话人，导致严重误判。

#### 10.2.1 现代 VAD 的做法

早期的能量阈值法已不适用复杂环境。现代主流方案是：

1. **基于统计模型**：WebRTC VAD（轻量级，基于 GMM）。
2. **基于神经网络**：
* **输入**：Mel-spectrogram 或 MFCC。
* **模型**：小型的 CRDNN 或 LSTM，二分类输出（Speech vs Non-Speech）。
* **后处理**：设置 `min_speech_duration`（如 0.25s）和 `min_silence_duration`（如 0.5s）进行平滑，避免产生过于破碎的片段。



> **Rule-of-Thumb 10.1**：在 Diarization 任务中，VAD 宁可**稍微“激进”一点保留更多内容（高 Recall）**，也不要切掉微弱的尾音。因为后续的聚类算法通常对少量噪声有一定的鲁棒性，但前面丢掉的语音永远找不回来。

---

### 10.3 Speaker Embedding：从声音到向量

这是流水线的核心引擎。我们需要一个函数 f(x)，使得同一人的向量夹角小，不同人的夹角大。

#### 10.3.1 模型架构演进

* **TDNN (x-vector)**：Kaldi 时代的经典。利用一维卷积（Time-Delay NN）捕捉时序上下文。
* *关键层*：**Statistical Pooling**。它计算倒数第二层特征在时间轴上的**均值（Mean）和标准差（Std）**。这一步将变长序列（T×D）“拍扁”成了定长向量（2D）。


* **ECAPA-TDNN**：目前的主流（SpeechBrain/Pyannote 默认）。引入了 SE-Block（通道注意力）和多尺度特征聚合（Multi-scale Feature Aggregation），在短语音上表现更稳。
* **ResNet34 / Conformer**：在超大规模数据上训练时，这些大模型往往能提取出更细腻的特征。

#### 10.3.2 提取策略：滑动窗口

我们不能只对整段话提一个特征。通常采用**滑动窗口（Sliding Window）**：

* **Window Size**：1.5s ~ 2.0s（保证包含足够音素信息）。
* **Step Size**：0.5s ~ 0.75s（产生重叠，保证时间分辨率）。
* *结果*：一段 10 秒的音频，可能会产生约 15-20 个 x-vectors。

#### 10.3.3 Scoring Backend：PLDA

拿到 Embedding 后，直接计算 Cosine Distance 往往不够好，因为 Embedding 包含了**说话人信息** + **信道信息（Channel）**。
**PLDA (Probabilistic Linear Discriminant Analysis)** 是解决此问题的“核武器”：

* 它假设 x = μ + V y + U z + ε。
* μ: 全局均值。
* V y: 说话人空间（我们想要的）。
* U z: 信道/干扰空间（我们想去除的）。


* PLDA 计算的是 **Log-Likelihood Ratio (LLR)**：log p(x1, x2 | same) - log p(x1, x2 | diff)。

---

### 10.4 聚类 (Clustering)：在未知中寻找结构

聚类是将提取出的一堆无标签向量（Vectors）归堆的过程。最大的难点是 **Unknown Number of Speakers (K)**。

| 特性 | AHC (层次聚类) | Spectral Clustering (谱聚类) |
| --- | --- | --- |
| **原理** | 贪婪算法，基于距离矩阵逐步合并 | 图论，基于亲和矩阵的特征分解 |
| **计算量** | O(N^2)~O(N^3) | O(N^3) (主要在特征分解) |
| **阈值敏感度** | **极高** (停止合并的阈值很难调) | **中等** (可通过 Eigengap 自动估算 k) |
| **全局观** | 弱 (只看局部最近) | 强 (看全局图结构) |
| **适用场景** | 简单场景、流式处理 | 复杂会议、噪声环境、离线处理 |

#### 10.4.1 如何自动决定有多少人？

在谱聚类中，我们计算拉普拉斯矩阵的特征值（Eigenvalues）。理论上，如果数据有 K 个完美的簇，就会有 K 个接近 0 的特征值，第 K+1 个特征值会突然变大。
**Maximum Eigengap** 准则：寻找 λ_{k+1} - λ_k 最大的位置，对应的 k 即为人数。

> **Rule-of-Thumb 10.2**：对于短音频（< 1分钟），AHC 往往比谱聚类好，因为数据量太少时“图”结构构建不起来。对于长会议（> 15分钟），谱聚类 + Eigengap 是首选。

---

### 10.5 Resegmentation：VBx 的精修魔法

聚类得到的结果是粗糙的（因为基于 1.5s 的滑动窗口，分辨率低）。
**VBx (Variational Bayes HMM)** 是目前几乎所有 SOTA 系统（如 DIHARD 冠军方案）的标配后处理。

* **原理**：它构建一个 HMM 模型。
* **状态**：每个说话人是一个状态。
* **发射概率**：当前帧的声学特征属于该说话人分布（GMM）的概率。
* **初化**：利用聚类结果初始化 GMM 参数。
* **迭代**：使用变分贝叶斯推断，重新分配每一帧的归属，直到收敛。


* **作用**：它能把时间边界精确到帧级别（Frame-level，如 10ms），并能修正聚类中个别的错误分配。

---

### 10.6 与 ASR 的组合策略

有了 RTTM（时间戳+ID），如何结合 ASR？

1. **Pipeline A: Diarization → ASR (Cut & Decode)**
* 根据时间戳把音频切成小段，分别送入 ASR。
* *缺点*：切断了句子的上下文，导致 ASR 在边界处经常识别错（例如切断了“I am”和“happy”）。


2. **Pipeline B: ASR → Diarization (Alignment)**
* 先对整段长音频做 ASR，拿到**词级别时间戳 (Word-level timestamps)**。
* 同时做 Diarization 拿到 RTTM。
* **对齐算法**：统计每个单词的时间范围内，哪个 Speaker ID 出现最多，就将该单词赋给谁。
* *优点*：ASR 能够利用完整上下文，识别率最高。这是目前推荐的方案。



---

### 10.7 对 MLLM 的借鉴意义

Diarization 是连接“无结构音频”与“强语义理解”的桥梁。

1. **Contextual Steering (上下文引导)**：
* MLLM 无法直接听懂“第二个人说的那句话”。但如果你提供 `Speaker 02: [Text]`，你就可以 Prompt MLLM：“请总结 Speaker 02 的观点”。


2. **Speaker-Aware RAG**：
* 我们可以建立一个**声纹数据库**（Key: Embedding, Value: "张三"）。
* 在推理时，经典流水线提取 Embedding → 检索向量库 → 替换 Speaker ID。
* 最终喂给 MLLM 的是：`张三: 咱们明天的会议改期吧。` MLLM 就能处理“张三想要改期”的意图，而不是“Speaker 1”。


3. **Role Prompting**：
* Diarization 结果可以帮助我们推断角色。通过分析说话时长和交互模式，可以先判断出谁是“客服”谁是“用户”，然后在 MLLM Prompt 中加入角色设定，提升回答的准确性。



---

## 3. 本章小结

* **VAD** 决定召回率，**Embedding** 决定区分，**Clustering** 决定人数估计，**Resegmentation** 决定边界精度。
* **x-vector / ECAPA-TDNN** 配合 **PLDA** 是特征提取的黄金组合。
* **谱聚类 (Spectral Clustering)** 在人数未知的复杂场景下优于 AHC。
* **VBx** 是将粗糙聚类结果转化为帧级精确结果的关键步骤。
* **ASR first, Diarization second** 的策略通常能获得更好的 WER 和可读性。

---

## 4. 练习题

### 基础题

<details>
<summary><strong>1. 为什么 x-vector 模型在训练时使用短片段（如 2-4秒），但在测试时可以处理任意长度的音频？</strong></summary>

**答案：**
这是由于 **Statistical Pooling** 层的存在。
无论输入音频有多少帧（T），池化层都会在时间维度上计算均值和方差，从而把 T×D 的特征矩阵变为 2D 的固定维度向量。全连接层（DNN）只处理这个固定维度的向量。因此，模型架构对输入长度是不敏感的（尽管过短的音频会导致统计量不准）。

</details>

<details>
<summary><strong>2. 在使用 PLDA 进行打分时，为什么需要先对 Embedding 做 Length Normalization？</strong></summary>

**答案：**
原始的 x-vector 模长通常与音频的质量、时长或响度有关，而不仅是说话人身份。
PLDA 本质上是基于高斯分布的假设。如果不做归一化，模长大的向量会在高斯分布中占据主导地位，或者偏离高斯假设。将向量投影到单位超球面上（Length Norm），可以消除这些非身份因素的干扰，让 PLDA 专注于角度（方向）差异。

</details>

<details>
<summary><strong>3. Diarization Error Rate (DER) 是如何计算的？如果我把所有时间戳都平移了 0.2秒，DER 会受影响吗？</strong></summary>

**答案：**

是的，**DER 会受严重影响**。通常评测时允许一个 **Collar**（如 0.25秒）的误差宽容度，即在参考边界 ±0.25s 内的误差不计入。但如果平移量超过 Collar，系统会被判为严重的 Miss（开头没对上）和 False Alarm（结尾多出来了），导致 DER 飙升。

</details>

<details>
<summary><strong>4. 什么是 "Overlap" 问题？经典流水线为什么难处理它？</strong></summary>

**答案：**
Overlap 指两人或多人同时说话。
经典聚类（K-means/Spectral/AHC）是 **Hard Clustering**，即它强制把每一个时间片或 Embedding 分配给**唯一**的一个簇 ID。因此，它天然无法输出“Spk1 和 Spk2 同时存在”的结果。解决办法通常需要专门的 Overlap Detection 模型或使用 EEND 端到端方案。

</details>

### 挑战题

<details>
<summary><strong>5. 假设你在调试一个会议 Diarization 系统，发现系统总是倾向于把一个人拆成两个 ID（Over-clustering）。你应该调整哪些参数？</strong></summary>

**Hint**：考虑聚类阈值和 PLDA 行为。

**答案：**

1. **提高聚类阈值**：如果是 AHC，提高停止合并的阈值（让更多像的簇合并）；如果是谱聚类，调整 Eigengap 判定策略，倾向于选择更小的 。
2. **检查 VAD 切分**：如果一个人中间停顿被切开太久，且录音环境发生微弱变化（如转身），可能会导致 embedding 漂移。尝试缩短 `min_silence_duration`。
3. **PLDA 域适配**：如果训练 PLDA 的数据是录音棚数据，而测试数据是混响严重的会议室，PLDA 会认为“带混响的张三”和“不带混响的张三”是两个人。需要做 **Domain Adaptation**（如无监督的 PLDA 适应）。

</details>

<details>
<summary><strong>6. 为什么 VBx Resegmentation 通常能降低 DER？它利用了哪些前面步骤没利用的信息？</strong></summary>

**Hint**：考虑时间上的连续性。

**答案：**
聚类通常是把所有 Embedding 作为一个无序集合（Bag of Segments）来处理的，忽略了**时间顺序**。
VBx 利用了 HMM 的 **Transition Probability（转移概率）**。它隐含了一个先验知识：**“说话人倾向于在一段时间内持续说话”**。这抑制了聚类结果中出现的快速跳变（Spk A -> B -> A -> B）。此外，VBx 是在帧级别（Frame-level）操作，比基于 1.5s 窗口的聚类分辨率高得多。

</details>

<details>
<summary><strong>7. (场景设计) 你要为法庭庭审记录设计一个系统，要求极高的说话人准确度，允许人工辅助。你会如何设计这个 "Human-in-the-loop" 流程？</strong></summary>

**Hint**：利用可视化与聚类约束。

**答案：**

1. **First Pass**：运行全自动流水线（Spectral Clustering），为了避免漏人，参数调向 Over-clustering（宁多勿缺）。
2. **可视化交互**：将 Embedding 降维（t-SNE/UMAP）展示在 2D 平面上。同一簇的点标相同颜色。
3. **人工修正（约束注入）**：
* **Must-link**：法记员点选两个点，标记“这是同一个人”。
* **Cannot-link**：法记员标记“这是两个人”。


4. **Constrained Clustering**：重新运行聚类算法，但强制满足上述约束（Constrained K-Means / Spectral）。
5. **VBx Finetuning**：最后运行 VBx 精修边界。
这种 Semi-supervised 模式比纯人工标注快得多，又比纯自动准确得多。

</details>

---

## 5. 常见陷阱与错误 (Gotchas)

### 5.1 采样率陷阱 (8k vs 16k)

* **陷阱**：用 16kHz 的数据训练了 x-vector 模型，然后在 8kHz 的电话录音上做推理（或者反之）。
* **后果**：模型性能几乎随机。因为频谱特征完全变了。
* **解决**：确保推理时上采样/下采样到模型训练时的采样率。或者训练一个混合带宽模型。

### 5.2 归一化的顺序

* **陷阱**：在 Embedding 提取后，先做 PCA 降维，再做 Length Normalization。
* **修正**：**通常建议先 Length Norm，再 LDA/PCA，再 Length Norm**。尤其是最后进 PLDA 之前，必须保证向量在超球面上。这一步顺序搞反，EER（等错误率）可能差 20%。

### 5.3 “幽灵”说话人 (Ghost Speakers)

* **现象**：只有 2 个人在说话，系统却输出了 5 个 ID。其中 Spk3, Spk4, Spk5 只出现了几秒钟。
* **原因**：VAD 没切干净的噪声（拍手、笑声、咳嗽）形成了独立的簇。
* **调试**：在输出 RTTM 后，增加一个后处理脚本：**“丢弃总时长小于 X 秒的说话人”**。如果 Spk3 总共只出现了 2 秒，极大概率是误报。

### 5.4 混响（Reverb）灾难

* **现象**：在大会议室录音，距离麦克风远的人识别极差，且容易被分成多个 ID。
* **原因**：混响拖尾破坏了短时特征。
* **解决**：
* **Data Augmentation**：训练 x-vector 时必须加入 **RIR (Room Impulse Response)** 混响增强。
* **Dermix**：使用去混响（De-reverberation）的前端处理（如 WPE 算法）在 VAD 之前清洗音频。
