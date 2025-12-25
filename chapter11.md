# Chapter 11: 神经 Diarization 与端到端联合：EEND、TS-VAD、SA-ASR

## 11.1 开篇段落：打破“单人说话”的假设

在 Chapter 10 中，我们探讨了基于聚类（Clustering-based）的传统 Diarization 流水线。这种方法的假设底座是：**语音是稀疏的，且任意时刻主要只有一人在说话**。在新闻播报或有序访谈中，这很有效。

然而，人类自然的交流充满了“抢话”、“插嘴”和“附和”。在计算语言学中，这被称为 **重叠语音（Overlapping Speech）**。在自然会议场景中，重叠语音占比通常在 10% 到 20% 之间，而在激烈的辩论或家庭聚会（如 CHiME 数据集）中，这一比例可能高达 40% 以上。

传聚类方法在面对重叠时会发生什么？

1. **漏检（Miss）**：把重叠段判定为单人（通常是声音大的那个），丢掉弱势说话人。
2. **混淆（Confusion）**：提取的 Embedding 是两人声纹的平均值，导致聚类中心偏移，甚至产生新的“幽灵说话人”。

**神经说话人日志（Neural Diarization）** 应运而生。它不再依赖几何聚类，而是直接训练深度神经网络来解决**多标签分类问题（Multi-label Classification）**。本章将详细拆解 EEND（End-to-End Neural Diarization）、TS-VAD 等核心架构，并展示如何通过 SA-ASR 将“识别”与“区分”合二为一。

**本章学习目标**：

1. **掌握 EEND 架构**：理解多置换不变训练（PIT）如何解决标签分配难题。
2. **精通数据仿真**：学会如何从零构建包含重叠、混响、噪声的高质量训练数据（这是神经 Diarization 的生命线）。
3. **理解 TS-VAD**：学习如何利用先验声纹进行二次精细化分割。
4. **探索联合建模**：理解 t-SOT 序列化方案及其对 MLLM 输入格式的启示。

---

## 11.2 EEND：端到端神经 Diarization 详解

### 11.2.1 范式转换：从“归类”到“开关”

EEND 的核心思想极其简单但暴力：如果网络足够强，为什么不让它为每个人单独输出一条“活性曲线”？

* **输入**：声学特征序列 （ 为帧数， 为特征维数）。
* **输出**：激活概率序列 （ 为最大说话人数，通常设为 2 到 4）。

在时刻 ，输出向量 。

* 如果 ，表示通道 1 对应的说话人在说话。
* 如果  **且** ，表示通道 1 和通道 2 **同时**在说话（重叠）。
* 如果所有 ，表示静音。

这种**多二分类（Multi-binary Classification）**机制天生就能处理重叠，因为不同通道是独立的。

### 11.2.2 核心痛点与解决方案：PIT (Permutation Invariant Training)

训练 EEND 最大的障碍是 **“标签模糊性”（Label Ambiguity）**
假设 Ground Truth (GT) 标注显示：Alice 在 0-5s 说话，Bob 在 4-10s 说话。
你的模型有两个输出通道：Channel A 和 Channel B。
**问题**：你应该强迫 Channel A 学 Alice，还是 Channel A 学 Bob？

如果在 Dataset 中随机指定（例如：谁先说话谁占 Channel A），模型会难以收敛，因为模型无法根据当前的声学特征判断“我是第几个开始说话的”。

**PIT（置换不变训练）** 在 Loss 计算层面解决了这个问题。它的逻辑是：**不管模型怎么输出，只要能找到一种对应关系，让 Loss 最小，我们就按那个关系更新梯度。**

数学定义如下：
设  为参考标签（Reference）， 为模型预测（Hypothesis）。 为说话人数， 为  到  的全排列（Permutations）。

**算法步骤**：

1. 模型输出  个通道的预测 。
2. 获取  个通道的真实标签 。
3. 生成  种可能的排列组合（例如  时，有  和  两种）。
4. 分别计算这  种组合下的二元交叉熵（BCE）损失。
5. 取损失**最小**的那一组组合。
6. 仅根据该最小 Loss 进行反向传播（Backpropagation）。

**ASCII 图解 PIT 逻辑**：

```text
Ground Truth:  [Alice: 11100]  [Bob: 00111]

Model Out Ch1: [0.9, 0.9, 0.9, 0.1, 0.1]  (Looks like Alice)
Model Out Ch2: [0.1, 0.1, 0.8, 0.9, 0.9]  (Looks like Bob)

Permutation 1 (Ch1=Alice, Ch2=Bob):
  Loss(Ch1, Alice) is Low + Loss(Ch2, Bob) is Low = Total Loss VERY LOW ✅

Permutation 2 (Ch1=Bob, Ch2=Alice):
  Loss(Ch1, Bob) is High + Loss(Ch2, Alice) is High = Total Loss HUGE ❌

Result: System picks Permutation 1 to update weights.

```

### 11.2.3 模型架构演进：BLSTM  Transformer  Conformer

1. **BLSTM 时代**：最早的 EEND 使用堆叠的 Bi-LSTM。优点是能捕捉长时序，缺点是无法并行训练，且长序列遗忘问题导致难以处理长会议。
2. **Self-Attention (Transformer) 时代**：利用 Self-Attention 捕捉全局说话人特征。通过 Masked Attention 甚至可以做流 EEND。
3. **Conformer / EEND-Global**：目前的主流。Conformer 结合了 CNN 的局部特征提取能力（对短语调边界敏感）和 Transformer 的全局能力（对同一说话人的长时一致性敏感）。

### 11.2.4 处理人数不确定性：EEND-EDA

基础 EEND 必须预设 （例如 ）。如果会议有 5 个人怎么办？
**EEND-EDA (Encoder-Decoder Attractor)** 引入了序列生成思想：

1. **Encoder**：基于 Conformer 提取声学特征序列。
2. **Attractor Decoder**：这是一个 LSTM，它不输出文本，而是根据特征概况，逐个输出 **“吸引子向量”（Attractor Vector）**。
* 输出 ：代表第一个发现的说话人的声纹中心。
* 输出 ：代表第二个发现的说话人的声纹中心。
* ... 直到输出 `<stop>` 标记。


3. **Dot Product**：将每个时间步的声学特征与所有  做点积（Dot Product）并 Sigmoid，得到该说话人在该时刻的活性。

> **Rule-of-Thumb**
> 尽管 EDA 理论优美，但在工界，**固定通道数的 EEND（如 S=4）** 往往更稳健。对于超过 4 人的场景，通常采用“滑窗处理 + 聚类拼接”的混合策略。因为在一个短时间窗口（如 30 秒）内，同时出现的活跃说话人极少超过 4 人。

---

## 11.3 训练数据的“炼金术”：数据仿真（Simulation）

神经 Diarization 也是“数据饥渴”的。然而，拥有精确到 10ms 级重叠标注的真实数据极其稀缺（即便有，也往往标注不准）。
因此，**90% 以上的 EEND 训练依赖于人工合成数据**。

### 11.3.1 仿真配方（Simulation Recipe）

你需要构建一个 Pipeline，能够源源不断地生成“假”会议数据。

**原材料**：

* **语音源**：LibriSpeech, AISHELL-3, CommonVoice (清洗过的单人片段)。
* **噪声源**：MUSAN (Music, Speech, Noise), Audioset。
* **房间冲击响应 (RIR)**：Simulated RIRs (image method) 或 Real RIRs (AIR dataset)。

**生成步骤**：

1. **采样说话人**：随机选  个说话人（例如 2-4 人）。
2. **采样对话模式**：
* 生成  个轮次（Turns）。
* 每个轮次随机采样一个说话人的语音片段。
* **关键步骤：重叠注入**。不要简单拼接，而是设定一个 `overlap_ratio`（如 ）。当前说话人还没结束，下一个人就开始（Time shift）。


3. **声学增强**：
* **混响（Reverb）**：对每个说话人的音轨分别卷积**相同**房间但**不同**位置的 RIR。
* **加噪（Noise）**：叠加背景噪声（SNR 10-20dB）。


4. **标签生成**：根据原始语音的长度和 Shift 偏移量，生成对应的 RTTM 标签（0/1 矩阵）。

### 11.3.2 域适应（Domain Adaptation）

**陷阱**：仅用 LibriSpeech 仿真训练的模型，在真实电话或会议中效果极差（DER 可能从 5% 飙升到 30%）。
**原因**：真实录音的重叠模式、停顿节奏、麦克风非线性失真，仿真很难完全模拟。
**对策**：

1. **Real Data Finetuning**：必须保留少量真标注数据（如 CallHome, AISHELL-4, AMI）进行微调。哪怕只有 5-10 小时，也能大幅纠正模型的“合成味”。
2. **数据混合**：训练集 = 80% 仿真 + 20% 真实。

---

## 11.4 TS-VAD：基于声纹的二次精修

如果 EEND 是“端到端黑盒”，TS-VAD 就是“模块化精修”。它常用于 **Iterative Diarization**。

**场景**：你已经用聚类方法得到了初步结果，知道本次会议大概有 4 个人（A, B, C, D），并提取了他们各自的声纹中心（Enrollment Embeddings）。但是聚类无法处理重叠，边界也不准。

**TS-VAD 工作流**：
它将问题转化为  个并行的 **个人 VAD (Personal VAD)** 任务。

* **Query**：说话人 A 的声纹向量 。
* **Context**：当前的音频帧序列特征 。
* **Model**：将  拼接到  的每一帧上（或者通过 Cross-Attention 融合），输入 Bi-LSTM/CNN。
* **Output**：输出序列仅表示“说话人 A 在这一帧是否说话”。

对 B, C, D 重复此过（或并行批处理）。这就把复杂的 Diarization 拆解成了简单的“目标说话人检测”。TS-VAD 在 CHiME-6 等高难度竞赛中曾是冠军方案的核心组件。

---

## 11.5 联合建模：SA-ASR (Speaker-Attributed ASR)

终极目标是：不要单独的 RTTM 时间戳，我要直接输出“A说了什么，B说了什么”。

### 11.5.1 序列化输出 (Serialized Output Training, SOT)

SA-ASR 巧妙地修改了 ASR 的 Tokenizer，引入了 `<sc>` (Speaker Change) 或 `<spk_id>` 标记。

**t-SOT (token-level SOT)** 是目前的 SOTA 方法。它处理重叠语音的方式是：**按“先来后到”原则将重叠文本序列化**。

**示例**：

* **音频**：
* Speaker A (0s - 3s): "Hello world"
* Speaker B (1s - 4s): "Hi there"


* **t-SOT 目标序列**：
`<spk_A> Hello <spk_B> Hi <spk_A> world <spk_B> there`

这种交错输出对模型要求极高，因为它要求 Attention 机制必须能够同时追踪两个人的语义流，并在不同说话人之间快速换。

### 11.5.2 MLLM 的连接点

SA-ASR 的输出格式实际上就是 **Prompt Engineering** 的理想输入。
在 MLLM 时代，我们可以分两步走：

1. **Frontend**：用一个轻量级的 SA-ASR 或 EEND 模型生成结构化日志（Transcript + Speaker Labels）。
2. **Backend**：将这个结构化文本喂给 LLM，提示：“请整理上述会议记录，区分说话人 A 和 B 的观点”。

相比于直接让 MLLM 听音频（端到端 Audio-LLM），这种方案目前在控制幻觉和处理长音频方面更具可落地性。

---

## 11.6 本章小结

1. **范式革命**：Diarization 已从无监督聚类转向监督学习（EEND），核心驱动力是解决重叠语音（Overlap）问题。
2. **PIT 是灵魂**：没有 Permutation Invariant Training，模型无法学习区分无序的说话人输出通道。
3. **仿真即生命**：高质量的 EEND 极其依赖包含真实声学特性（RIR）和自然重叠分布的仿真数据。工业界往往花费 70% 的时间调优仿真 Pipeline 上。
4. **TS-VAD**：适合作为聚类后的精修模块，利用先验声纹进行“已知人找语音”。
5. **SA-ASR**：通过 t-SOT 等序列化手段，将 Diarization 隐式地融入 ASR 解码过程，是未来的统一方向。

---

## 11.7 练习题

> **提示**：每道题都尽量先思考，再查看折叠的答案。

### 基础题

1. **核心概念**：简述为什么在 EEND 中使用 Sigmoid 激活函数而不是 Softmax？
> **Hint**: 思考“互斥”与“共存”的区别。


<details>
<summary>查看答案</summary>
Softmax 强制所有输出通道的概率之和为 1，这意味着它认为同一时刻只能有一个类（或者概率分布），这本质上是互斥的。
Sigmoid 对每个输出通道独立计算概率（0到1），互不影响。这允许模型输出如 `[0.9, 0.8]`，即表示两个通道同时高概率激活（重叠语音）。
</details>
2. **PIT 计算**：如果模型设定 （最大3人），计算 PIT Loss 时需要遍历少种排列组合？如果  增加到 10，会发生什么问题？
> **Hint**: 排列组合公式 。


<details>
<summary>查看答案</summary>
 时，排列数为  种。
如果 ，排列数为  种。
**问题**：计算量爆炸。每次前向传播都要计算 300 多万次 Loss 来找最小值，这是不可接受的。因此 EEND 通常限制 。对于更多人数，通常使用贪婪算法近似 PIT 或改用 EEND-EDA / 聚类方法。
</details>
3. **数据仿真**：在生成训练数据时，如果我不小心把“信噪比（SNR）”设置得太高（例如全部 > 30dB，非常干净），模型在真实会议中可能会出现什么行为？
> **Hint**: 域不匹配（Domain Mismatch）。


<details>
<summary>查看答案</summary>
模型会“过拟合”到干净语音的特征。在真实会议中，背景噪声（空调声、电流声）可能会被模型误判为某个说话人的低语，或者模型因为无法处理真实环境中的混响拖尾，导致 VAD 边界切分极不确（通常会把拖尾切掉，导致丢字）。
</details>

### 挑战题

4. **架构思考**：EEND-EDA 解决了人数不确定问题，但它是一个自回归（Autoregressive）过程。这对“流式（Streaming）”推断有什么影响？
> **Hint**: 自回归需要等 Encoder 处理完，还是可以边听边生成？


<details>
<summary>查看答案</summary>
EDA 的 Attractor 生成通常依赖于输入序列的全局信息（Global Average Pooling 或类似机制）来概括全场有哪些人。这使得它天然是**离线（Offline）**的。
要做流式 EEND，通常不使用 EDA，而是使用固定通道 EEND，并结合 **Buffered Streaming** 策略（每隔几秒输出一次），或者通过在线聚类算法动态维护一个 Active Speaker Pool。
</details>
5. **SA-ASR 辨析**：t-SOT 虽然好，但如果重叠极其严重（例如 4 个人同时吵架），这种序列化方案会遇到什么瓶颈？
> **Hint**: Transformer 的 Attention 负荷与 Token 序列长度。


<details>
<summary>查看答案</summary>
1. **序列过长**：4人同时说话，单位时间内的 Token 数量翻了 4 倍，解码延迟大幅增加。
2. **歧义性**：t-SOT 依赖时间顺序交错 Token。当 4 人语速极快且混叠时，模型很难决定 Token 的确切插入顺序（谁先谁后可能只有几毫秒差别），这种“顺序抖动”会干扰语言模型（LM）的上下文预测能力，导致 WER 升高。
</details>



---

## 11.8 常见陷阱与错误 (Gotchas)

### 1. 沉默的通道陷阱 (The Silent Channel Trap)

* **现象**：训练 2-speaker EEND 模型时，数据集中包含大量单人说话片段。训练后发现模型倾向于永远只输出 Channel 1，Channel 2 即使在重叠时也保持静默。
* **原因**：训练数据的通道分配没有随机化，或者 PIT 实现有 Bug。如果数据总是把说话人放在 Channel 1，模型会学会“偷懒”。
* **Fix**：确保 PIT 实现正确，且在数据加载器（DataLoader）中随机 shuffle 说话人 ID 的顺序。

### 2. 泛化性灾难

* **现象**：在仿真集上 DER < 5%，在真实会议上 DER > 25%。
* **原因**：仿真 RIR 与真实声学环境差异太大。
* **Fix**：**不要只用 Image Method 生成的 RIR**。尽量下载开源的真实 RIR 数据集（如 AIR, BUT ReverbDB）混入训练。此外，加入 **SpecAugment**（时频掩蔽）可以强迫模型不依赖特定的频段特征，提升鲁棒性。

### 3. 指标作弊 (Metric Hacking)

* **现象**：Collar（容差窗口）设置过大。
* **Gotcha**：Diarization 标准评测通常允许 0.25s 的 Collar（不计分区域）。有些论文为了刷榜，在训练时过度优化 VAD 边缘，或者在评测时私自扩大 Collar。
* **建议**：在对比模型时，务必检查评测脚本中的 `collar` 设置是否一致（标准通常是 0.0 或 0.25 秒）。对于 SA-ASR，主要看 cpWER (Concatenated Minimum Permutation WER)。
