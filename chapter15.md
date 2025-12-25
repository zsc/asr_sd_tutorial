# Chapter 15: 开源数据集大全：ASR / 多语种 / 会议 / 噪声 / Diarization

> **本章摘要**：数据是 AI 的燃料，但并非所有燃料都是等价的。本章汇集了目前（截至 2024/2025）主流的 ASR 与 Speaker Diarization 开源数据集。
> 我们不仅提供下载清单，更侧重于**“鉴宝”**——分析每个数据集的**声学特性、标注陷阱、适用场景**，并提供工业界常的**数据混合配方 (Data Recipes)**。此外，针对 MLLM 时代，本章特别梳理了适合**指令微调 (Instruction Tuning)** 和**上下文增强**的数据资源。
> **学习目标**：
> 1. 建立“数据分级”意识：区分验证集、预训练集和微调集。
> 2. 掌握中、英、日、粤及多语种的主流数据集特性。
> 3. 学会构建针对特定场景（如远场会议、低资源方言）的数据组合。
> 4. 了解如何利用开源数据构建 MLLM 的指令集。
> 
> 

---

## 15.1 数据集的“六维能力”评估

在下载 TB 级的数据之前，不要只看“小时数”。作为算法工程师，你需要用以下六个维度来评估一个数据集的价值：

```text
       真实度 (Spontaneity)
          ^
          | (会议/电话/采访 -> 极难)
          |
(噪杂/远场) +--------------------> 规模 (Scale)
环境复杂度 |                      (10k h+ -> 预训练基石)
          |
          |
      标注丰富度 (Label Richness)
      (文本 / 时间戳 / 说话人ID / 情感 / 语种)

```

1. **真实度 (Spontaneity)**：
* **朗读 (Read)**：AISHELL-1, LibriSpeech。语法完美，发音清晰。**作用**：验证模型收敛，学术刷榜。
* **自发 (Spontaneous)**：WenetSpeech, GigaSpeech。含口吃、重复、抢话、语法错误。**作用**：产品落地的必修课。


2. **环境 (Acoustic Environment)**：
* **近讲 (Near-field)**：信噪比高。
* **远场 (Far-field)**：含混响 (Reverb)、能量衰减。
* **电话 (Telephony)**：8kHz 采样，窄带，有编解码失真。


3. **规模 (Scale)**：
* **< 100h**：仅用于 Few-shot 微调或评测。
* **1k h ~ 5k h**：训练一个像样的专用模型。
* **10k h ~ 100k h**：训练基础模型 (Foundation Model) 的门槛。


4. **标注精度 (Label Quality)**：
* **Golden**: 人工双盲校验（错误率 < 1%）。
* **Weak**: 机器生成或爬虫抓取（错误率 5% - 20%）。**切记：大规模预训练可以用 Weak，但微调和评测必须用 Golden。**


5. **元数据 (Metadata)**：
* 是否包含：话题 (Topic)、说话人性别/年龄、录音设备信息。这对 MLLM 的 Prompt 构建至关重要。


6. **许可 (License) —— 你的红线**：
* **Commercial Friendly**: CC-BY, Apache 2.0, MIT, CC0.
* **Research Only**: CC-BY-NC (Non-Commercial). **警告**：企业内部研发（即使不直接卖模型）使用 NC 数据也存在法律灰色地带，合规团队通常会禁止。
* **LDC**: 通常需付费购买。



---

## 15.2 中文 ASR 数据集 (Mandarin)

中文数据的核心挑战在于**口音（Accent）**、**领域（Domain）和繁简混杂**。

### 15.2.1 朗读与基础类 (Read Speech)

| 数据集 | 时长 | 质量 | 许可 | 核心点评 |
| --- | --- | --- | --- | --- |
| **AISHELL-1** | 178h | ★★★★★ | Apache 2.0 | **中文 ASR 的 "Hello World"**。录音棚高保真，人工精标。如果你连这个都跑不通，检查代码。**缺点**：太干净，模型易过拟合。 |
| **AISHELL-3** | 85h | ★★★★ | Apache 2.0 | **多说话人 TTS 数据**。虽然主要用于合成，但因其高保真和丰富的说话人特征，常用于辅助 ASR 训练说话人适应性。 |
| **Aidatatang_200zh** | 200h | ★★★★ | CC-BY-NC | 类似 AISHELL-1，但更偏口语化朗读。适合作为基础数据的补充。 |
| **Primewords** | 100h | ★★★★ | CC-BY-NC | 包含丰富的标点符号和数字表达，适合测试 TN (Text Normalization) 流程。 |
| **Thchs-30** | 30h | ★★★ | Apache 2.0 | 清华老牌数据。最有价值的是它附带了**同一句话的“加噪版”**，是研究抗噪算法的绝佳微型实验室。 |

### 15.2.2 互联网与自发类 (Spontaneous / Web)

| 数据集 | 时长 | 质量 | 许可 | 核心点评 |
| --- | --- | --- | --- | --- |
| **WenetSpeech** | 10k h+ | ★★★ (Weak) | Apache 2.0 | **工业界基石**。来自 YouTube/Podcast。覆盖综艺、访谈、讲座。**注意**：标注是“弱监督”的，存在错别字和时间戳漂移。建议使用其提供的置信度分数进行筛选。 |
| **MagicData (开源集)** | 755h | ★★★★★ | CC-BY-NC* | 质量极高。包含朗读和**Scripted Conversation**。比纯朗读更自然，比纯自发更规范。*注意检查具体子集协议。* |
| **ST-CMDS** | 100h | ★★★★ | Free | 命令词数据（“打开空调”、“播放音乐”）。适合做**端侧低功耗关键词唤醒 (KWS)**。 |

> **Rule of Thumb (中文)**:
> * **做演示**: AISHELL-1 足够。
> * **做产品**: WenetSpeech (Strong label subset) + AISHELL-1 + 购买数据/自采数据。
> * **繁体中文**: 既然开源繁体数据稀缺，通常策略是将简体数据通过 OpenCC 转换，辅以 **Common Voice (Zh-HK/TW)** 进行微调。
> 
> 

---

## 15.3 英文 ASR 数据集 (English)

英文生态不仅用于英文识别，更是训练 MLLM 音频编码器（Audio Encoder）的主力。

| 数据集 | 时长 | 类型 | 许可 | 核心点评 |
| --- | --- | --- | --- | --- |
| **LibriSpeech** | 960h | 有声书 | CC-BY | **学术界通货**。几乎所有论文 benchmark 都有它。分为 `clean` 和 `other`。全美音。模型如果在此数据集 WER > 5%，说明结构有问题。 |
| **GigaSpeech** | 10k h | 多领域 | Apache 2.0 | 类似 WenetSpeech。分为 XS/S/M/L/XL。包含 Podcast, YouTube。非常考验对**非正式口语**、**背景音乐**的鲁棒性。 |
| **Common Voice (EN)** | 3k h+ | 全球口音 | CC0 | Mozilla 众包项目。**含金量在于口音**（印度、澳洲、欧洲口音）。做国际化业务必选。 |
| **Ted-Lium 3** | 450h | 演讲 | CC-BY-NC | TED 演讲。语速快，专有名词（科技、生物、政治）密集。适合训练**长句建模**能力。 |
| **SPGISpeech** | 5k h | 财经会议 | Free* | 包含大量**数字、货币、公司名**。如果你的目标是 RAG 金融助手，这是最好的微调数据。 |

---

## 15.4 方言与CJK语系 (Dialects & CJK)

这是多语种模型最容易“翻车”的地方。

### 15.4.1 粤语 (Cantonese / Yue)

* **Common Voice (zh-HK)**: 目前最易获取的粤语源。
* *陷阱*：注意区分朗读（书面语读音）和口语（包含“嘅”、“喺”等粤语字）。训练时建议使用**字级 (Character)** 建模而非拼音。


* **MDCC**: 竞赛数据集，质量高但有时效性。
* **Guangzhou Daily**: 部分高校发布的朗读新闻，风格较老旧。

### 15.4.2 日语 (Japanese)

* **CSJ (Corpus of Spontaneous Japanese)**: **行业标准**，但付费且贵。主要包含学术演讲。
* **ReazonSpeech**: **新星 (Game Changer)**。约 19,000h+，从日本电视节目提取。开源、量大、真实。是目前训练日语 MLLM 的首选。
* **Common Voice (JA)**: 规模尚可，适合补充口音多样性。
* **JSUT**: 10h 女声高保真，主要用于 TTS，可用于 ASR 快速适应（Domain Adaptation）。

### 15.4.3 韩语 (Korean)

* **KsponSpeech**: 1000h 自发对话。韩语 ASR 的主要基准。注意：由 AI Hub Korea 发布，非韩 IP 下载流程极度繁琐（通常需要VPN+实名认证）。
* **Zeroth-Korean**: 约 50h，开源友好，适合做基线测试。

---

## 15.5 会议、Diarization 与 噪声 (The Hard Mode)

如果不做会议和抗噪训练，ASR 系统在现实世界（如咖啡厅、会议室）几乎“不可用”。

### 15.5.1 会议与多说话人 (Meeting & Multi-talker)

此类数据包含**重叠语音 (Overlap)** 和**远场 (Far-field)** 特性。

| 数据集 | 语种 | 场景 | 通道 | 用途 |
| --- | --- | --- | --- | --- |
| **AMI Corpus** | 英文 | 模拟会议 | 阵列+耳麦 | **Diarization 圣经**。包含极详细的标注（Head gesture, movement）。 |
| **AISHELL-4** | 中文 | 真实会议 | 8通道 | 包含重叠率标注。适合研究**波束形成 (Beamforming)** 后端接 ASR。 |
| **AliMeeting** | 中文 | 医疗/通用 | **阵列+耳麦** | **极佳的对比学习材料**。由于它同时提供了同一个会议的“近讲（清晰）”和“远场（混响）”录音，你可以用它来训练 Speech Enhancement 模型或者做 Teacher-Student 蒸馏。 |
| **LibriCSS** | 英文 | 朗读拼接 | 单通道 | 用 LibriSpeech 合成的多通道重叠音频。主要用于评估分离 (Separation) 算法。 |

### 15.5.2 Diarization 专项

* **VoxConverse**: 基于 YouTube 视频，真实环境下的说话人转换。Diarization 比赛常用集。
* **DIHARD 系列**: **地狱难度**。包含餐厅、临床、室外、儿童等极端环境。如果你的模型能过这一关，就能过任何关。

### 15.5.3 噪声与脉冲响应 (Noise & RIR)

用于数据增广（Data Augmentation），**必须下载，不得跳过**。

* **MUSAN**: 包含 Speech (干扰人声), Music (背景音乐), Noise (环境噪)。**工业界标配**。
* **RIR_noises (OpenSLR 28)**: 包含真实的房间脉冲响应（Reverb）。用于通过卷积操作将“录音棚声音”变成“会议室声音”。
* **DNS Challenge**: 微软组织的降噪比赛数据，包含极其丰富的噪声类型（键盘声、关门声、风扇声）。

---

## 15.6 说话人识别 (Speaker Verification / Embedding)

用于训练 x-vector / ECAPA-TDNN / ResNet34 等 Embedding 模型，这是 Diarization 系统中“认人”的关键模块。

* **VoxCeleb 1 & 2**: **绝对霸主**。包含 7000+ 名人，100万+ 语音片段。涵盖多语种、多环境。
* *Gotcha*: 很多视频已经被 YouTube 删除了，下载完整的 dataset 需要找第三方镜像或 torrent。


* **CN-Celeb 1 & 2**: **中文首选**。包含 3000+ 中国名人。弥补了 VoxCeleb 中文覆盖的不足。包含唱歌、朗诵等多种体裁。

---

## 15.7 面向 MLLM 的特种数据 (For Large Models)

在 MLLM 时代，我们需要不仅仅是 (Audio, Text) 对，还需要指令和翻译数据。

* **CoVoST 2**: 基于 Common Voice 的多语种**语音翻译 (ST)** 数据。适合训练 MLLM 的 `Translate this to English` 能力。
* **GigaSpeech (Metadata)**: 利用其丰富的元数据（如 tag: `Science`, `Comedy`），可构造如下指令数据：
* *Input*: Audio
* *Instruction*: "Identify the topic of this speech."
* *Output*: "Science."


* **Prompt-Response 构造**: 利用 AMI 会议数据，构造摘要任务：
* *Input*: Meeting Audio Chunk
* *Instruction*: "Summarize the key decision."
* *Output*: (Annotated Abstract).



---

## 15.8 工业界数据组合“推荐菜单” (Recommended Recipes)

针对不同预算和目标，提供几套经得起考验的“配方”。

### 菜单 A：低成本冷启动 (The "MVP" Starter)

* **目标**：快速跑通全流程，做演示 Demo。
* **ASR**: AISHELL-1 (CN) + LibriSpeech-clean-100 (EN)。
* **Diarization**: AMI (Headset mix) —— 避开远场难点。
* **特点**: 数据干净，无需复杂清洗，单卡 24小时内可训完。

### 菜单 B：生产级中文通用 ASR (The "Production" Chinese)

* **目标**：抗噪、通用、能处理口音。
* **基础 (Base)**: WenetSpeech (L subset) + MagicData。
* **精调 (Fine-tune)**: AISHELL-1 + AISHELL-2 (如有预算) + 业务数据。
* **增广 (Augmentation)**:
* MUSAN (噪声) + RIR (混响) 概率设为 0.5。
* **SpecAugment** 掩码策略加重。
* **变速 (Speed Perturb)**: 0.9x, 1.0x, 1.1x 三倍数据扩充。


* **技巧**: 使用 WenetSpeech 预训练模型作为 Checkpoint 初始化，**严禁从零训练**。

### 菜单 C：智能会议记录员 (The "Meeting" Agent)

* **目标**：处理多说话人、有回声、需区分角色的会议录音。
* **声学模型**:
* AliMeeting (远场) + AISHELL-4。
* **关键**: 混入 30% 的近讲数据 (AISHELL-1) 并**人工添加重混响 (Reverb)**，模拟不同房间大小。


* **Diarization**:
* Embedding: VoxCeleb + CN-Celeb 联合训练。
* SAD (VAD): 使用 AliMeeting 的静音段微调。


* **LLM 后端**: 使用 AMI 的摘要数据微调 LLM，使其学会忽略口语词（"um", "ah"）。

---

## 15.9 本章小结

1. **没有完美的数据集**，只有完美的组合。WenetSpeech 量大但脏，AISHELL 纯净但窄。**Dirty Pre-training + Clean Fine-tuning** 是黄金法则。
2. **噪声不是敌人，是朋友**。如果不加 MUSAN 和 RIR，模型在真实世界就是“温室里的花朵”。
3. **多语种平衡**。训练多语种模型时，要对低资源语种进行过采样 (Oversampling)，否则英文会主导 Embedding 空间。
4. **License 审查**。这是技术负责人的生命线，切勿在商用代码库中混入 NC 数据。

---

## 15.10 练习题

1. **基础题 : 假设你要训练一个用于车载语音助手**的 ASR。你会如何组合 AISHELL-1 和 MUSAN 数据集？具体的增广策略应该侧重什么类型的噪声？
> *Hint: 考虑汽车环境的特殊性（风噪、引擎声、回声）。*


2. **基础题**: 下载了 WenetSpeech 后，发现 JSON 标注文件中每条数据都有一个 `confidence` 字段。你应该如何设定阈值？如果设得太高（如 >0.95）会有什么副作用？
> *Hint: 权衡数据量与数据质量。*


3. **挑战题**: 你只有 10 小时的**四话**数据，但有 10,000 小时的普通话数据。请设计一个训练方案，使得四川话识别率最高。(考虑 Tokenizer 和 预训练策略)
> *Hint: 四川话和普通话共享汉字，但声调和部分词汇不同。Adapter 还是 Full Fine-tuning？*


4. **挑战题 **: 在处理 AMI 会议数据时，你发现重叠语音（Overlap）导致的识别错误很高。利用 LibriSpeech，你如何人工合成一份“重叠语音训练集”来提升模型对 Overlap 的鲁棒性？
> *Hint: 简单的音频相加是否足够？是否需要对齐文本？*


5. **开放题 (MLLM)**: 现有的 ASR 数据集（如 LibriSpeech）只有 `(Audio, Text)` 对。如何利用这些数据训练一个支持“语音检索”（例如：用户说一段话，模型找出这段话在长音频中的位置）的模型？需要构造什么样的指令？

<details>
<summary>点击展开参考答案</summary>

1. **答案**：
* **组合策略**：以 AISHELL-1 为主，必须进行**在线动态混合 (On-the-fly mixing)**。
* **噪声侧重**：MUSAN 中的 `noise` 类别，特别是引擎声、风声、路面噪音。
* **RIR**：车内空间狭小，有独特的短混响，应选择类似 "Small Room" 或 "Car cabin" 的 RIR 冲激响应进行卷积。
* **SpecAugment**：加大频域掩码 (F-mask)，因为车载噪音通常集中在低频。


2. **答案**：
* **策略**：通常在预训练阶段（Epoch 1-50）使用较低阈值（如 > 0.6）以获取最大数据量，覆盖长尾词汇。
* **副作用**：如果阈值设得太高（> 0.95），会过滤掉大量**语速快、口音重或背景有噪**的困难样本。模型会变得“偏科”，只能识别清晰的标准音，泛化能力大幅下降。


3. **答案**：
* **Tokenizer**：使用**字 (Character)** 为单位。因为四川话也是写汉字，与普通话共享 Token 空间。
* **步骤 1**：用 10,000h 普通话训练一个强底座 (Base Model)。
* **步骤 2 (关键)**：保持底座大部分参数冻结（Freeze），只解冻 Encoder 的最后 2-3 层或者插入 **LoRA / Adapter** 层。
* **步骤 3**：用 10h 四川话进行微调。为了防止过拟合，应使用较大的 Dropout 和较小的 Learning Rate。
* **额外技巧**：在普通话数据中混入四川话做联合训练（比例 10:1），效果通常优于两阶段微调。


4. **答案**：
* **合成方法**：随机选取两条 LibriSpeech 音频 A 和 B。
* **操作**：
1. 将 B 以随机信噪比（如 0dB, 5dB）叠加到 A 上。
2. **标签处理**：这是难点。如果是做 ASR，标签通常设为 "A 的文本"（假设 A 是主说话人）或者 "A文本 <sep> B文本"（如果是 SOT 模型）。
3. **对齐**：简单的相加是不够的，最好让两段语音有部分重叠（Partial Overlap）而非全重叠，模拟真实的抢话场景。




5. **答案**：
* **数据构造**：将长音频切片，记录每个切片的 `(Start_Time, End_Time)`。
* **指令构造**：
* Input Audio: (Full long audio or a context window).
* Input Text prompt: "Find the timestamp for the phrase '{Text_Snippet}'".
* Output: "{Start_Time} to {End_Time}".


* **训练目标**：这实际上是把 ASR 的强制对齐（Force Alignment）任务转化为了 MLLM 的生成任务。



</details>

---

## 15.11 常见陷阱 (Gotchas)

* **编码地狱 (Encoding Hell)**：处理 2015 年以前的中文数据集（如某些早期大学发布的数据）时，常遇到 `GBK` 或 `GB2312` 编码。在 Linux/Python 环境下直接读取会报错或乱码。
* *Solution*: `iconv -f GBK -t UTF-8 input.txt > output.txt`。


* **采样率混搭 (Sample Rate Mismatch)**：电话数据 (Aidatatang) 是 8kHz，有声书 (AISHELL) 是 16kHz。
* *Gotcha*: 直接喂给模型会导致声学特征维度错乱或频谱被压缩。
* *Fix*: **全部上采样 (Upsample) 到 16kHz**。虽然 8k 变 16k 不会增加信息，但能保证模型输入的一致性。


* **测试集泄露 (Test Set Leakage)**：这是学术不端的重灾区，也是工程灾难的源头。
* WenetSpeech 等爬虫数据集中，可能包含了 AISHELL-1 的测试集音频（因为有人把它传到了网上）。
* *Must Do*: 在训练前，使用音频指纹（Audio Fingerprinting）或简单的 MD5 对比，从庞大的训练集中剔除掉所有测试集样本。


* **MP3 vs WAV**:
* MP3 解码需要 CPU 算力。如果在训练时 `On-the-fly` 解码大量 MP3，你的 4090 GPU 可能会因为等待 CPU 喂数据而利用率只有 30%。
* *Pro Tip*: 硬盘便宜，算力贵。预先将 MP3 转码为 **FLAC** 或 **PCM WAV** 存放在 SSD 上。
