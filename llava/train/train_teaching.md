# `llava/train/train.py` 教学版讲解

这份文档不是在重复代码表面含义，而是在回答 4 个问题：

1. 这段代码在整条训练链路里处在什么位置？
2. 它的输入是什么？
3. 它的输出是什么？
4. 为什么要这么写？

建议你一边打开原文件 [`train.py`](/workspace/LLaVA-NeXT/llava/train/train.py)，一边看这份文档。

## 1. 这份文件到底负责什么

[`train.py`](/workspace/LLaVA-NeXT/llava/train/train.py) 是这个仓库最核心的训练入口。

它一口气做了这几件事：

- 定义训练时所有命令行参数
- 定义文本/图像/视频样本怎么预处理
- 定义训练数据集怎么读
- 定义 batch 怎么拼
- 定义模型怎么加载
- 定义训练怎么启动
- 定义训练完怎么保存

也就是说，这不是一个“只负责启动 Trainer”的薄入口，而是一个“把训练主链路全部串起来的大文件”。

你可以把它粗略理解成：

`命令行参数 -> 构建模型 -> 构建数据集 -> 取样本 -> 文本/图像预处理 -> 拼 batch -> Trainer.train() -> 保存模型`

---

## 2. 顶部那几行全局设置在干什么

文件开头有几行很关键：

- `torch.multiprocessing.set_sharing_strategy("file_system")`
- `ImageFile.LOAD_TRUNCATED_IMAGES = True`
- `IS_TOKENIZER_GREATER_THAN_0_14 = ...`

它们分别在解决不同问题。

### 2.1 `set_sharing_strategy("file_system")`

训练时 DataLoader 可能会启动多个 worker，分布式训练又有多个进程，大家之间会共享张量。

这里显式设成 `file_system`，本质上是为了让多进程共享张量更稳一点，减少因为默认共享机制导致的奇怪问题。

你可以把它理解成：  
“先把多进程运行环境调成作者验证过更稳定的一种模式。”

### 2.2 `LOAD_TRUNCATED_IMAGES = True`

多模态训练经常会遇到脏数据。

有些图片文件并不是完全损坏，只是结尾截断了。默认情况下 PIL 会直接报错。  
这里把这个开关打开后，PIL 会尽量把还能读的内容读出来。

目的很简单：  
“不要让一张坏图把整个训练打断。”

### 2.3 `IS_TOKENIZER_GREATER_THAN_0_14`

不同版本的 tokenizer 在这些地方会有微妙差异：

- 特殊 token 是否自动插入
- BOS/EOS 的位置
- 某些分隔符会不会多出 1 个 token

这会直接影响后面“label mask 对不对”。  
所以作者在文件一开始就记住当前 tokenizer 版本，后面某些模板会用它做兼容补偿。

---

## 3. 三个参数类在讲什么

这三个类是训练脚本的参数入口：

- `ModelArguments`
- `DataArguments`
- `TrainingArguments`

它们最终会被 `HfArgumentParser` 自动实例化。

### 3.1 `ModelArguments`

这个类回答的是：  
“我要训练一个什么样的模型？”

核心字段可以按这个思路理解：

- `model_name_or_path`
  - 底座语言模型是谁
  - 例如 Qwen2、Llama、Mistral

- `vision_tower`
  - 视觉编码器是谁
  - 例如 CLIP、SigLIP

- `mm_projector_type`
  - 视觉特征怎么投影到语言模型隐藏空间

- `mm_tunable_parts`
  - 哪些模块参与训练
  - 可以只训 projector，也可以连视觉塔、语言模型一起训

- `mm_patch_merge_type`
  - 图像 patch 特征怎么组织
  - 是直接拉平，还是保留空间结构

- `mm_resampler_type`
  - 视觉 token 送入 LLM 前要不要压缩/重采样

- `rope_scaling_factor` / `rope_scaling_type`
  - 长上下文扩展怎么做

这一类参数主要决定模型结构和可训练范围。

### 3.2 `DataArguments`

这个类回答的是：  
“训练数据从哪里来，图片和视频怎么处理？”

最重要的字段：

- `data_path`
  - 训练样本入口
  - 可以是单个 JSON，也可以是 YAML 数据混合配置

- `image_folder`
  - 图片根目录

- `video_folder`
  - 视频根目录

- `image_aspect_ratio`
  - 图片预处理策略
  - `square` / `pad` / `highres` / `anyres` / `crop_split`

- `video_fps`
  - 视频抽帧频率

- `frames_upbound`
  - 一个视频最多保留多少帧

- `add_time_instruction`
  - 是否把“这几帧分别来自视频什么时间点”写进 prompt

这一类参数主要决定数据如何变成模型输入。

### 3.3 `TrainingArguments`

这个类回答的是：  
“模型怎么优化？”

它继承自 Hugging Face 的 `TrainingArguments`，所以默认就有：

- 学习率
- batch size
- gradient accumulation
- save steps
- output_dir

在这个项目里又额外加了：

- `bits`
  - 是否做 4bit / 8bit 量化训练

- `lora_enable`
  - 是否启用 LoRA

- `mm_projector_lr`
  - projector 单独学习率

- `mm_vision_tower_lr`
  - 视觉塔单独学习率

- `group_by_modality_length`
  - 是否按模态长度分组采样

- `attn_implementation`
  - 用 `flash_attention_2` 还是 `sdpa`

所以这个类控制的是优化与训练运行方式，而不是数据和结构。

---

## 4. `maybe_zero_3` 和保存相关函数为什么这么麻烦

你会看到一组函数：

- `maybe_zero_3`
- `get_peft_state_maybe_zero_3`
- `get_peft_state_non_lora_maybe_zero_3`
- `get_mm_adapter_state_maybe_zero_3`
- `safe_save_model_for_hf_trainer`

这些函数看起来很绕，根源只有一个：

`DeepSpeed ZeRO-3 会把参数切分到不同 GPU 上`

这会带来一个问题：

“保存模型的时候，你手里不一定有完整参数，可能只有一片。”

### 4.1 `maybe_zero_3`

它的作用是：

“如果这个参数被 ZeRO-3 切分了，就先 gather 成完整参数，再拿出来保存。”

所以它本质是一个“取完整参数副本”的工具函数。

### 4.2 `get_peft_state_maybe_zero_3`

这个函数是给 LoRA 用的。

LoRA 训练通常只更新一小部分低秩矩阵，所以保存时不需要整模型，只需要：

- LoRA 权重
- 可选 bias

这个函数就是把这些权重筛出来，并兼容 ZeRO-3。

### 4.3 `get_peft_state_non_lora_maybe_zero_3`

LoRA 场景下，除了 LoRA 权重本身，还可能有一些非 LoRA 参数被训练了。  
这个函数负责把这些参数也单独收集出来。

### 4.4 `get_mm_adapter_state_maybe_zero_3`

如果你只训练多模态 adapter，例如：

- `mm_projector`
- `vision_resampler`

那保存整模型很浪费。  
这个函数只筛出这些 adapter 参数。

### 4.5 `safe_save_model_for_hf_trainer`

这个函数是在统一收口“怎么保存模型”。

它会根据当前训练方式判断：

- 如果只训练多模态 adapter
  - 就只保存 projector / resampler

- 如果是 DeepSpeed 全量训练
  - 让 DeepSpeed 自己处理保存

- 如果是普通训练
  - 正常拿 state_dict 保存

所以它本质上不是“复杂”，而是在解决：

“不同训练策略下，保存逻辑完全不同。”

---

## 5. `smart_tokenizer_and_embedding_resize` 在做什么

这个函数很重要，因为多模态训练经常要给 tokenizer 增加特殊 token，例如：

- `[PAD]`
- `<image>`
- `<im_start>`
- `<im_end>`

问题在于：

如果 tokenizer 新增了 token，而模型 embedding 大小没变，就会维度对不上。

所以这个函数做两件事：

1. 给 tokenizer 加特殊 token
2. 同步扩展模型输入/输出 embedding

然后它还做了一个细节：

“新 token 的 embedding 用旧 token embedding 的均值初始化”

原因是：

如果完全随机初始化，训练一开始新 token 会非常不稳定；  
均值初始化相当于给它一个比较温和的起点。

---

## 6. 文本预处理函数到底在干什么

中间一大坨 `preprocess_*` 容易把人看晕：

- `_tokenize_fn`
- `_mask_targets`
- `_add_speaker_and_signal`
- `preprocess_multimodal`
- `preprocess_llama_2`
- `preprocess_gemma`
- `preprocess_qwen`
- `preprocess_llama3`
- `preprocess_v1`
- `preprocess_mpt`
- `preprocess_plain`
- `preprocess`

其实它们都在做同一件事：

“把原始对话样本变成模型可训练的 `input_ids` 和 `labels`”

### 6.1 预处理的目标是什么

训练一个对话模型时，你不会让它去预测“用户问题”，而是让它预测“助手回答”。

所以预处理的最终目标是：

- `input_ids`
  - 模型真正输入的整段上下文

- `labels`
  - 只有助手回答部分保留原 token
  - 用户、系统提示、图片占位这些位置全部设成 `IGNORE_INDEX`

这样 loss 就只会在助手回答位置计算。

### 6.2 `_mask_targets`

这个函数的作用最核心：

“把不该学习的位置都 mask 掉。”

训练时常见思路是：

- 用户说的话不让模型学
- 系统提示不让模型学
- 只让模型学 assistant 的输出

所以你会看到它用 `IGNORE_INDEX` 去覆盖很多位置。

### 6.3 `preprocess_multimodal`

这个函数处理的是 `<image>` 占位符。

它的任务是把原始数据里的图片标记规范化，例如：

- 确保 `<image>` 在句首
- 如果需要，就替换成 `<im_start><image><im_end>`

为什么要这么做？

因为后面 tokenizer 和模型都假设图片 token 的格式是统一的。  
如果数据里每条样本写法不统一，训练时就会乱。

### 6.4 为什么有这么多 `preprocess_xxx`

因为不同底座模型的对话模板不一样。

例如：

- Qwen 用自己的 chat template
- Llama-3 用自己的 special tokens
- Gemma 分隔符不一样
- MPT 模板风格也不一样

所以这些函数其实不是在做不同任务，而是在做：

“同一件事，不同模型模板下的具体实现。”

### 6.5 `preprocess`

这是总分发函数。

它会根据当前 `conversation_lib.default_conversation` 的模板类型，自动选择走：

- `preprocess_qwen`
- `preprocess_llama3`
- `preprocess_gemma`
- ...

也就是说：

`preprocess()` 是统一入口，`preprocess_xxx()` 是不同底座的具体实现。

---

## 7. `LazySupervisedDataset` 是怎么把样本读出来的

这个类是训练数据集的核心。

它解决三个问题：

1. 样本列表从哪里加载
2. 图片/视频怎么读
3. 单条样本最后怎么变成 `input_ids + labels + image`

### 7.1 为什么叫 Lazy

因为它不是在 `__init__` 里就把所有图片、视频都处理好。

它只会在初始化时：

- 读取 JSON / JSONL / YAML
- 把样本元数据放进 `self.list_data_dict`

真正取样本时，也就是 `__getitem__` 被调用时，才会：

- 打开图片
- 读视频
- 抽帧
- token 化

这就是 lazy 的含义：  
“延迟到真正要用的时候再做重处理。”

### 7.2 `__init__`

它支持三种数据入口：

- 单个 `.json`
- 单个 `.jsonl`
- `.yaml` 数据混合配置

YAML 模式尤其重要，因为它允许你把多个数据集按比例混合进一次训练。

### 7.3 `process_image`

这个函数负责把单张图片变成视觉张量。

它会根据 `image_aspect_ratio` 选择不同路径：

- `highres`
- `anyres`
- `crop_split`
- `pad`
- 默认普通 preprocess

这一步的输出不是文本，而是 pixel tensor。

### 7.4 `_get_item`

这是真正的数据流核心。

对一条样本，它会判断：

- 是图片样本？
- 是视频样本？
- 还是纯文本样本？

然后分别处理。

#### 图片样本

如果样本里有 `image` 字段：

- 读图片
- 单图或多图分别处理
- 调用 `preprocess_multimodal`
- 最后再调用 `preprocess` 生成文本 token

#### 视频样本

如果样本里有 `video` 字段：

- 找到视频文件
- 如果是 `shareVideoGPTV` 这种帧目录格式，自己组帧
- 否则走 `process_video_with_decord`
- 把抽出来的帧送进 image processor
- 如果开启 `add_time_instruction`，把时间信息写进首轮问题
- 再走 `preprocess_multimodal` 和 `preprocess`

#### 纯文本样本

如果既没有图片也没有视频，就只处理文本。

### 7.5 `__getitem__`

这里额外做了容错：

- 如果当前样本坏了，会重试几次
- 还会尝试相邻样本

这说明作者默认训练数据里确实可能有坏文件，所以这里加了防御式处理。

---

## 8. `DataCollatorForSupervisedDataset` 在做什么

Dataset 返回的是“单条样本”。  
Collator 的任务是把很多单样本拼成一个 batch。

它做了几件事：

- 对 `input_ids` 做 padding
- 对 `labels` 做 padding
- 生成 `attention_mask`
- 整理 `images`
- 整理 `image_sizes`
- 整理 `modalities`

为什么 `labels` padding 用 `IGNORE_INDEX`？  
因为 padding 部分不能参与 loss。

为什么图片和视频都放进 `images`？  
因为对模型来说，视频本质上就是“一组帧组成的视觉输入”，在 collator 层可以统一处理。

---

## 9. `get_model` 在解决什么问题

这个函数的作用是：

“根据命令行参数，把正确的多模态模型实例化出来。”

它要处理很多分支，因为底座可能是：

- Llama
- Qwen
- Mistral
- Mixtral
- Gemma

而且还要同时考虑：

- 有没有视觉塔
- 有没有量化
- 有没有自定义 config 覆写

### 9.1 为什么先改 config

如果用户设置了：

- rope scaling
- spatial pool
- resampler

这些配置必须在模型加载前就写进 config。  
否则模型初始化出来的结构就不对了。

### 9.2 为什么按模型名字分支

因为每个底座模型对应的 LLaVA 包装类不一样，例如：

- `LlavaQwenForCausalLM`
- `LlavaLlamaForCausalLM`
- `LlavaMistralForCausalLM`

所以这里实际上是在做：

“按底座类型选择正确的多模态外壳类。”

---

## 10. `train()` 才是整条主线

这就是训练脚本真正的主入口。

你可以按下面顺序读：

### 第一步：解析参数

```python
parser = transformers.HfArgumentParser(...)
model_args, data_args, training_args = ...
```

这一步就是把命令行参数装进三个 dataclass。

### 第二步：准备量化参数

如果 `bits in [4, 8]`：

- 构造 `BitsAndBytesConfig`
- 让模型按低比特方式加载

这一步的核心目标是省显存。

### 第三步：构建模型

```python
model = get_model(...)
```

这一步会根据前面的参数，把底座语言模型和视觉塔组合起来。

### 第四步：处理梯度检查点、LoRA、冻结策略

这里会依次判断：

- 要不要 freeze backbone
- 要不要 prepare k-bit training
- 要不要 gradient checkpointing
- 要不要挂 LoRA

这一段的本质是：

“把模型改造成你想要的可训练形态。”

### 第五步：加载 tokenizer

不同底座模型 padding 偏好不同：

- Mistral / Mixtral 常用 left padding
- Qwen 常用 right padding

这不是小细节，因为 padding side 会影响 mask 和 batch 拼接。

### 第六步：设置 conversation template

这里根据 `model_args.version` 选择对话模板。

这会影响：

- prompt 怎么拼
- labels 怎么 mask
- 图像 token 怎么出现

### 第七步：初始化视觉模块

如果有 `vision_tower`：

- 初始化 vision tower
- 初始化 projector / resampler
- 把 image processor 挂到 `data_args`
- 把图像相关配置写到 `model.config`

然后再根据：

- `mm_tunable_parts`
- `tune_mm_mlp_adapter`
- `tune_mm_vision_resampler`
- `unfreeze_mm_vision_tower`

决定哪些参数真正可训练。

这是多模态训练里最关键的“冻结/解冻策略”阶段。

### 第八步：构建数据模块

```python
data_module = make_supervised_data_module(...)
```

本质就是拿到：

- `train_dataset`
- `data_collator`

### 第九步：实例化 Trainer

```python
trainer = LLaVATrainer(...)
```

这一步之后，训练主循环就交给 Trainer 了。

### 第十步：开始训练

```python
trainer.train()
```

如果输出目录里已经有 checkpoint，就恢复训练；否则从头开始。

### 第十一步：保存

训练结束后：

- 如果是 LoRA
  - 分别保存 LoRA 权重和额外可训练参数

- 否则
  - 走 `safe_save_model_for_hf_trainer`

---

## 11. 你真正应该怎样读这个文件

如果你是零基础，不要从第一行硬啃到最后一行。

正确顺序是：

1. 先读 `train()`
   - 搞清楚总流程

2. 再读 `get_model()`
   - 搞清楚模型怎么构造

3. 再读 `LazySupervisedDataset`
   - 搞清楚一条样本怎么从 json 变成训练输入

4. 再读 `preprocess()` 和对应模板函数
   - 搞清楚文本怎么变成 `input_ids` / `labels`

5. 最后回头看保存相关函数
   - 搞清楚为什么 ZeRO-3 / LoRA 保存这么复杂

---

## 12. 一句话总结整条数据流

一条训练样本在这份文件里的完整路径是：

`json/yaml 样本 -> Dataset 读取 -> 图像/视频预处理 -> 对话模板拼接 -> tokenizer -> labels mask -> collator 拼 batch -> 模型前向 -> Trainer 反向传播 -> 保存权重`

如果你愿意，我下一步可以继续做两件事：

1. 把这份文档补成“逐函数精读版”
2. 直接从 `train.py` 的 `train()` 函数开始，带你一段一段读源码
