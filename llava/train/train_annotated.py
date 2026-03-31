# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np

import time
import random
import yaml
import math
import re
import torch

import transformers
import tokenizers
import deepspeed

from transformers import AutoConfig
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


# 把下面这个类声明成 dataclass，自动生成初始化方法，方便承载配置。
@dataclass
# 定义类 `ModelArguments`，把同一类配置或逻辑组织到一起。
class ModelArguments:
    # 这一组参数控制底座模型、视觉塔以及多模态连接层的结构。
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)



# 把下面这个类声明成 dataclass，自动生成初始化方法，方便承载配置。
@dataclass
# 定义类 `DataArguments`，把同一类配置或逻辑组织到一起。
class DataArguments:
    # 这一组参数控制训练数据的位置、图像/视频读取方式以及预处理策略。
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)


# 把下面这个类声明成 dataclass，自动生成初始化方法，方便承载配置。
@dataclass
# 定义类 `TrainingArguments`，把同一类配置或逻辑组织到一起。
class TrainingArguments(transformers.TrainingArguments):
    # 这一组参数继承自 HF Trainer，并额外补充多模态训练用到的选项。
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    trainer_mode: str = field(default="regular", metadata={"help": "Mode of training (`regular`, `zo`)."})
    zo_eps: float = field(default=1e-3, metadata={"help": "MeZO hyperparameter epsilon."})
    zo_num_directions: int = field(default=1, metadata={"help": "Number of directions for MeZO."})


# @dataclass
# class EvaluationArguments:
#     eval_num_processes: int = field(default=1)
#     task_names: str = field(default=None)
#     model: str = field(default="llava")
#     model_args: Optional[str] = field(default=None)
#     num_fewshot: Optional[int] = field(default=None)
#     batch_size: int = field(default=1)
#     device: Optional[str] = field(default=None)
#     limit: Optional[int] = field(default=None)
#     check_integrity: Optional[bool] = field(default=False)
#     show_task_to_terminal: Optional[bool] = field(default=False)
#     log_samples: Optional[bool] = field(default=True)
#     gen_kwargs: Optional[str] = field(default="")
#     log_samples_suffix: Optional[str] = field(default="")
#     output_path: Optional[str] = field(default="./logs/")


# 定义函数 `maybe_zero_3`，把一段可复用逻辑封装起来。
def maybe_zero_3(param, ignore_status=False, name=None):
    # ZeRO-3 会把参数切分到不同卡上，这里负责在保存时临时聚合出完整参数。
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        # 使用上下文管理器安全地打开资源，结束后会自动清理。
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    # 当前面的条件都不成立时，执行这个兜底分支。
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
# 定义函数 `get_peft_state_maybe_zero_3`，把一段可复用逻辑封装起来。
def get_peft_state_maybe_zero_3(named_params, bias):
    # 提取 LoRA 权重，并在 ZeRO-3 场景下把切分后的参数先 gather 再保存。
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    # 如果前面的条件不成立，再检查 `bias == "all"` 这个分支。
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    # 如果前面的条件不成立，再检查 `bias == "lora_only"` 这个分支。
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                # 给 `bias_name` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            # 如果前面的条件不成立，再检查 `"bias" in k` 这个分支。
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    # 当前面的条件都不成立时，执行这个兜底分支。
    else:
        # 主动抛出异常，说明当前输入或状态不符合预期。
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


# 定义函数 `get_peft_state_non_lora_maybe_zero_3`，把一段可复用逻辑封装起来。
def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    # 提取除 LoRA 外的可训练参数，通常与 LoRA 权重一起落盘。
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


# 定义函数 `get_mm_adapter_state_maybe_zero_3`，把一段可复用逻辑封装起来。
def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    # 只筛出 projector / resampler 这类多模态适配器权重。
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


# 定义函数 `find_all_linear_names`，把一段可复用逻辑封装起来。
def find_all_linear_names(model):
    # 自动扫描语言模型中的线性层，作为 LoRA 的挂载目标。
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            # 给 `names` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


# 定义函数 `safe_save_model_for_hf_trainer`，把一段可复用逻辑封装起来。
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    # 文档字符串：说明下面这个函数或代码块的用途。
    """Collects the state dict and dump to disk."""
    # 如果当前只训练多模态适配器，则只保存 projector / resampler，而不是整模型。
    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    # only has mm_mlp_adapter and mm_vision_resampler in the tuneable parts
    # 如果前面的条件不成立，再检查 `hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts))` 这个分支。
    elif hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
        check_only_save_mm_adapter_tunnable = True
    # 当前面的条件都不成立时，执行这个兜底分支。
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    if check_only_save_mm_adapter_tunnable:
        # 只保存 Adapter，方便继续在基础语言模型上恢复训练或推理。
        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        # 给 `current_folder` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                # 给 `mm_projector_folder` 赋值：拼接平台无关的文件路径。
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                # 把当前张量或权重保存到磁盘文件。
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            # 当前面的条件都不成立时，执行这个兜底分支。
            else:
                # 把当前张量或权重保存到磁盘文件。
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        # 直接结束当前函数。
        return

    if trainer.deepspeed:
        # DeepSpeed 自己处理了分布式保存逻辑，这里直接调用其保存接口。
        trainer.save_model(output_dir)
        # 直接结束当前函数。
        return

    # 给 `state_dict` 赋值：构造一个字典，把相关结果打包返回。
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


# 定义函数 `smart_tokenizer_and_embedding_resize`，把一段可复用逻辑封装起来。
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    # 文档字符串：说明下面这个函数或代码块的用途。
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # 先把新增特殊 token 注册到 tokenizer 中。
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # 再把模型的输入/输出 embedding 扩展到新的词表大小。
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        # 用已有 embedding 的均值初始化新增 token，避免随机初始化过于不稳定。
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# 定义函数 `_tokenize_fn`，把一段可复用逻辑封装起来。
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    # 文档字符串：说明下面这个函数或代码块的用途。
    """Tokenize a list of strings."""
    # 对若干段文本分别做 tokenize，并保留每段的有效 token 长度。
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


# 定义函数 `_mask_targets`，把一段可复用逻辑封装起来。
def _mask_targets(target, tokenized_lens, speakers):
    # 监督微调时只学习 assistant 的输出，因此 human 段落要整体 mask 掉。
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


# 定义函数 `_add_speaker_and_signal`，把一段可复用逻辑封装起来。
def _add_speaker_and_signal(header, source, get_conversation=True):
    # 文档字符串：说明下面这个函数或代码块的用途。
    """Add speaker and start/end signal on each round."""
    # 把原始 `human/gpt` 对话样本格式化成带角色标签的纯文本 prompt。
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        # 如果前面的条件不成立，再检查 `from_str.lower() == "gpt"` 这个分支。
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        # 当前面的条件都不成立时，执行这个兜底分支。
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            # 把当前片段继续拼接到完整对话字符串后面。
            conversation += sentence["value"]
    # 把当前片段继续拼接到完整对话字符串后面。
    conversation += BEGIN_SIGNAL
    return conversation


# 定义函数 `preprocess_multimodal`，把一段可复用逻辑封装起来。
def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    # 统一整理 `<image>` 占位符的位置和包裹形式，保证后续 tokenizer 行为一致。
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            # 给 `num_im` 赋值：用正则解析字符串里的结构化信息。
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


# 定义函数 `preprocess_llama_2`，把一段可复用逻辑封装起来。
def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    # 按 LLaMA-2 对话模板拼接对话，并构造训练 labels。
    # 给 `conv` 赋值：复制当前对象，后续修改副本时不污染原对象。
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # 先把每条对话样本套进对应模板，得到最终训练 prompt。
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # 用断言强制检查一个前提条件，条件不满足就立即报错。
            assert role == conv.roles[j % 2], f"{i}"
            # 往对话模板中追加一轮消息，逐步构造完整对话。
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # 如果样本里带图像 token，就用专门的 tokenizer_image_token 处理。

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    # 当前面的条件都不成立时，执行这个兜底分支。
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    # 用断言强制检查一个前提条件，条件不满足就立即报错。
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # 只保留 assistant 回复为可学习目标，其余位置统一设为 IGNORE_INDEX。
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # 给 `rounds` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            # 给 `parts` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                # 给 `round_len` 赋值：把带 `<image>` 占位符的文本编码成 token，并把图片位置映射成专用图像 token。
                round_len = len(tokenizer_image_token(rou, tokenizer))
                # 给 `instruction_len` 赋值：把带 `<image>` 占位符的文本编码成 token，并把图片位置映射成专用图像 token。
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            # 当前面的条件都不成立时，执行这个兜底分支。
            else:
                # 给 `round_len` 赋值：把文本编码成 token id 序列，供模型训练或推理使用。
                round_len = len(tokenizer(rou).input_ids)
                # 给 `instruction_len` 赋值：把文本编码成 token id 序列，供模型训练或推理使用。
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
                target[:] = IGNORE_INDEX
                # 把关键信息打印到终端，便于调试或观察运行状态。
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


# 定义函数 `preprocess_gemma`，把一段可复用逻辑封装起来。
def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    # Gemma 的模板和分隔符与其他模型不同，需要单独处理。
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # 先按 Gemma 的多轮对话格式拼出完整 prompt。
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            # 用断言强制检查一个前提条件，条件不满足就立即报错。
            assert role == conv.roles[j % 2], f"{i}"
            # 往对话模板中追加一轮消息，逐步构造完整对话。
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # 对完整 prompt 做 tokenize，图文样本使用图像感知版 tokenizer。
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    # 当前面的条件都不成立时，执行这个兜底分支。
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    # 用断言强制检查一个前提条件，条件不满足就立即报错。
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # 只让 assistant 的部分参与 loss 计算。
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            # 给 `parts` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                # 给 `round_len` 赋值：把带 `<image>` 占位符的文本编码成 token，并把图片位置映射成专用图像 token。
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                # 给 `instruction_len` 赋值：把带 `<image>` 占位符的文本编码成 token，并把图片位置映射成专用图像 token。
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            # 当前面的条件都不成立时，执行这个兜底分支。
            else:
                # 给 `round_len` 赋值：把文本编码成 token id 序列，供模型训练或推理使用。
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                # 给 `instruction_len` 赋值：把文本编码成 token id 序列，供模型训练或推理使用。
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
                target[:] = IGNORE_INDEX
                # 把关键信息打印到终端，便于调试或观察运行状态。
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


# 定义函数 `preprocess_qwen`，把一段可复用逻辑封装起来。
def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # Qwen 依赖 chat template，因此这里直接复用 tokenizer.apply_chat_template。
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # 拷贝一份 tokenizer，避免把 `<image>` 永久污染到外部 tokenizer 实例里。
    # 深拷贝一份对象，避免后续修改影响原始输入。
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    # 给 `nl_tokens` 赋值：把文本编码成 token id 序列，供模型训练或推理使用。
    nl_tokens = tokenizer("\n").input_ids

    # 重置 chat template，避免系统消息被重复插入。
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # 按轮次把整段对话编码成输入 token 和监督目标。
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # 先编码系统消息；系统消息只作为上下文，不参与 loss。
        # 把当前这轮编码结果接到总输入序列后面。
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        # 把当前这轮标签结果接到总标签序列后面。
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # 兼容两种字段命名：role/content 与 from/value。
            # 尝试执行下面这段可能出错的逻辑。
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            # 给 `encode_id` 赋值：用 tokenizer 内置的 chat template 把对话格式化成模型实际看到的字符串/token。
            encode_id = tokenizer.apply_chat_template(conv)
            # 把当前这轮编码结果接到总输入序列后面。
            input_id += encode_id
            # user/system 位置不计算 loss，assistant 位置保留原 token 作为监督目标。
            if role in ["user", "system"]:
                # 把当前这轮标签结果接到总标签序列后面。
                target += [IGNORE_INDEX] * len(encode_id)
            # 当前面的条件都不成立时，执行这个兜底分支。
            else:
                # 把当前这轮标签结果接到总标签序列后面。
                target += encode_id



        # 用断言强制检查一个前提条件，条件不满足就立即报错。
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            # 特殊控制 token 需要取消 mask，否则 chat template 结构会被模型遗忘。
            if encode_id in unmask_tokens_idx:
                # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
                target[idx] = encode_id
            # 把 tokenizer 私有的 `<image>` id 映射回项目统一的 IMAGE_TOKEN_INDEX。
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


# 定义函数 `preprocess_llama3`，把一段可复用逻辑封装起来。
def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # Llama-3 同样依赖 chat template，但它的特殊 token 集与 Qwen 不同。
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # 同样先复制 tokenizer，避免原始 tokenizer 被就地修改。
    # 深拷贝一份对象，避免后续修改影响原始输入。
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # llama3 tokenizer 会自动补 bos，这里包一层是为了避免重复。
    # 定义函数 `safe_tokenizer_llama3`，把一段可复用逻辑封装起来。
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # 用 Llama-3 的 chat template 逐轮构造训练样本。
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # 系统消息只提供上下文，不参与损失。
        # 把当前这轮编码结果接到总输入序列后面。
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        # 把当前这轮标签结果接到总标签序列后面。
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # 兼容 llava 原始数据格式与 role/content 风格数据格式。
            # 尝试执行下面这段可能出错的逻辑。
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            # apply_chat_template 返回的首位 bos 在多轮拼接时不需要重复保留。
            # 给 `encode_id` 赋值：用 tokenizer 内置的 chat template 把对话格式化成模型实际看到的字符串/token。
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            # 把当前这轮编码结果接到总输入序列后面。
            input_id += encode_id
            if role in ["user", "system"]:
                # 把当前这轮标签结果接到总标签序列后面。
                target += [IGNORE_INDEX] * len(encode_id)
            # 当前面的条件都不成立时，执行这个兜底分支。
            else:
                # 把当前这轮标签结果接到总标签序列后面。
                target += encode_id



        # 用断言强制检查一个前提条件，条件不满足就立即报错。
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            # 保留 chat template 的结构 token，避免模板被完全 mask 掉。
            if encode_id in unmask_tokens_idx:
                # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


# 定义函数 `preprocess_v1`，把一段可复用逻辑封装起来。
def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    # 处理 Vicuna/LLaVA v1 风格模板。
    # 给 `conv` 赋值：复制当前对象，后续修改副本时不污染原对象。
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # 先把多轮对话套进模板，得到训练 prompt。
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # 用断言强制检查一个前提条件，条件不满足就立即报错。
            assert role == conv.roles[j % 2], f"{i}"
            # 往对话模板中追加一轮消息，逐步构造完整对话。
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # 之后把完整 prompt 转成 token 序列。

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    # 当前面的条件都不成立时，执行这个兜底分支。
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    # 用断言强制检查一个前提条件，条件不满足就立即报错。
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # human 段落全部 mask，assistant 段落参与损失。
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # 给 `rounds` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            # 给 `parts` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                # 给 `round_len` 赋值：把带 `<image>` 占位符的文本编码成 token，并把图片位置映射成专用图像 token。
                round_len = len(tokenizer_image_token(rou, tokenizer))
                # 给 `instruction_len` 赋值：把带 `<image>` 占位符的文本编码成 token，并把图片位置映射成专用图像 token。
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            # 当前面的条件都不成立时，执行这个兜底分支。
            else:
                # 给 `round_len` 赋值：把文本编码成 token id 序列，供模型训练或推理使用。
                round_len = len(tokenizer(rou).input_ids)
                # 给 `instruction_len` 赋值：把文本编码成 token id 序列，供模型训练或推理使用。
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
                target[:] = IGNORE_INDEX
                # 把关键信息打印到终端，便于调试或观察运行状态。
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


# 定义函数 `preprocess_mpt`，把一段可复用逻辑封装起来。
def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    # MPT 使用自己的分隔风格，目标 mask 规则也要跟着调整。
    # 给 `conv` 赋值：复制当前对象，后续修改副本时不污染原对象。
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # 先构造模板化后的完整对话文本。
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # 用断言强制检查一个前提条件，条件不满足就立即报错。
            assert role == conv.roles[j % 2], f"{i}"
            # 往对话模板中追加一轮消息，逐步构造完整对话。
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # 再做 token 化。

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    # 当前面的条件都不成立时，执行这个兜底分支。
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    # 用断言强制检查一个前提条件，条件不满足就立即报错。
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # 仅监督 assistant 回复部分。
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # 给 `rounds` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            # 给 `parts` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                # 给 `round_len` 赋值：把带 `<image>` 占位符的文本编码成 token，并把图片位置映射成专用图像 token。
                round_len = len(tokenizer_image_token(rou, tokenizer))
                # 给 `instruction_len` 赋值：把带 `<image>` 占位符的文本编码成 token，并把图片位置映射成专用图像 token。
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            # 当前面的条件都不成立时，执行这个兜底分支。
            else:
                # 给 `round_len` 赋值：把文本编码成 token id 序列，供模型训练或推理使用。
                round_len = len(tokenizer(rou).input_ids)
                # 给 `instruction_len` 赋值：把文本编码成 token id 序列，供模型训练或推理使用。
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
                target[:] = IGNORE_INDEX
                # 把关键信息打印到终端，便于调试或观察运行状态。
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


# 定义函数 `preprocess_plain`，把一段可复用逻辑封装起来。
def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # 最简单的模式：只保留一张图和一段回答，常用于预训练 projector。
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        # 用断言强制检查一个前提条件，条件不满足就立即报错。
        assert len(source) == 2
        # 用断言强制检查一个前提条件，条件不满足就立即报错。
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        # 给 `tokenized_len` 赋值：把带 `<image>` 占位符的文本编码成 token，并把图片位置映射成专用图像 token。
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        # 直接修改标签张量中的某一段，决定这些位置是否参与损失计算。
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


# 定义函数 `preprocess`，把一段可复用逻辑封装起来。
def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # 先根据当前选定的模板风格分派到对应的预处理实现。
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # 默认路径下，把多轮对话手动拼接成统一文本。
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # 再把拼接后的整段对话转成 token，并构造 labels。
    # 定义函数 `get_tokenize_len`，把一段可复用逻辑封装起来。
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    # 当前面的条件都不成立时，执行这个兜底分支。
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            # 给 `tokenized_lens` 赋值：计算当前对象的长度。
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        # 当前面的条件都不成立时，执行这个兜底分支。
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


# 定义类 `LazySupervisedDataset`，把同一类配置或逻辑组织到一起。
class LazySupervisedDataset(Dataset):
    # 定义函数 `__init__`，把一段可复用逻辑封装起来。
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        # 这里只建立样本清单，真正耗时的图片/视频预处理延迟到 __getitem__ 再做。
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = []

        # 支持 `{a,b,c}.json` 这种简写，一次加载多个 JSON。
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            # 给 `file_names` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
            file_names = file_pattern.split(",")
            # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
                rank0_print(f"Loading {full_path}")
                # 使用上下文管理器安全地打开资源，结束后会自动清理。
                with open(full_path, "r") as file:
                    # 给 `cur_data_dict` 赋值：从 JSON 文件一次性读入结构化样本。
                    cur_data_dict = json.load(file)
                    # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        # 如果前面的条件不成立，再检查 `data_path.endswith(".yaml")` 这个分支。
        elif data_path.endswith(".yaml"):
            # YAML 模式允许把多个数据源混在一起，并指定各自采样策略。
            # 使用上下文管理器安全地打开资源，结束后会自动清理。
            with open(data_path, "r") as file:
                # 给 `yaml_data` 赋值：从 YAML 文件读取多数据源配置。
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        # 使用上下文管理器安全地打开资源，结束后会自动清理。
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    # 如果前面的条件不成立，再检查 `json_path.endswith(".json")` 这个分支。
                    elif json_path.endswith(".json"):
                        # 使用上下文管理器安全地打开资源，结束后会自动清理。
                        with open(json_path, "r") as json_file:
                            # 给 `cur_data_dict` 赋值：从 JSON 文件一次性读入结构化样本。
                            cur_data_dict = json.load(json_file)
                    # 当前面的条件都不成立时，执行这个兜底分支。
                    else:
                        # 主动抛出异常，说明当前输入或状态不符合预期。
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            # 给 `sampling_number` 赋值：计算当前对象的长度。
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        # 当前面的条件都不成立时，执行这个兜底分支。
                        else:
                            sampling_number = int(sampling_number)

                    # 根据 first/end/random 等策略对子数据集做裁剪。
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    # 如果前面的条件不成立，再检查 `sampling_strategy == "end" and sampling_number is not None` 这个分支。
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    # 如果前面的条件不成立，再检查 `sampling_strategy == "random" and sampling_number is not None` 这个分支。
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        # 当前面的条件都不成立时，执行这个兜底分支。
        else:
            data_args.dataset_paths = [data_path]
            # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
            rank0_print(f"Loading {data_path}")
            # 使用上下文管理器安全地打开资源，结束后会自动清理。
            with open(data_path, "r") as file:
                # 给 `cur_data_dict` 赋值：从 JSON 文件一次性读入结构化样本。
                cur_data_dict = json.load(file)
                # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    # 定义函数 `__len__`，把一段可复用逻辑封装起来。
    def __len__(self):
        # 返回样本总数，供 DataLoader / Sampler 使用。
        return len(self.list_data_dict)

    @property
    # 定义函数 `lengths`，把一段可复用逻辑封装起来。
    def lengths(self):
        # 给长度分组采样器一个粗粒度长度估计。
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    # 定义函数 `modality_lengths`，把一段可复用逻辑封装起来。
    def modality_lengths(self):
        # 多模态样本记正值，纯文本样本记负值，便于按模态分桶采样。
        length_list = []
        for sample in self.list_data_dict:
            # 给 `cur_len` 赋值：计算当前对象的长度。
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            # 用断言强制检查一个前提条件，条件不满足就立即报错。
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            # 当前面的条件都不成立时，执行这个兜底分支。
            else:
                length_list.append(-cur_len)
        return length_list

    # 定义函数 `process_image`，把一段可复用逻辑封装起来。
    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        # 按配置把单张图片转成模型实际接收的 pixel tensor。
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        # 尝试执行下面这段可能出错的逻辑。
        try:
            # 给 `image` 赋值：从磁盘读取图片，并转成 PIL 图像对象。
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        # 如果上面的 try 出现异常，就走这个异常处理分支。
        except Exception as exn:
            # 把关键信息打印到终端，便于调试或观察运行状态。
            print(f"Failed to open image {image_file}. Exception:", exn)
            # 主动抛出异常，说明当前输入或状态不符合预期。
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            # 高分辨率模式会把图像切成多块 patch。
            # 给 `image` 赋值：按高分辨率模式把大图切成多个 patch，提高细节保留能力。
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        # 如果前面的条件不成立，再检查 `image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio` 这个分支。
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            # anyres 模式会根据原图比例选择最合适的 patch 网格。
            # 给 `image` 赋值：按任意分辨率策略切图并编码，尽量保留原图比例信息。
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        # 如果前面的条件不成立，再检查 `image_aspect_ratio == "crop_split"` 这个分支。
        elif image_aspect_ratio == "crop_split":
            # 给 `image` 赋值：按高分辨率模式把大图切成多个 patch，提高细节保留能力。
            image = process_highres_image_crop_split(image, self.data_args)
        # 如果前面的条件不成立，再检查 `image_aspect_ratio == "pad"` 这个分支。
        elif image_aspect_ratio == "pad":

            # 定义函数 `expand2square`，把一段可复用逻辑封装起来。
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                # 如果前面的条件不成立，再检查 `width > height` 这个分支。
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                # 当前面的条件都不成立时，执行这个兜底分支。
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            # 给 `image` 赋值：把图片或视频帧转换成视觉编码器可直接接收的像素张量。
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # 当前面的条件都不成立时，执行这个兜底分支。
        else:
            # 给 `image` 赋值：把图片或视频帧转换成视觉编码器可直接接收的像素张量。
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    # 定义函数 `__getitem__`，把一段可复用逻辑封装起来。
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # 数据文件容易损坏，这里做多次重试，尽量避免单个坏样本中断整轮训练。
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            # 尝试执行下面这段可能出错的逻辑。
            try:
                sample = self._get_item(i)
                return sample
            # 如果上面的 try 出现异常，就走这个异常处理分支。
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                # 把关键信息打印到终端，便于调试或观察运行状态。
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            # 尝试执行下面这段可能出错的逻辑。
            try:
                # 给 `next_index` 赋值：计算当前对象的长度。
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            # 如果上面的 try 出现异常，就走这个异常处理分支。
            except Exception as e:
                # no need to sleep
                # 把关键信息打印到终端，便于调试或观察运行状态。
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        # 尝试执行下面这段可能出错的逻辑。
        try:
            sample = self._get_item(i)
            return sample
        # 如果上面的 try 出现异常，就走这个异常处理分支。
        except Exception as e:
            # 主动抛出异常，说明当前输入或状态不符合预期。
            raise e

    # 定义函数 `_get_item`，把一段可复用逻辑封装起来。
    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        # 真正的单样本解析逻辑：读文本、读图片/视频、再统一做 token 化。
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        # 用断言强制检查一个前提条件，条件不满足就立即报错。
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            # 图像样本既支持单图，也支持多图。
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            # 当前面的条件都不成立时，执行这个兜底分支。
            else:
                image = [self.process_image(image_file)]
            # 给 `sources` 赋值：深拷贝一份对象，避免后续修改影响原始输入。
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        # 如果前面的条件不成立，再检查 `"video" in sources[0]` 这个分支。
        elif "video" in sources[0]:
            # 视频样本会先抽帧，再复用图像处理器做逐帧预处理。
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.data_args.video_folder
            # 给 `video_file` 赋值：拼接平台无关的文件路径。
            video_file = os.path.join(video_folder, video_file)
            # 给 `suffix` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
            suffix = video_file.split(".")[-1]
            if not os.path.exists(video_file):
                # 把关键信息打印到终端，便于调试或观察运行状态。
                print("File {} not exist!".format(video_file))

            # 尝试执行下面这段可能出错的逻辑。
            try:
                if "shareVideoGPTV" in video_file:
                    # shareVideoGPTV 是特殊存储格式：视频预先被解成了帧目录。
                    # 给 `frame_files` 赋值：拼接平台无关的文件路径。
                    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                    # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
                    if self.data_args.force_sample:
                        num_frames_to_sample = self.data_args.frames_upbound
                    # 当前面的条件都不成立时，执行这个兜底分支。
                    else:
                        num_frames_to_sample = 10

                    avg_fps = 2

                    # 给 `total_frames` 赋值：计算当前对象的长度。
                    total_frames = len(frame_files)
                    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)


                    frame_time = [i/2 for i in sampled_indices]
                    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                    video_time = total_frames / avg_fps

                    # Read and store the sampled frames
                    video = []
                    for idx in sampled_indices:
                        frame_path = frame_files[idx]
                        # 尝试执行下面这段可能出错的逻辑。
                        try:
                            # 使用上下文管理器安全地打开资源，结束后会自动清理。
                            with Image.open(frame_path) as img:
                                frame = img.convert("RGB")
                                video.append(frame)
                        # 如果上面的 try 出现异常，就走这个异常处理分支。
                        except IOError:
                            # 把关键信息打印到终端，便于调试或观察运行状态。
                            print(f"Failed to read frame at path: {frame_path}")
                # 当前面的条件都不成立时，执行这个兜底分支。
                else:
                    # 普通视频文件走 decord 抽帧逻辑。
                    video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)

                processor = self.data_args.image_processor
                # 给 `image` 赋值：把图片或视频帧转换成视觉编码器可直接接收的像素张量。
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                if self.data_args.add_time_instruction:
                    # 可选地把视频时长与采样帧时间写入首轮提示词。
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                    sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                image = [(image, video[0].size, "video")]
                # 给 `sources` 赋值：深拷贝一份对象，避免后续修改影响原始输入。
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                # print(sources)
            # 如果上面的 try 出现异常，就走这个异常处理分支。
            except Exception as e:
                # 把关键信息打印到终端，便于调试或观察运行状态。
                print(f"Error: {e}")
                # 把关键信息打印到终端，便于调试或观察运行状态。
                print(f"Failed to read video file: {video_file}")
                return self._get_item(i + 1)
        # 当前面的条件都不成立时，执行这个兜底分支。
        else:
            # 给 `sources` 赋值：深拷贝一份对象，避免后续修改影响原始输入。
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # 对于训练代码而言，视频和图像都走 has_image=True 这条多模态分支。
        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        # 把当前样本处理结果整理成字典，便于后续统一传递。
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        # 当前面的条件都不成立时，执行这个兜底分支。
        else:
            prompt = None

        if isinstance(i, int):
            # 把当前样本处理结果整理成字典，便于后续统一传递。
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # 把模态张量塞回 data_dict，供 collator 聚合成 batch。
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        # 如果前面的条件不成立，再检查 `"video" in self.list_data_dict[i]` 这个分支。
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        # 如果前面的条件不成立，再检查 `self.data_args.is_multimodal` 这个分支。
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)

        return data_dict


# 把下面这个类声明成 dataclass，自动生成初始化方法，方便承载配置。
@dataclass
# 定义类 `DataCollatorForSupervisedDataset`，把同一类配置或逻辑组织到一起。
class DataCollatorForSupervisedDataset(object):
    # 文档字符串：说明下面这个函数或代码块的用途。
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    # 定义函数 `pad_sequence`，把一段可复用逻辑封装起来。
    def pad_sequence(self, input_ids, batch_first, padding_value):
        # 兼容 left padding 和 right padding 两种 tokenizer 行为。
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    # 定义函数 `__call__`，把一段可复用逻辑封装起来。
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 把若干条样本拼成一个 batch，并同步整理图像/视频张量。
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # 某些 tokenizer 没有 pad token，这里给一个保底值。
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        if "image" in instances[0]:
            # 图像和视频在 collator 层统一整理成 `images + modalities + image_sizes`。
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        return batch


# 定义函数 `make_supervised_data_module`，把一段可复用逻辑封装起来。
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    # 文档字符串：说明下面这个函数或代码块的用途。
    """Make dataset and collator for supervised fine-tuning."""
    # 返回 Trainer 需要的标准数据模块。
    # 给 `train_dataset` 赋值：实例化惰性加载的数据集，样本会在取用时才做预处理。
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    # 给 `data_collator` 赋值：实例化 batch 拼接器，把单条样本整理成训练批次。
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


# 定义函数 `get_model`，把一段可复用逻辑封装起来。
def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    # 根据底座模型名称、视觉塔配置和量化配置构建最终模型实例。
    # 用断言强制检查一个前提条件，条件不满足就立即报错。
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        # 主动抛出异常，说明当前输入或状态不符合预期。
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    # 给 `customized_kwargs` 赋值：构造一个字典，把相关结果打包返回。
    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    # 这部分参数会覆盖原始 checkpoint 的 config。
    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        # 给 `cfg_pretrained` 赋值：读取预训练模型的配置文件，便于在加载前覆写配置项。
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        # 用断言强制检查一个前提条件，条件不满足就立即报错。
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
        # overwrite_config["max_sequence_length"] = model_args.max_sequence_length
        # overwrite_config["tokenizer_model_max_length"] = model_args.tokenizer_model_max_length

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if overwrite_config:
        # 如果用户指定了 rope/pool/resampler 等参数，就在加载前注入 config。
        # 用断言强制检查一个前提条件，条件不满足就立即报错。
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained

    if model_args.model_class_name is not None:
        # 用户显式指定模型类时，直接按类名实例化。
        actual_model_class_name = f"{model_args.model_class_name}ForCausalLM"
        model_class = getattr(transformers, actual_model_class_name)
        # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
        rank0_print(f"Using model class {model_class} from {model_args.model_class_name}")
        # 从预训练权重初始化模型或 tokenizer。
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    # 如果前面的条件不成立，再检查 `model_args.vision_tower is not None` 这个分支。
    elif model_args.vision_tower is not None:
        # 常见路径：根据底座类型构建带视觉塔的 LlavaXXXForCausalLM。
        if "mixtral" in model_args.model_name_or_path.lower():
            # 从预训练权重初始化模型或 tokenizer。
            model = LlavaMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        # 如果前面的条件不成立，再检查 `"mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower()` 这个分支。
        elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
            # 从预训练权重初始化模型或 tokenizer。
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        # 如果前面的条件不成立，再检查 `(` 这个分支。
        elif (
            "wizardlm-2" in model_args.model_name_or_path.lower()
            or "vicuna" in model_args.model_name_or_path.lower()
            or "llama" in model_args.model_name_or_path.lower()
            or "yi" in model_args.model_name_or_path.lower()
            or "nous-hermes" in model_args.model_name_or_path.lower()
            and "wizard-2" in model_args.model_name_or_path.lower()
        ):
            # 从预训练权重初始化模型或 tokenizer。
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        # 如果前面的条件不成立，再检查 `"qwen" in model_args.model_name_or_path.lower()` 这个分支。
        elif "qwen" in model_args.model_name_or_path.lower():
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                # 从预训练权重初始化模型或 tokenizer。
                model = LlavaQwenMoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            # 当前面的条件都不成立时，执行这个兜底分支。
            else:
                # 从预训练权重初始化模型或 tokenizer。
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
        # 如果前面的条件不成立，再检查 `"gemma" in model_args.model_name_or_path.lower()` 这个分支。
        elif "gemma" in model_args.model_name_or_path.lower():
            # 从预训练权重初始化模型或 tokenizer。
            model = LlavaGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        # 当前面的条件都不成立时，执行这个兜底分支。
        else:
            # 主动抛出异常，说明当前输入或状态不符合预期。
            raise ValueError(f"Unknown model class {model_args}")
    # 当前面的条件都不成立时，执行这个兜底分支。
    else:
        # 没有视觉塔时退化成纯语言模型。
        # 从预训练权重初始化模型或 tokenizer。
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    return model


# 定义函数 `train`，把一段可复用逻辑封装起来。
def train(attn_implementation=None):
    # 整个训练脚本的真正入口。
    global local_rank

    # 从命令行解析模型、数据和训练参数。
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.verbose_logging:
        # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        # 4bit/8bit 模式下额外准备 bitsandbytes 的量化加载参数。
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                # 给 `quantization_config` 赋值：定义 4bit/8bit 量化加载参数，减少显存占用。
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    # 构建模型并关闭 use_cache，避免训练时缓存 KV 浪费显存。
    # 从预训练权重初始化模型或 tokenizer。
    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        # 只训练上层适配器时，可以把底座全部冻结。
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        # k-bit 训练前，需要先把模型转换到 PEFT 推荐的可训练形态。
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        # 把量化模型调整成可训练形态，避免低比特训练出问题。
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        # 开启 gradient checkpointing 时，确保输入 embedding 支持梯度回传。
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        # 当前面的条件都不成立时，执行这个兜底分支。
        else:

            # 定义函数 `make_inputs_require_grad`，把一段可复用逻辑封装起来。
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        # 如果启用 LoRA，就在语言模型线性层上挂低秩适配器。
        from peft import LoraConfig, get_peft_model

        # 给 `lora_config` 赋值：定义 LoRA 低秩适配器的超参数。
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            # 给 `target_modules` 赋值：扫描模型里的线性层名称，作为 LoRA 的目标模块集合。
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
        rank0_print("Adding LoRA adapters...")
        # 把 LoRA 适配器挂到基础模型上。
        model = get_peft_model(model, lora_config)

    # 不同底座模型对 padding side 的偏好不同，这里分别处理 tokenizer。
    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        # 从预训练权重初始化模型或 tokenizer。
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    # 如果前面的条件不成立，再检查 `"qwen" in model_args.model_name_or_path.lower()` 这个分支。
    elif "qwen" in model_args.model_name_or_path.lower():
        # 从预训练权重初始化模型或 tokenizer。
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    # 如果前面的条件不成立，再检查 `(` 这个分支。
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        # 从预训练权重初始化模型或 tokenizer。
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        # 旧版模板如果没有 pad token，就手动补一个并同步扩展 embedding。
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                # 给 `special_tokens_dict` 赋值：构造一个字典，把相关结果打包返回。
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    # 如果前面的条件不成立，再检查 `model_args.version == "v0.5"` 这个分支。
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    # 当前面的条件都不成立时，执行这个兜底分支。
    else:
        # 新版模板通常直接复用 unk 作为 pad，并选择对应的 conversation template。
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        # 当前面的条件都不成立时，执行这个兜底分支。
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        # 初始化视觉塔、projector、resampler 等视觉相关组件。
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        # 视觉塔通常单独放到 bf16/fp16 上运行，以降低显存占用。
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        # 把视觉处理器挂到 data_args 上，供 dataset 读取图片和视频时复用。
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            # 支持把 `(1x1),...,(6x6)` 这种字符串展开成具体的 patch 网格列表。
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                # 尝试执行下面这段可能出错的逻辑。
                try:
                    patch_size = data_args.image_processor.size[0]
                # 如果上面的 try 出现异常，就走这个异常处理分支。
                except Exception as e:
                    patch_size = data_args.image_processor.size["shortest_edge"]

                # 用断言强制检查一个前提条件，条件不满足就立即报错。
                assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
                # Use regex to extract the range from the input string
                # 给 `matches` 赋值：用正则解析字符串里的结构化信息。
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                # 给 `grid_pinpoints` 赋值：生成整数序列，供循环使用。
                grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
            # 如果前面的条件不成立，再检查 `isinstance(data_args.image_grid_pinpoints, str)` 这个分支。
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.add_time_instruction = data_args.add_time_instruction
        model.config.force_sample = data_args.force_sample
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride

        # 决定哪些模块参与训练：传统开关模式或 mm_tunable_parts 精细控制模式。
        if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
            if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
                # 如果只训练适配器模块，先把整模型冻结，再单独打开对应子模块梯度。
                model.requires_grad_(False)
            if model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
            if training_args.freeze_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = False

            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)
            # 当前面的条件都不成立时，执行这个兜底分支。
            else:
                vision_tower.requires_grad_(False)

        # 当前面的条件都不成立时，执行这个兜底分支。
        else:
            # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
            # Set the entire model to not require gradients by default
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            model.get_model().vision_resampler.requires_grad_(False)
            # Parse the mm_tunable_parts to decide which parts to unfreeze
            # 给 `tunable_parts` 赋值：按分隔符把字符串拆成若干段，便于后续解析。
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if "mm_vision_resampler" in tunable_parts:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                        param.requires_grad_(True)

        # 给 `total_params` 赋值：统计模型参数量，便于核对当前训练规模。
        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
        # 给 `trainable_params` 赋值：统计模型参数量，便于核对当前训练规模。
        trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        # 输出总参数量和可训练参数量，方便确认当前训练策略是否符合预期。
        # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
        rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
        # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
        rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        # 把图像相关特殊 token 注入 tokenizer 与 embedding。
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        # 量化训练下，LoRA / norm / lm_head 需要额外处理 dtype，避免数值不稳定。
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    # 给 `module` 赋值：把张量或模块移动到指定设备/精度。
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                # 给 `module` 赋值：把张量或模块移动到指定设备/精度。
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        # 给 `module` 赋值：把张量或模块移动到指定设备/精度。
                        module = module.to(torch.bfloat16)

    # 构建数据模块并实例化自定义 Trainer。
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # 给 `trainer` 赋值：实例化自定义 Trainer，负责训练循环、保存和采样逻辑。
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # 输出目录里已有 checkpoint 时，自动从最近中断位置恢复。
        # 启动 Trainer 的训练循环，内部会执行前向、反向传播和参数更新。
        trainer.train(resume_from_checkpoint=True)
    # 当前面的条件都不成立时，执行这个兜底分支。
    else:
        # 启动 Trainer 的训练循环，内部会执行前向、反向传播和参数更新。
        trainer.train()
    # 保存 Trainer 的状态信息，例如 optimizer / scheduler / rng state。
    # 保存 Trainer 的运行状态，便于下次从断点继续训练。
    trainer.save_state()

    # 训练完成后恢复 use_cache，便于后续推理。
    model.config.use_cache = True

    if training_args.lora_enable:
        # LoRA 训练会分别保存 LoRA 权重和额外可训练参数。
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                # 把模型配置单独保存下来，保证之后能正确恢复结构。
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                # 把生成配置保存下来，保证推理参数可复现。
                model.generation_config.save_pretrained(training_args.output_dir)
            # 把模型按 Hugging Face 约定格式保存，方便后续加载。
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            # 把当前张量或权重保存到磁盘文件。
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    # 当前面的条件都不成立时，执行这个兜底分支。
    else:
        # 全量或 adapter 训练走统一的安全保存逻辑。
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # 只让主进程打印日志，避免多卡训练时重复输出相同信息。
    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    # 允许直接 `python llava/train/train.py` 方式启动训练。
    train()
