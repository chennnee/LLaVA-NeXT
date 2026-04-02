#    Copyright 2024 Hao Zhang
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


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


# 这个文件把 Hugging Face 的 Qwen2 因果语言模型包装成 LLaVA 的多模态版本，
# 让它在训练和生成时都能先插入图像/视频特征，再沿用原生 Qwen2 的语言建模逻辑。

class LlavaQwenConfig(Qwen2Config):
    # 定义 LLaVA-Qwen 的配置类型，便于 AutoConfig 正确识别这个模型家族。
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    # 在原生 Qwen2Model 外叠加 LLaVA 的视觉模块初始化与管理能力。
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        # 这里会沿着继承链进入 LlavaMetaModel.__init__，从而按配置创建 vision tower / projector 等模块。
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    # 这是最终对外使用的多模态因果语言模型：底层主干是 Qwen2，上层多了图像/视频输入能力。
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        # 先初始化原生 Qwen2ForCausalLM 的语言模型结构。
        Qwen2ForCausalLM.__init__(self, config)
        # 把配置类型改成 llava_qwen，后续加载/保存时会按这个模型族处理。
        config.model_type = "llava_qwen"
        # 这里显式清空 rope_scaling，避免和外部配置冲突。
        config.rope_scaling = None

        # 用带视觉模块的 LlavaQwenModel 替换原始的 self.model。
        self.model = LlavaQwenModel(config)
        # 重新建立输出词表映射层，保持和当前配置的 hidden_size / vocab_size 对齐。
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        # 统一返回真正持有视觉模块和语言主干的底层模型对象。
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            # 训练时如果还没有现成的 embedding，就先把文本里的 <image> 占位符替换成真正的视觉 embedding。
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            # DPO 训练分支只需要 logits 和 labels，不走 Hugging Face 默认的 loss 打包逻辑。
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            # 把隐藏状态映射成词表 logits，交给上层 DPO 代码自己计算偏好损失。
            logits = self.lm_head(hidden_states)
            return logits, labels
        else:
            # 常规 SFT / 推理前向仍然复用 Qwen2ForCausalLM 的标准实现。
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            # 多模态生成时，先把图像/视频特征拼到文本 embedding 里，再调用父类 generate。
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            # 纯文本生成则直接走普通 token embedding。
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # 在自回归生成的每一步，把图像和图像尺寸继续挂到 inputs 上，确保缓存解码时不会丢失视觉上下文。
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


# 把自定义配置和模型注册到 Hugging Face 的自动加载体系里，这样可以通过 AutoConfig / AutoModel 直接识别 llava_qwen。
AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
