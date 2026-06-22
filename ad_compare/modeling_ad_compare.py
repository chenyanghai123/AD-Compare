"""
AD-Compare Qwen3-VL-8B modeling

继承自 transformers 4.57+ 的 Qwen3VLForConditionalGeneration。
实现 ComparisonEncoder（CE）模块，适配 Qwen3-VL：
- ViT hidden_size: 1152
- LLM hidden_size: 4096
- ViT 输出含 deepstack_feature_lists
- get_rope_index 不再依赖 second_per_grid_ts
- forward 用 masked_scatter 注入 image_embeds

CE 注入策略：
1. ViT.forward 在 merger 之前切分 hidden_states 喂给 CE，得到 [B, 100, 1152] 的对比特征
2. CE 把 1152 投影到 4096，cat 到每张图的 image_embeds 末尾
3. deepstack_feature_lists 也按图切分，并在每张图末尾 padding 100 个零向量（CE token 不参与 deepstack）
4. get_rope_index 在每张图后插入 100 个 compare token 的位置编码（共享 t/h/w）
5. Processor 把 image_token 替换为 (num_image_tokens + compare_token_size) 个 placeholder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union

from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLVisionMLP,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLVisionConfig,
    Qwen3VLTextConfig,
)

# Configuration

class AdCompareQwen3VLVisionConfig(Qwen3VLVisionConfig):
    """Qwen3-VL ViT config + CE token 数。"""
    model_type = "ad_compare_qwen3_vision"

    def __init__(self, compare_token_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.compare_token_size = compare_token_size


class AdCompareQwen3VLConfig(Qwen3VLConfig):
    model_type = "ad_compare_qwen3"
    sub_configs = {"vision_config": AdCompareQwen3VLVisionConfig, "text_config": Qwen3VLTextConfig}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保 vision_config 上有 compare_token_size 字段（兼容 from_pretrained）
        if not hasattr(self.vision_config, "compare_token_size"):
            self.vision_config.compare_token_size = 100
        self.architectures = ["AdCompareQwen3VLForConditionalGeneration"]
        self.sequence_compare = True


# Comparison Encoder

class OptimizedCrossAttention(nn.Module):
    """双向交叉注意力模块；config.hidden_size=1152, num_heads=16。"""

    def __init__(self, config, is_cross_attention: bool = True):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = False
        self.is_cross_attention = is_cross_attention

        if is_cross_attention:
            self.q_proj = nn.Linear(self.dim, self.dim, bias=True)
            self.kv = nn.Linear(self.dim, self.dim * 2, bias=True)
        else:
            self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        orig_2d = False
        if query_states.dim() == 2:
            query_states = query_states.unsqueeze(0)
            orig_2d = True
        batch_size, seq_len_q, _ = query_states.shape

        if self.is_cross_attention and key_value_states is not None:
            if key_value_states.dim() == 2:
                key_value_states = key_value_states.unsqueeze(0)
            q = self.q_proj(query_states)
            kv = self.kv(key_value_states)
            seq_len_kv = kv.shape[1]
            k, v = kv.reshape(batch_size, seq_len_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
            q = q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            if key_value_states is None:
                key_value_states = query_states
            qkv = self.qkv(query_states)
            q, k, v = qkv.reshape(batch_size, seq_len_q, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        # CE 内部固定走 sdpa（避免 FA2 对 Qwen3-VL 的差异化 cu_seqlens 处理）
        if attn_impl not in ("sdpa", "eager"):
            attn_impl = "sdpa"
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[attn_impl]

        attn_output, _ = attention_interface(
            self,
            q, k, v,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=self.is_causal,
            **kwargs,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len_q, self.dim)
        attn_output = self.proj(attn_output)
        if orig_2d:
            attn_output = attn_output.squeeze(0)
        return attn_output.contiguous()


class AdCompareQwen3CompareVisualEncoder(nn.Module):
    """
    Comparison Encoder：
    - encoder: 双向 cross attention（previous<->current）+ 2x SwiGLU MLP
    - decoder: 100 个 learnable query 通过 cross attention 提取对比特征
    - compare_projector: hidden_size(1152) -> out_hidden_size(4096)
    """

    def __init__(self, config: AdCompareQwen3VLVisionConfig):
        super().__init__()
        self.config = config
        self.sequence_compare = getattr(config, "sequence_compare", True)
        self.hidden_size = config.hidden_size  # 1152
        self.token_size = getattr(config, "compare_token_size", 100)

        # Encoder
        self.encoder_cross_attn1 = OptimizedCrossAttention(config, is_cross_attention=True)
        self.encoder_cross_attn2 = OptimizedCrossAttention(config, is_cross_attention=True)
        self.encoder_norm1 = Qwen3VLTextRMSNorm(self.hidden_size, eps=1e-6)
        self.encoder_norm2 = Qwen3VLTextRMSNorm(self.hidden_size, eps=1e-6)
        self.encoder_norm3 = Qwen3VLTextRMSNorm(self.hidden_size, eps=1e-6)
        self.encoder_norm4 = Qwen3VLTextRMSNorm(self.hidden_size, eps=1e-6)
        self.encoder_mlp1 = Qwen3VLVisionMLP(config)
        self.encoder_mlp2 = Qwen3VLVisionMLP(config)

        # Decoder
        self.query_embeddings = nn.Parameter(torch.empty(self.token_size, self.hidden_size))
        self.decoder_cross_attn = OptimizedCrossAttention(config, is_cross_attention=True)
        self.decoder_norm1 = Qwen3VLTextRMSNorm(self.hidden_size, eps=1e-6)
        self.decoder_norm2 = Qwen3VLTextRMSNorm(self.hidden_size, eps=1e-6)
        self.decoder_mlp = Qwen3VLVisionMLP(config)

        # 投影到 LLM 输入维度（Qwen3-VL 是 4096）
        self.compare_projector = nn.Linear(config.hidden_size, config.out_hidden_size)

    def init_query_embeddings(self):
        nn.init.normal_(self.query_embeddings, mean=0.0, std=0.02)

    def forward(self, images_hidden_states: list) -> torch.Tensor:
        """
        Args:
            images_hidden_states: list of [seq_len_i, hidden_size=1152]
        Returns:
            [num_images, token_size=100, out_hidden_size=4096]
        """
        if not images_hidden_states:
            return torch.empty(0, self.token_size, self.config.out_hidden_size)

        seq_lengths = [s.size(0) for s in images_hidden_states]
        max_seq_len = max(seq_lengths)
        batch_size = len(images_hidden_states)
        device = images_hidden_states[0].device
        dtype = images_hidden_states[0].dtype

        padded_states = []
        attention_masks = []
        for state in images_hidden_states:
            pad_len = max_seq_len - state.size(0)
            if pad_len > 0:
                padded_state = F.pad(state, (0, 0, 0, pad_len), mode="constant", value=0)
                attn = torch.ones(max_seq_len, dtype=torch.bool, device=device)
                attn[state.size(0):] = False
            else:
                padded_state = state
                attn = torch.ones(max_seq_len, dtype=torch.bool, device=device)
            padded_states.append(padded_state)
            attention_masks.append(attn)

        batched_states = torch.stack(padded_states)
        attention_masks = torch.stack(attention_masks)

        previous_states = torch.roll(batched_states, shifts=1, dims=0)
        previous_masks = torch.roll(attention_masks, shifts=1, dims=0)
        if previous_states.size(0) > 1 and self.sequence_compare:
            previous_states[0] = previous_states[1]
            previous_masks[0] = previous_masks[1]

        encoded_features = self._encoder_forward(
            batched_states, previous_states, attention_masks, previous_masks
        )

        batch_queries = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)
        compare_visual_embeds = self._decoder_forward(
            batch_queries,
            encoded_features,
            torch.ones(batch_size, self.token_size, dtype=torch.bool, device=device),
            attention_masks,
        )

        token_size = compare_visual_embeds.size(1)
        flattened = compare_visual_embeds.reshape(-1, compare_visual_embeds.size(-1))
        merged = self.compare_projector(flattened)
        compare_visual_embeds = merged.view(batch_size, token_size, -1)
        return compare_visual_embeds  # [B, 100, 4096]

    def _encoder_forward(self, current_features, previous_features, current_mask=None, previous_mask=None):
        # previous attend to current
        residual = previous_features
        previous_normed = self.encoder_norm1(previous_features)
        current_normed1 = self.encoder_norm1(current_features)
        cross1 = self.encoder_cross_attn1(
            query_states=previous_normed,
            key_value_states=current_normed1,
            attention_mask=current_mask.unsqueeze(1).unsqueeze(2) if current_mask is not None else None,
        )
        previous_features = residual + cross1
        residual = previous_features
        mlp1 = self.encoder_mlp1(self.encoder_norm2(previous_features))
        previous_features = residual + mlp1

        # current attend to previous
        residual = current_features
        current_normed2 = self.encoder_norm3(current_features)
        previous_normed2 = self.encoder_norm3(previous_features)
        cross2 = self.encoder_cross_attn2(
            query_states=current_normed2,
            key_value_states=previous_normed2,
            attention_mask=previous_mask.unsqueeze(1).unsqueeze(2) if previous_mask is not None else None,
        )
        current_features = residual + cross2
        residual = current_features
        mlp2 = self.encoder_mlp2(self.encoder_norm4(current_features))
        # 残差连接使用减法（与 CE 原始设计一致）
        current_features = residual - mlp2
        return current_features

    def _decoder_forward(self, queries, encoded_features, query_mask=None, encoded_mask=None):
        residual = queries
        queries_normed = self.decoder_norm1(queries)
        encoded_normed = self.decoder_norm1(encoded_features)
        cross = self.decoder_cross_attn(
            query_states=queries_normed,
            key_value_states=encoded_normed,
            attention_mask=encoded_mask.unsqueeze(1).unsqueeze(2) if encoded_mask is not None else None,
        )
        queries = residual + cross
        residual = queries
        mlp = self.decoder_mlp(self.decoder_norm2(queries))
        queries = residual + mlp
        return queries


# Vision Model

class AdCompareQwen3VLVisionModel(Qwen3VLVisionModel):
    config: AdCompareQwen3VLVisionConfig

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.compare_visual_encoder = AdCompareQwen3CompareVisualEncoder(config)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        """
        Returns:
            (image_embeds, deepstack_feature_lists, compare_visual_embeds)
            注意比 base 多了 compare_visual_embeds —— 直接返回 [num_images, 100, 4096]
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        # ========== CE 关键修改：在 merger 之前喂给 CE ==========
        # hidden_states 当前 shape = [total_patches, 1152]
        split_sizes = grid_thw.prod(-1).tolist()
        splited_hidden_states_before_merger = list(torch.split(hidden_states, split_sizes))
        compare_visual_embeds = self.compare_visual_encoder(
            splited_hidden_states_before_merger
        )  # [num_images, 100, 4096]

        hidden_states = self.merger(hidden_states)  # [total_patches/spatial_merge_unit, 4096]

        return hidden_states, deepstack_feature_lists, compare_visual_embeds


# Top-level Model

class AdCompareQwen3VLModel(Qwen3VLModel):
    config: AdCompareQwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock", "AdCompareQwen3CompareVisualEncoder"]

    def __init__(self, config):
        super().__init__(config)
        # 替换 visual 为带 CE 的版本
        self.visual = AdCompareQwen3VLVisionModel._from_config(config.vision_config)
        self.compare_token_size = getattr(config.vision_config, "compare_token_size", 100)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        """
        与 base Qwen3VLModel.get_image_features 兼容的签名，但额外把 CE 的 100 个 token
        cat 到每张图末尾，并对 deepstack 特征做 zero-padding 以保持长度一致。

        Returns:
            image_embeds: tuple of [tokens_per_image_with_ce, 4096]
            deepstack_image_embeds: list of [total_visual_tokens_with_ce, 4096]
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds_flat, deepstack_feature_lists, compare_visual_embeds = self.visual(
            pixel_values, grid_thw=image_grid_thw
        )

        spatial_merge_unit = self.visual.spatial_merge_size ** 2
        # 每张图在 merger 之后的 token 数
        split_sizes = (image_grid_thw.prod(-1) // spatial_merge_unit).tolist()
        image_embeds_per_img = list(torch.split(image_embeds_flat, split_sizes))
        deepstack_per_layer = [
            list(torch.split(feat, split_sizes)) for feat in deepstack_feature_lists
        ]

        num_images = len(image_embeds_per_img)
        compare_token_size = self.compare_token_size

        enhanced_image_embeds = []
        for i in range(num_images):
            embeds = image_embeds_per_img[i]
            ce = compare_visual_embeds[i].to(device=embeds.device, dtype=embeds.dtype)
            enhanced = torch.cat([embeds, ce], dim=0)
            enhanced_image_embeds.append(enhanced)

        # 对 deepstack 特征做 zero-padding：CE token 不参与 deepstack 注入
        enhanced_deepstack = []
        for layer_idx in range(len(deepstack_per_layer)):
            per_img_padded = []
            for i in range(num_images):
                feat = deepstack_per_layer[layer_idx][i]
                pad = torch.zeros(
                    compare_token_size, feat.size(-1),
                    device=feat.device, dtype=feat.dtype,
                )
                per_img_padded.append(torch.cat([feat, pad], dim=0))
            enhanced_deepstack.append(torch.cat(per_img_padded, dim=0))

        return tuple(enhanced_image_embeds), enhanced_deepstack

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """重写 Qwen3-VL 的 get_rope_index：在每张图后插入 100 个 compare token 的位置编码（共享 t/h/w）。"""
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        compare_token_size = self.compare_token_size

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1],
                dtype=input_ids.dtype, device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, sample_ids in enumerate(total_input_ids):
                sample_ids = sample_ids[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(sample_ids == vision_start_token_id).squeeze(1)
                vision_tokens = sample_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = sample_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    is_image = ed_image < ed_video
                    if is_image:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                    if is_image and compare_token_size > 0:
                        # 100 个 compare token 共享最后一个图像 token 的 t/h/w
                        compare_t_index = t_index[-1].repeat(compare_token_size)
                        compare_h_index = compare_t_index
                        compare_w_index = compare_t_index
                        llm_pos_ids_list.append(
                            torch.stack([compare_t_index, compare_h_index, compare_w_index]) + text_len + st_idx
                        )
                        st = st + compare_token_size

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas


# Generation Model

class AdCompareQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    config_class = AdCompareQwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        # 替换 self.model 为带 CE 的版本
        self.model = AdCompareQwen3VLModel(config)


# Backward-compat alias（兼容旧 model_type="ad_copilot_qwen3" 的 checkpoint）

class _AdCopilotLegacyVisionConfig(AdCompareQwen3VLVisionConfig):
    model_type = "ad_copilot_qwen3_vision"


class _AdCopilotLegacyConfig(AdCompareQwen3VLConfig):
    """兼容旧 checkpoint 的 config alias，实体完全等同于 AdCompareQwen3VLConfig。"""
    model_type = "ad_copilot_qwen3"
    sub_configs = {"vision_config": _AdCopilotLegacyVisionConfig, "text_config": Qwen3VLTextConfig}


def _register():
    try:
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        from .processing_ad_compare import AdCompareQwen3VLProcessor

        AutoConfig.register("ad_compare_qwen3", AdCompareQwen3VLConfig)
        AutoModelForImageTextToText.register(
            AdCompareQwen3VLConfig, AdCompareQwen3VLForConditionalGeneration
        )
        AutoProcessor.register(AdCompareQwen3VLConfig, AdCompareQwen3VLProcessor)

        # 向后兼容旧 model_type
        AutoConfig.register("ad_copilot_qwen3", _AdCopilotLegacyConfig)
        AutoModelForImageTextToText.register(
            _AdCopilotLegacyConfig, AdCompareQwen3VLForConditionalGeneration
        )
        AutoProcessor.register(_AdCopilotLegacyConfig, AdCompareQwen3VLProcessor)
    except Exception:
        # 静默失败：训练脚本可以显式 import 模型类
        pass


_register()
