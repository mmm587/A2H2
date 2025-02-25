import math
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from Module.modeling_xlnet import XLNetPreTrainedModel
from Module.modeling_xlnet import (
    XLNetLayer,
    SequenceSummary,
)

from Module.early_fusion import KANSequenceGate
from configs import *

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=8):
        super(KANLayer, self).__init__()
        self.kernel_weights = nn.Parameter(torch.randn(kernel_size, input_dim))  # 核权重
        self.linear = nn.Linear(kernel_size, output_dim)  # 输出投影

    def forward(self, x):
        # 使用核权重进行特征变换
        kernel_features = F.relu(torch.matmul(x, self.kernel_weights.T))
        # 投影到目标维度
        output = self.linear(kernel_features)
        return output
class XLNetModel(XLNetPreTrainedModel):
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        pass

    def _reorder_cache(self, past_key_values, beam_idx):
        pass

    def __init__(self, config):
        super().__init__(config)

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.layer = nn.ModuleList([XLNetLayer(config, currentLayer+1) for currentLayer in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)

        self.AMG = KANSequenceGate(config.hidden_size)

        self.init_weights()

    def get_input_embeddings(self):
        return self.word_embedding

    def set_input_embeddings(self, new_embeddings):
        self.word_embedding = new_embeddings

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def create_mask(self, qlen, mlen):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.
        Args:
            qlen: Sequence length
            mlen: Mask length
        ::
                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]
        """
        attn_mask = torch.ones([qlen, qlen])
        mask_up = torch.triu(attn_mask, diagonal=1)
        attn_mask_pad = torch.zeros([qlen, mlen])
        ret = torch.cat([attn_mask_pad, mask_up], dim=1)
        if self.same_length:
            mask_lo = torch.tril(attn_mask, diagonal=-1)
            ret = torch.cat([ret[:, :qlen] + mask_lo, ret[:, qlen:]], dim=1)

        ret = ret.to(self.device)
        return ret

    def cache_mem(self, curr_out, prev_mem):
        # cache hidden states into memory.
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]
        if prev_mem is None:
            new_mem = curr_out[-self.mem_len:]
        else:
            new_mem = torch.cat([prev_mem, curr_out], dim=0)[-self.mem_len:]
        return new_mem.detach()

    def positional_embedding(self, pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat(
            [torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]
        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)
        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        # create relative positional encoding.
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))
        if self.attn_type == "bi":
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            raise ValueError("Unknown `attn_type` {}.".format(self.attn_type))

        if self.bi_data:
            fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.float)
            bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.float)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)
            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)
        pos_emb = pos_emb.to(self.device)
        return pos_emb

    def forward(self, input_ids, visual, acoustic, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                token_type_ids=None, input_mask=None, head_mask=None, inputs_embeds=None, use_cache=True,
                output_attentions=None, output_hidden_states=None):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = input_ids.transpose(0, 1).contiguous()
            qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")
        visual = visual.transpose(0, 1).contiguous()
        acoustic = acoustic.transpose(0, 1).contiguous()
        token_type_ids = (
            token_type_ids.transpose(0, 1).contiguous()
            if token_type_ids is not None
            else None
        )
        input_mask = (
            input_mask.transpose(0, 1).contiguous(
            ) if input_mask is not None else None
        )
        attention_mask = (
            attention_mask.transpose(0, 1).contiguous()
            if attention_mask is not None
            else None
        )
        perm_mask = (
            perm_mask.permute(1, 2, 0).contiguous(
            ) if perm_mask is not None else None
        )
        target_mapping = (
            target_mapping.permute(1, 2, 0).contiguous()
            if target_mapping is not None
            else None
        )

        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        dtype_float = self.dtype
        device = self.device

        # Attention mask
        # causal attention mask
        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError(
                "Unsupported attention type: {}".format(self.attn_type))

        # data mask: input mask & perm mask
        assert (
                input_mask is None or attention_mask is None
        ), "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatbility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            if mlen > 0:
                mems_mask = torch.zeros(
                    [data_mask.shape[0], mlen, bsz]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = torch.cat(
                    [torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1
                )
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(
                attn_mask
            )
        else:
            non_tgt_mask = None

        # Word embeddings and prepare h & g hidden states
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
            # else:  # We removed the inp_q input which was same as target mapping
            #     inp_q_ext = inp_q[:, :, None]
            #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        # Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            if mlen > 0:
                mem_pad = torch.zeros(
                    [mlen, bsz], dtype=torch.long, device=device)
                cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)
            else:
                cat_ids = token_type_ids

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = F.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(
                        0).unsqueeze(0).unsqueeze(0)
                )
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = []
        hidden_states = []

        for i, layer_module in enumerate(self.layer):
            if self.mem_len is not None and self.mem_len > 0 and use_cache is True:
                # cache new mems
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if output_hidden_states:
                hidden_states.append(
                    (output_h, output_g) if output_g is not None else output_h
                )
            # print("\n", i+1, "\t") i[0,12]
            # TODO: 且仅在 第一层 首层的 input embedding做出修改
            if i == 1:
                output_h = self.AMG(output_h, visual, acoustic)

            outputs = layer_module(
                output_h,
                output_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                seg_mat=seg_mat,
                mems=mems[i],
                target_mapping=target_mapping,
                head_mask=head_mask[i],
                output_attentions=output_attentions,

            )
            output_h, output_g = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])

        # Add last hidden state
        if output_hidden_states:
            hidden_states.append(
                (output_h, output_g) if output_g is not None else output_h
            )

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        outputs = (output.permute(1, 0, 2).contiguous(),)

        if self.mem_len is not None and self.mem_len > 0 and use_cache is True:
            outputs = outputs + (new_mems,)

        if output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(
                    h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs
                )
            else:
                hidden_states = tuple(
                    hs.permute(1, 0, 2).contiguous() for hs in hidden_states
                )
            outputs = outputs + (hidden_states,)
        if output_attentions:
            if target_mapping is not None:
                # when target_mapping is provided, there are 2-tuple of attentions
                attentions = tuple(
                    tuple(
                        att_stream.permute(2, 3, 0, 1).contiguous() for att_stream in t
                    )
                    for t in attentions
                )
            else:
                attentions = tuple(
                    t.permute(2, 3, 0, 1).contiguous() for t in attentions
                )
            outputs = outputs + (attentions,)

        return outputs  # outputs, (new_mems), (hidden_states), (attentions)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ImprovedMultimodalFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_dim = config.hidden_size

        # 修改特征提取层，注意输入输出维度
        self.audio_encoder = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim),  # 输入已经是TEXT_DIM维度
            nn.LayerNorm(self.text_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.visual_encoder = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim),  # 输入已经是TEXT_DIM维度
            nn.LayerNorm(self.text_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 跨模态注意力
        self.text_audio_attention = CrossModalAttention(self.text_dim)
        self.text_visual_attention = CrossModalAttention(self.text_dim)
        self.audio_visual_attention = CrossModalAttention(self.text_dim)

        # 门控融合
        self.fusion_gate1 = GatedFusion(self.text_dim)
        self.fusion_gate2 = GatedFusion(self.text_dim)

        # 残差连接和层归一化
        self.norm1 = nn.LayerNorm(self.text_dim)
        self.norm2 = nn.LayerNorm(self.text_dim)

        # 自适应权重
        self.modality_weights = nn.Parameter(torch.ones(3))

    def forward(self, text_features, visual_features, audio_features):
        # 注意：此时输入的特征已经经过了前面的维度对齐

        # 进一步特征编码
        audio_aligned = self.audio_encoder(audio_features)
        visual_aligned = self.visual_encoder(visual_features)

        # 跨模态注意力
        text_audio = self.text_audio_attention(text_features, audio_aligned)
        text_visual = self.text_visual_attention(text_features, visual_aligned)
        audio_visual = self.audio_visual_attention(audio_aligned, visual_aligned)

        # 动态权重
        weights = F.softmax(self.modality_weights, dim=0)

        # 多阶段融合
        fusion1 = self.fusion_gate1(text_features, text_audio)
        fusion1 = self.norm1(fusion1)

        fusion2 = self.fusion_gate2(fusion1, text_visual)
        fusion2 = self.norm2(fusion2)

        # 加权融合
        final_output = (weights[0] * fusion2 +
                        weights[1] * text_audio +
                        weights[2] * audio_visual)

        return final_output + text_features  # 残差连接


class FineTune(nn.Module):
    def __init__(self, config):
        super(FineTune, self).__init__()
        # 保留原有的基础组件用于初始特征对齐
        self.audio_conv = nn.Conv1d(ACOUSTIC_DIM, TEXT_DIM, kernel_size=1, padding=0, bias=False)
        self.vision_conv = nn.Conv1d(VISUAL_DIM, TEXT_DIM, kernel_size=1, padding=0, bias=False)
        self.positional_encoding = PositionalEncoding(TEXT_DIM, 50)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(TEXT_DIM, 12)
        self.a_transformer_encoder = nn.TransformerEncoder(encoder_layer, 4)
        self.v_transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)

        # 规范化层
        self.LayerNorm = nn.LayerNorm(TEXT_DIM)
        self.dropout = nn.Dropout(0.5)

        # 可以选择保留KAN或使用新的融合机制
        self.use_kan = True  # 通过这个标志控制使用哪种融合方式
        if self.use_kan:
            self.KANLayer1 = KANLayer(TEXT_DIM * 2, TEXT_DIM)
            self.KANLayer2 = KANLayer(TEXT_DIM, TEXT_DIM)
            self.KANLayer3 = KANLayer(TEXT_DIM, TEXT_DIM)
        else:
            self.modal_fusion = ImprovedMultimodalFusion(config)

    def forward(self, hidden_states, visual, acoustic):
        # 1. 初始特征对齐
        audio_aligned = self.audio_conv(acoustic.permute(0, 2, 1)).permute(0, 2, 1)
        vision_aligned = self.vision_conv(visual.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. 位置编码和Transformer处理
        audio_aligned = self.positional_encoding(audio_aligned.permute(1, 0, 2))
        vision_aligned = self.positional_encoding(vision_aligned.permute(1, 0, 2))

        audio_encoded = self.a_transformer_encoder(audio_aligned).permute(1, 0, 2)
        vision_encoded = self.v_transformer_encoder(vision_aligned).permute(1, 0, 2)

        if self.use_kan:
            # 使用原有的KAN融合方式
            fused_features = torch.cat((audio_encoded, hidden_states), dim=2)
            hidden_states_new = self.dropout(F.relu(self.KANLayer1(fused_features)))
            hidden_states_new = self.dropout(F.relu(self.KANLayer2(hidden_states_new)))
            hidden_states_new = self.dropout(F.relu(self.KANLayer3(hidden_states_new)))
        else:
            # 使用新的融合机制，注意此时特征维度已经对齐
            hidden_states_new = self.modal_fusion(hidden_states, vision_encoded, audio_encoded)

        return hidden_states + hidden_states_new


    # 辅助组件定义
class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        return attn @ v


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1)))
        return gate * x1 + (1 - gate) * x2


# class FineTune(nn.Module):            # 这里的FineTune(nn.Module)是加入kan后的
#     def __init__(self, config):
#         super(FineTune, self).__init__()
#         self.audio_conv = nn.Conv1d(ACOUSTIC_DIM, TEXT_DIM, kernel_size=1, padding=0, bias=False)
#         self.vision_conv = nn.Conv1d(VISUAL_DIM, TEXT_DIM, kernel_size=1, padding=0, bias=False)
#         self.self_attention = nn.MultiheadAttention(TEXT_DIM, 8)
#         self.cross_attention = nn.MultiheadAttention(TEXT_DIM, 8)
#         self.positional_encoding = PositionalEncoding(TEXT_DIM, 50)
#         encoder_layer = nn.TransformerEncoderLayer(TEXT_DIM, 12)
#
#         self.a_transformer_encoder = nn.TransformerEncoder(encoder_layer, 4)
#         self.v_transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)
#         self.LayerNorm = nn.LayerNorm(TEXT_DIM)
#         self.dropout = nn.Dropout(0.5)
#
#         # 使用 KAN 层替换线性融合
#         self.KANLayer1 = KANLayer(TEXT_DIM * 2, TEXT_DIM)
#         self.KANLayer2 = KANLayer(TEXT_DIM, TEXT_DIM)
#         self.KANLayer3 = KANLayer(TEXT_DIM, TEXT_DIM)
#
#     def forward(self, hidden_states, visual, acoustic):
#         # 模态对齐
#         audio_aligned = self.audio_conv(acoustic.permute(0, 2, 1)).permute(0, 2, 1)
#         vision_aligned = self.vision_conv(visual.permute(0, 2, 1)).permute(0, 2, 1)
#         audio_aligned = audio_aligned.permute(1, 0, 2)
#         vision_aligned = vision_aligned.permute(1, 0, 2)
#
#         # 模态编码
#         audio_aligned = self.positional_encoding(audio_aligned)
#         audio_aligned = self.a_transformer_encoder(audio_aligned)
#         vision_aligned = self.positional_encoding(vision_aligned)
#         vision_aligned = self.v_transformer_encoder(vision_aligned)
#
#         audio_aligned = audio_aligned.permute(1, 0, 2)
#         vision_aligned = vision_aligned.permute(1, 0, 2)
#
#         # 自注意力
#         audio_output, _ = self.self_attention(audio_aligned, audio_aligned, audio_aligned)
#         audio_output = self.dropout(audio_output)
#         audio_output = self.LayerNorm(audio_output + audio_aligned)
#
#         vision_output, _ = self.self_attention(vision_aligned, vision_aligned, vision_aligned)
#         vision_output = self.dropout(vision_output)
#         vision_output = self.LayerNorm(vision_output + vision_aligned)
#
#         # 跨模态交叉注意力机制
#         fused_output_audio, _ = self.cross_attention(audio_output, vision_output, vision_output)
#         fused_output_audio = self.dropout(fused_output_audio)
#         fused_output_audio = self.LayerNorm(fused_output_audio + audio_output)
#
#         # 使用 KAN 层进行融合
#         fused_features = torch.cat((fused_output_audio, hidden_states), dim=2)
#         hidden_states_new = self.dropout(F.relu(self.KANLayer1(fused_features)))
#         hidden_states_new = self.dropout(F.relu(self.KANLayer2(hidden_states_new)))
#         hidden_states_new = self.dropout(F.relu(self.KANLayer3(hidden_states_new)))
#
#         # 残差连接
#         hidden_states_new = hidden_states + hidden_states_new
#
#         return hidden_states_new


# class FineTune(nn.Module):  #原始的
#     def __init__(self, config):
#         super(FineTune, self).__init__()
#         self.audio_conv = nn.Conv1d(ACOUSTIC_DIM, TEXT_DIM, kernel_size=1, padding=0, bias=False)
#         self.vision_conv = nn.Conv1d(VISUAL_DIM, TEXT_DIM, kernel_size=1, padding=0, bias=False)
#         self.self_attention = nn.MultiheadAttention(TEXT_DIM, 8)
#         self.cross_attention = nn.MultiheadAttention(TEXT_DIM, 8)
#         self.positional_encoding = PositionalEncoding(TEXT_DIM, 50)
#         encoder_layer = nn.TransformerEncoderLayer(TEXT_DIM, 12)
#
#         self.a_transformer_encoder = nn.TransformerEncoder(encoder_layer, 4)
#         self.v_transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)
#         self.LayerNorm = nn.LayerNorm(TEXT_DIM)
#         self.dropout = nn.Dropout(0.5)
#         # Projection layers 预测层
#         self.Linear = nn.Linear(TEXT_DIM * 2, TEXT_DIM)
#         self.proj1 = nn.Linear(TEXT_DIM, TEXT_DIM)
#         self.proj2 = nn.Linear(TEXT_DIM, TEXT_DIM)
#
#     def forward(self, hidden_states, visual, acoustic):#         h_s=ft
#         # 1. 先换维度 再对齐
#         audio_aligned = self.audio_conv(acoustic.permute(0, 2, 1)).permute(0, 2, 1)
#         vision_aligned = self.vision_conv(visual.permute(0, 2, 1)).permute(0, 2, 1)
#         audio_aligned = audio_aligned.permute(1, 0, 2)
#         vision_aligned = vision_aligned.permute(1, 0, 2)
#
#         audio_aligned = self.positional_encoding(audio_aligned)
#         audio_aligned = self.a_transformer_encoder(audio_aligned)
#         vision_aligned = self.positional_encoding(vision_aligned)
#         vision_aligned = self.v_transformer_encoder(vision_aligned)
#
#         audio_aligned = audio_aligned.permute(1, 0, 2)   #            fa
#         vision_aligned = vision_aligned.permute(1, 0, 2)   #                 fv
#         # # 2. AV 自注意力
#         audio_output, _ = self.self_attention(audio_aligned, audio_aligned, audio_aligned)
#         audio_output = self.dropout(audio_output)
#         audio_output = self.LayerNorm(audio_output + audio_aligned)#        fa过自注意力
#
#         vision_output, _ = self.self_attention(vision_aligned, vision_aligned, vision_aligned)
#         vision_output = self.dropout(vision_output)
#         vision_output = self.LayerNorm(vision_output + vision_aligned)#        fv过自注意力
#         # 3. 跨模态交叉注意力机制
#         fused_output_audio, _ = self.cross_attention(audio_output, vision_output, vision_output)#（q.k.v）
#         fused_output_audio = self.dropout(fused_output_audio)
#         fused_output_audio = self.LayerNorm(fused_output_audio + audio_output)
#         fused_features = torch.cat((fused_output_audio, hidden_states), dim=2)
#
#         hidden_states_new = self.dropout(F.relu(self.Linear(fused_features)))
#         hidden_states_new = self.dropout(F.relu(self.proj1(hidden_states_new)))
#         hidden_states_new = self.dropout(F.relu(self.proj2(hidden_states_new)))
#         hidden_states_new = hidden_states + hidden_states_new
#
#         return hidden_states_new



class XLNetForSequenceClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.Finetune = FineTune(config)
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.init_weights()
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, input_ids, visual, acoustic, attention_mask=None, mems=None, perm_mask=None,
                target_mapping=None,
                token_type_ids=None, input_mask=None, head_mask=None, inputs_embeds=None, use_cache=True,
                output_attentions=None, output_hidden_states=None):
        # 调用 transformer 主体
        transformer_outputs = self.transformer(input_ids, visual, acoustic, attention_mask=attention_mask,
                                                   mems=mems,
                                                   perm_mask=perm_mask, target_mapping=target_mapping,
                                                   token_type_ids=token_type_ids, input_mask=input_mask,
                                                   head_mask=head_mask, inputs_embeds=inputs_embeds,
                                                   use_cache=use_cache,
                                                   output_attentions=output_attentions,
                                                   output_hidden_states=output_hidden_states)
        hidden_states = transformer_outputs[0]
        if args.Use_LFusion:
            hidden_states = self.Finetune(hidden_states, visual, acoustic)
        # 分类预测
        output = self.sequence_summary(hidden_states)
        outputs = self.logits_proj(output)
        return outputs

# class XLNetForSequenceClassification(XLNetPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.Finetune = FineTune(config)
#         self.transformer = XLNetModel(config)
#         self.sequence_summary = SequenceSummary(config)
#         self.init_weights()
#         self.logits_proj = nn.Linear(config.d_model, config.num_labels)
#
#     def forward(self, input_ids, visual, acoustic, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
#                 token_type_ids=None, input_mask=None, head_mask=None, inputs_embeds=None, use_cache=True,
#                 output_attentions=None, output_hidden_states=None):
#         transformer_outputs = self.transformer(input_ids, visual, acoustic, attention_mask=attention_mask, mems=mems,
#                                                perm_mask=perm_mask, target_mapping=target_mapping,
#                                                token_type_ids=token_type_ids, input_mask=input_mask,
#                                                head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache,
#                                                output_attentions=output_attentions,
#                                                output_hidden_states=output_hidden_states)
#         hidden_states = transformer_outputs[0]
#         if args.Use_LFusion:
#             hidden_states = self.Finetune(hidden_states, visual, acoustic)
#         output = self.sequence_summary(hidden_states)
#         outputs = self.logits_proj(output)
#         return outputs

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        pass

    def _reorder_cache(self, past_key_values, beam_idx):
        pass
