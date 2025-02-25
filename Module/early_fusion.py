import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import *


class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=8):
        super(KANLayer, self).__init__()
        self.kernel_weights = nn.Parameter(torch.randn(kernel_size, input_dim))
        self.linear = nn.Linear(kernel_size, output_dim)

    def forward(self, x):
        kernel_features = F.relu(torch.matmul(x, self.kernel_weights.T))
        output = self.linear(kernel_features)
        return output


class GRUFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUFusion, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.norm(gru_out)


class TransformerFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TransformerFusion, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # 转换维度为 [seq_len, batch, dim]
        x = x.transpose(0, 1)
        transformer_out = self.transformer(x)
        # 转换回 [batch, seq_len, dim]
        return self.norm(transformer_out.transpose(0, 1))


class TCNFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TCNFusion, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # 转换维度为 [batch, dim, seq_len]
        x = x.transpose(1, 2)
        tcn_out = self.tcn(x)
        # 转换回 [batch, seq_len, dim]
        return self.norm(tcn_out.transpose(1, 2))

class BiLSTMFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiLSTMFusion, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        return self.norm(lstm_out)

# #下面这个是使用kan+transformer+tcn版本
class KANSequenceGate(nn.Module):
    def __init__(self, hidden_size):
        super(KANSequenceGate, self).__init__()

        # 1. KAN特征变换层保持不变
        self.W_hv = KANLayer(VISUAL_DIM + TEXT_DIM, TEXT_DIM)
        self.W_ha = KANLayer(ACOUSTIC_DIM + TEXT_DIM, TEXT_DIM)
        self.W_v = KANLayer(VISUAL_DIM, TEXT_DIM)
        self.W_a = KANLayer(ACOUSTIC_DIM, TEXT_DIM)

        # 2. Transformer序列处理
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TEXT_DIM,
            nhead=8,
            dim_feedforward=TEXT_DIM * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # 3. TCN序列处理
        self.tcn = nn.Sequential(
            nn.Conv1d(TEXT_DIM, TEXT_DIM // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(TEXT_DIM // 2, TEXT_DIM, kernel_size=3, padding=1)
        )

        # 4. 融合层
        self.fusion_layer = nn.Linear(TEXT_DIM * 2, TEXT_DIM)

        # 5. 注意力和归一化层
        self.attention = nn.MultiheadAttention(TEXT_DIM, num_heads=8, dropout=0.1)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(args.drop)

        # 6. 动态缩放因子
        self.scaling_factor = args.scaling_factor
        self.dynamic_scale = nn.Parameter(torch.ones(1))

    def forward(self, text_embedding, visual, acoustic):
        # 1. KAN特征变换
        weight_v = F.gelu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.gelu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))
        transformed_v = self.W_v(visual)
        transformed_a = self.W_a(acoustic)
        weighted_v = weight_v * transformed_v
        weighted_a = weight_a * transformed_a

        # 2. 初始特征组合
        combined_features = weighted_v + weighted_a

        # 3. Transformer处理
        trans_input = combined_features.transpose(0, 1)
        trans_output = self.transformer(trans_input)
        trans_output = trans_output.transpose(0, 1)

        # 4. TCN处理
        tcn_input = combined_features.transpose(1, 2)
        tcn_output = self.tcn(tcn_input)
        tcn_output = tcn_output.transpose(1, 2)

        # 5. 特征融合
        fused_features = torch.cat([trans_output, tcn_output], dim=-1)
        fused_features = self.fusion_layer(fused_features)

        # 6. 自注意力处理
        fused_features = fused_features.transpose(0, 1)
        attn_output, _ = self.attention(fused_features, fused_features, fused_features)
        fused_features = attn_output.transpose(0, 1)

        # 7. 融合机制
        if args.Use_EFusion:
            em_norm = text_embedding.norm(2, dim=-1)
            fm_norm = fused_features.norm(2, dim=-1)
            scale = self.dynamic_scale * self.scaling_factor
            thresh_hold = torch.pow(em_norm / (fm_norm + 1e-6), 1 / 2) * scale
            fusion_weight = torch.sigmoid(thresh_hold).unsqueeze(dim=-1)
            output = self.LayerNorm(fused_features * fusion_weight + text_embedding)
        elif args.Use_Mag:
            em_norm = text_embedding.norm(2, dim=-1)
            fm_norm = fused_features.norm(2, dim=-1)
            thresh_hold = (em_norm / (fm_norm + 1e-6)) * self.dynamic_scale
            alpha = torch.sigmoid(thresh_hold).unsqueeze(dim=-1)
            output = self.LayerNorm(alpha * fused_features + text_embedding)
        else:
            output = text_embedding

        return self.dropout(output)
# 以下是目前确定版本修改KANBiLSTMGate类来使用不同的序列模型
# class KANSequenceGate(nn.Module):
#     def __init__(self, hidden_size, sequence_model='transformer'):#bilstm\gru\transformer\tcn 选择模型
#         super(KANSequenceGate, self).__init__()
#
#         # KAN层保持不变
#         self.W_hv = KANLayer(VISUAL_DIM + TEXT_DIM, TEXT_DIM)
#         self.W_ha = KANLayer(ACOUSTIC_DIM + TEXT_DIM, TEXT_DIM)
#         self.W_v = KANLayer(VISUAL_DIM, TEXT_DIM)
#         self.W_a = KANLayer(ACOUSTIC_DIM, TEXT_DIM)
#
#         # 选择序列模型
#         if sequence_model == 'bilstm':
#             self.sequence_fusion = BiLSTMFusion(TEXT_DIM, TEXT_DIM)
#         elif sequence_model == 'gru':
#             self.sequence_fusion = GRUFusion(TEXT_DIM, TEXT_DIM)
#         elif sequence_model == 'transformer':
#             self.sequence_fusion = TransformerFusion(TEXT_DIM, TEXT_DIM)
#         elif sequence_model == 'tcn':
#             self.sequence_fusion = TCNFusion(TEXT_DIM, TEXT_DIM)
#         else:
#             raise ValueError(f"Unknown sequence model: {sequence_model}")
#
#         # 其他层保持不变
#         self.attention = nn.MultiheadAttention(TEXT_DIM, num_heads=8, dropout=0.1)
#         self.LayerNorm = nn.LayerNorm(hidden_size)
#         self.dropout = nn.Dropout(args.drop)
#         self.scaling_factor = args.scaling_factor
#         self.dynamic_scale = nn.Parameter(torch.ones(1))
#
#     def fusion_mechanism(self, text_embedding, fused_features):
#         # 保持原有的融合机制不变
#         if args.Use_EFusion:
#             em_norm = text_embedding.norm(2, dim=-1)
#             fm_norm = fused_features.norm(2, dim=-1)
#             scale = self.dynamic_scale * self.scaling_factor
#             thresh_hold = torch.pow(em_norm / (fm_norm + 1e-6), 1 / 2) * scale
#             fusion_weight = torch.sigmoid(thresh_hold).unsqueeze(dim=-1)
#             return self.LayerNorm(fused_features * fusion_weight + text_embedding)
#         elif args.Use_Mag:
#             em_norm = text_embedding.norm(2, dim=-1)
#             fm_norm = fused_features.norm(2, dim=-1)
#             thresh_hold = (em_norm / (fm_norm + 1e-6)) * self.dynamic_scale
#             alpha = torch.sigmoid(thresh_hold).unsqueeze(dim=-1)
#             enhanced_features = fused_features * alpha
#             return self.LayerNorm(enhanced_features + text_embedding)
#         else:
#             return text_embedding
#
#     def forward(self, text_embedding, visual, acoustic):
#         # KAN特征转换部分保持不变
#         weight_v = F.gelu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
#         weight_a = F.gelu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))
#         transformed_v = self.W_v(visual)
#         transformed_a = self.W_a(acoustic)
#         weighted_v = weight_v * transformed_v
#         weighted_a = weight_a * transformed_a
#         combined_features = weighted_v + weighted_a
#
#         # 使用选择的序列模型
#         sequence_features = self.sequence_fusion(combined_features)
#
#         # 自注意力处理
#         sequence_features = sequence_features.transpose(0, 1)
#         attn_output, _ = self.attention(sequence_features, sequence_features, sequence_features)
#         fused_features = attn_output.transpose(0, 1)
#
#         # 融合机制和输出
#         embedding_output = self.fusion_mechanism(text_embedding, fused_features)
#         output = self.dropout(embedding_output)
#
#         return output



# class KANBiLSTMGate(nn.Module):
#     def __init__(self, hidden_size):
#         super(KANBiLSTMGate, self).__init__()
#
#         # KAN层用于特征转换
#         self.W_hv = KANLayer(VISUAL_DIM + TEXT_DIM, TEXT_DIM)
#         self.W_ha = KANLayer(ACOUSTIC_DIM + TEXT_DIM, TEXT_DIM)
#         self.W_v = KANLayer(VISUAL_DIM, TEXT_DIM)
#         self.W_a = KANLayer(ACOUSTIC_DIM, TEXT_DIM)
#
#         # BiLSTM用于序列建模
#         self.bilstm_fusion = BiLSTMFusion(TEXT_DIM, TEXT_DIM)
#
#         # 注意力层
#         self.attention = nn.MultiheadAttention(TEXT_DIM, num_heads=8, dropout=0.1)
#
#         # LayerNorm和Dropout
#         self.LayerNorm = nn.LayerNorm(hidden_size)
#         self.dropout = nn.Dropout(args.drop)
#
#         # 动态缩放因子
#         self.scaling_factor = args.scaling_factor
#         self.dynamic_scale = nn.Parameter(torch.ones(1))
#
#     def fusion_mechanism(self, text_embedding, fused_features):
#         if args.Use_EFusion:
#             # 改进的EFusion
#             em_norm = text_embedding.norm(2, dim=-1)
#             fm_norm = fused_features.norm(2, dim=-1)
#
#             # 动态缩放
#             scale = self.dynamic_scale * self.scaling_factor
#             thresh_hold = torch.pow(em_norm / (fm_norm + 1e-6), 1 / 2) * scale
#
#             # Soft fusion with sigmoid
#             fusion_weight = torch.sigmoid(thresh_hold).unsqueeze(dim=-1)
#             return self.LayerNorm(fused_features * fusion_weight + text_embedding)
#
#         elif args.Use_Mag:
#             # 改进的MAG
#             em_norm = text_embedding.norm(2, dim=-1)
#             fm_norm = fused_features.norm(2, dim=-1)
#
#             # 动态阈值
#             thresh_hold = (em_norm / (fm_norm + 1e-6)) * self.dynamic_scale
#             alpha = torch.sigmoid(thresh_hold).unsqueeze(dim=-1)
#
#             # 特征增强
#             enhanced_features = fused_features * alpha
#             return self.LayerNorm(enhanced_features + text_embedding)
#
#         else:
#             return text_embedding
#
#     def forward(self, text_embedding, visual, acoustic):
#         batch_size = text_embedding.size(0)
#         seq_length = text_embedding.size(1)
#
#         # 1. KAN特征转换
#         # 计算视觉和声学的权重
#         weight_v = F.gelu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
#         weight_a = F.gelu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))
#
#         # 转换视觉和声学特征
#         transformed_v = self.W_v(visual)
#         transformed_a = self.W_a(acoustic)
#
#         # 加权组合
#         weighted_v = weight_v * transformed_v
#         weighted_a = weight_a * transformed_a
#
#         # 2. BiLSTM处理
#         # 合并特征
#         combined_features = weighted_v + weighted_a
#
#         # BiLSTM序列建模
#         lstm_features = self.bilstm_fusion(combined_features)
#
#         # 3. 自注意力处理
#         lstm_features = lstm_features.transpose(0, 1)  # [seq_len, batch, hidden]
#         attn_output, _ = self.attention(lstm_features, lstm_features, lstm_features)
#         fused_features = attn_output.transpose(0, 1)  # [batch, seq_len, hidden]
#
#         # 4. 应用融合机制
#         embedding_output = self.fusion_mechanism(text_embedding, fused_features)
#
#         # 5. 残差连接和归一化
#         output = self.dropout(embedding_output)
#
#         return output

#下面是上个版本
# class KANLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, kernel_size=8): # 可调整 kernel_size
#         super(KANLayer, self).__init__()
#         self.kernel_weights = nn.Parameter(torch.randn(kernel_size, input_dim))
#         self.linear = nn.Linear(kernel_size, output_dim)
#
#     def forward(self, x):
#         # 核特征映射
#         kernel_features = F.relu(torch.matmul(x, self.kernel_weights.T))
#         # 输出线性组合
#         output = self.linear(kernel_features)
#         return output
#
# class Attention_multi_gate(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention_multi_gate, self).__init__()
#
#         # 使用 KAN 层替代线性层
#         self.W_hv = KANLayer(VISUAL_DIM + TEXT_DIM, TEXT_DIM)
#         self.W_ha = KANLayer(ACOUSTIC_DIM + TEXT_DIM, TEXT_DIM)
#         self.W_v = KANLayer(VISUAL_DIM, TEXT_DIM)
#         self.W_a = KANLayer(ACOUSTIC_DIM, TEXT_DIM)
#
#         self.scaling_factor = args.scaling_factor
#         self.LayerNorm = nn.LayerNorm(hidden_size)
#         self.dropout = nn.Dropout(args.drop)
#
#     def forward(self, text_embedding, visual, acoustic):
#         eps = 1e-6
#
#         # 计算视觉和声学的权重
#         weight_v = F.gelu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
#         weight_a = F.gelu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))
#
#         # 使用 KAN 层替代直接线性组合 KAN 融合视觉和声学特征
#         h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)
#
#         # 使用 EFusion 机制
#         if args.Use_EFusion:
#             em_norm = text_embedding.norm(1, dim=-1)
#             hm_norm = h_m.norm(1, dim=-1)
#             thresh_hold = torch.pow(em_norm / (hm_norm + eps), 1 / 3) * self.scaling_factor
#             embedding_output = self.dropout(
#                 self.LayerNorm((h_m * thresh_hold.unsqueeze(dim=-1) + 1) * text_embedding)
#             )
#         elif args.Use_Mag:
#             # 使用 MAG 机制
#             em_norm = text_embedding.norm(2, dim=-1)
#             hm_norm = h_m.norm(2, dim=-1)
#
#             hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(torch.device("cuda:0"))
#             hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
#
#             thresh_hold = (em_norm / (hm_norm + eps)) * 0.5
#
#             ones = torch.ones(thresh_hold.shape, requires_grad=True).to(torch.device("cuda:0"))
#             alpha = torch.min(thresh_hold, ones).unsqueeze(dim=-1)
#             acoustic_vis_embedding = alpha * h_m
#
#             embedding_output = self.dropout(
#                 self.LayerNorm(acoustic_vis_embedding + text_embedding)
#             )
#         else:
#             # 消融实验：只使用文本嵌入
#             embedding_output = text_embedding
#
#         return embedding_output

# class Attention_multi_gate(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention_multi_gate, self).__init__()
#
#         # 隐藏层 hv:v+t-> t   ha:a+t-> t
#         self.W_hv = nn.Linear(VISUAL_DIM + TEXT_DIM, TEXT_DIM)
#         self.W_ha = nn.Linear(ACOUSTIC_DIM + TEXT_DIM, TEXT_DIM)
#         # v:v-> t  a:a-> t
#         self.W_v = nn.Linear(VISUAL_DIM, TEXT_DIM)
#         self.W_a = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
#         self.scaling_factor = args.scaling_factor
#
#         self.LayerNorm = nn.LayerNorm(hidden_size)
#         self.dropout = nn.Dropout(args.drop)
#
#     def forward(self, text_embedding, visual, acoustic): #接收文本嵌入、视觉特征和声学特征作为输入。
#         eps = 1e-6
#         weight_v = F.gelu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
#         weight_a = F.gelu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))
#         #weight_v 和 weight_a 是通过将视觉和声学特征与文本嵌入连接后，经过线性层和激活函数（GELU）计算得到的权重。
#         h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)
#         #h_m 是融合后的特征，结合了视觉和声学特征的加权结果。
#         if args.Use_EFusion:
#             em_norm = text_embedding.norm(1, dim=-1)
#             hm_norm = h_m.norm(1, dim=-1)
#             thresh_hold = torch.pow(em_norm / (hm_norm + eps), 1 / 3) * self.scaling_factor
#             embedding_output = self.dropout(
#                 self.LayerNorm((h_m * thresh_hold.unsqueeze(dim=-1) + 1) * text_embedding)
#             )#计算文本嵌入和融合特征的L1范数，得到阈值并进行缩放。生成最终的嵌入输出
#         elif args.Use_Mag:
#             em_norm = text_embedding.norm(2, dim=-1)
#             hm_norm = h_m.norm(2, dim=-1)
#
#             hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(torch.device("cuda:0"))
#             hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
#
#             thresh_hold = (em_norm / (hm_norm + eps)) * 0.5  # * 1.0
#
#             ones = torch.ones(thresh_hold.shape, requires_grad=True).to(torch.device("cuda:0"))
#
#             alpha = torch.min(thresh_hold, ones)
#             alpha = alpha.unsqueeze(dim=-1)
#
#             acoustic_vis_embedding = alpha * h_m
#
#             embedding_output = self.dropout(
#                 self.LayerNorm(acoustic_vis_embedding + text_embedding)
#             )
#         else:  # 都不使用 直接使用Word Embedding
#             embedding_output = text_embedding#                消融实验
#         return embedding_output

