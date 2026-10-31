import torch
import torch.nn as nn
import torch.nn.functional as F

# 简易自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 三个线性层，把输入映射 Q K V
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim] 一批句子，序列长度，词向量维度
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Q·K^T 点积求相似度
        attn_score = torch.matmul(Q, K.transpose(-2, -1))
        # 缩放，防止数值爆炸
        d_k = Q.size(-1)**0.5
        attn_score = attn_score / d_k

        # softmax → 得到权重（代表每个词对其他词的关注度）
        attn_weight = F.softmax(attn_score, dim=-1)

        # 权重 × V，加权求和，融合上下文信息
        out = torch.matmul(attn_weight, V)
        return out, attn_weight


# 测试
if __name__ == "__main__":
    batch = 1
    seq_len = 4   # 一句话4个词
    embed_dim = 16
    x = torch.randn(batch, seq_len, embed_dim)
    attn = SelfAttention(embed_dim)
    result, weight = attn(x)
    print("输出形状：", result.shape)
    print("注意力权重形状：", weight.shape)
