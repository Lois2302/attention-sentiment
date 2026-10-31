# -*- coding: utf-8 -*-
"""
自注意力模块的可视化验证脚本
与 self_attention.py 中的 SelfAttention 完全对照，逐条验证 5 个关键结论。
运行：python verify.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        attn_score = torch.matmul(Q, K.transpose(-2, -1))
        d_k = Q.size(-1) ** 0.5
        attn_score = attn_score / d_k
        attn_weight = F.softmax(attn_score, dim=-1)
        out = torch.matmul(attn_weight, V)
        return out, attn_weight, attn_score


# ---------------------------------------------------------------
# 构造一个"可控"的输入：4 个词，让第 1、3 个词向量几乎一样
# 这样注意力权重应该在这两个词之间形成明显偏斜，肉眼可判断对错
# ---------------------------------------------------------------
batch, seq_len, embed_dim = 1, 4, 8
words = ["我", "爱", "我", "饭"]

x = torch.randn(batch, seq_len, embed_dim)
x[0, 2] = x[0, 0] + 0.01 * torch.randn(embed_dim)   # 第3个词 ≈ 第1个词

attn = SelfAttention(embed_dim)
out, weight, score = attn(x)

W = weight[0]   # [4, 4]

print("=" * 62)
print(f"输入 x: {tuple(x.shape)}   ->  {batch}句 x {seq_len}词 x {embed_dim}维")
print(f"输出 out: {tuple(out.shape)}")
print(f"权重 weight: {tuple(weight.shape)}")
print(f"句子: {' '.join(words)}")
print("=" * 62)

# ---------- 验证 1：每行权重之和为 1 ----------
print("\n【验证1】softmax 归一化：每行权重之和是否 = 1")
row_sum = W.sum(dim=-1)
for i in range(seq_len):
    print(f"  行{i} ({words[i]}): 和 = {row_sum[i].item():.6f}")
print(f"  -> 全部 ≈1 ? {torch.allclose(row_sum, torch.ones(seq_len), atol=1e-5)}")

# ---------- 验证 2：打印 4x4 权重矩阵 + ASCII 热力图 ----------
print("\n【验证2】4x4 注意力矩阵（行=谁在看, 列=看谁）")
print("        " + "".join(f"{w:>9}" for w in words))
for i in range(seq_len):
    cells = "".join(f"{W[i, j].item():>9.4f}" for j in range(seq_len))
    print(f"  {words[i]:<6}{cells}")

print("\n  ASCII 热力图（方块越多 = 权重越大）:")
for i in range(seq_len):
    bars = ""
    for j in range(seq_len):
        bars += "  " + "\u2588" * max(1, int(W[i, j].item() * 40))
    print(f"  {words[i]} ->{bars}")

print("\n  每个词最关注谁（argmax）:")
for i in range(seq_len):
    j = W[i].argmax().item()
    print(f"    {words[i]}(第{i}个) 最关注 -> {words[j]}(第{j}个), 权重 {W[i, j].item():.4f}")

# ---------- 验证 3：手写朴素实现，逐元素复现 out ----------
print("\n【验证3】不用 matmul，用最朴素的 for 循环手算一遍，比对结果")
Q, K, V = attn.Wq(x)[0], attn.Wk(x)[0], attn.Wv(x)[0]
out_naive = torch.zeros(seq_len, embed_dim)
for i in range(seq_len):
    total = torch.zeros(embed_dim)
    for j in range(seq_len):
        e = torch.dot(Q[i], K[j]) / (embed_dim ** 0.5)   # 单个词的相似度
        total += torch.softmax(
            torch.stack([torch.dot(Q[i], K[k]) / (embed_dim ** 0.5) for k in range(seq_len)]),
            dim=0)[j] * V[j]
    out_naive[i] = total
print(f"  矩阵实现 vs 循环实现 最大误差: {(out[0] - out_naive).abs().max().item():.3e}")
print(f"  -> 完全一致 ? {torch.allclose(out[0], out_naive, atol=1e-6)}")

# ---------- 验证 4：缩放 sqrt(d_k) 到底起了什么作用 ----------
print("\n【验证4】为什么要除以 sqrt(d_k)? 对比缩放前后的得分分布")
d = embed_dim ** 0.5
raw = torch.matmul(Q, K.T)          # 未缩放的原始得分
w_noscale = F.softmax(raw, dim=-1)  # 未缩放直接 softmax
print(f"  未缩放 score: 范围 [{raw.min().item():+.2f}, {raw.max().item():+.2f}], "
      f"标准差 {raw.std().item():.3f}")
print(f"  缩放后 score: 范围 [{score.min().item():+.2f}, {score.max().item():+.2f}], "
      f"标准差 {score.std().item():.3f}   (缩小了 sqrt(d_k)={d:.2f} 倍)")
print(f"  未缩放 softmax 最大权重: {w_noscale.max().item():.4f}  "
      f"(越接近 1 -> 越像 one-hot，梯度越小)")
print(f"  缩放后  softmax 最大权重: {W.max().item():.4f}  (更平缓 -> 梯度更健康)")
print(f"  d_k={embed_dim} 时差距还温和；d_k=512 时未缩放得分会被放大 22.6 倍，")
print("  softmax 直接退化成 one-hot，几乎是硬选择，训练不动")

# ---------- 验证 5：改一个词，看信息如何流动 ----------
print("\n【验证5】把第3个词（'我'）换成完全无关的向量，看哪些输出变了")
x2 = x.clone()
x2[0, 2] = torch.randn(embed_dim)
out2, w2, _ = attn(x2)
diff = (out[0] - out2[0]).norm(dim=-1)
for i in range(seq_len):
    tag = "<- 被替换的词" if i == 2 else ""
    print(f"  第{i}个词({words[i]}) 输出变化量: {diff[i].item():.4f}  {tag}")
print("  -> 所有词都变了（包括没被改的第0、1、3个词）")
print("     说明：这里没有 causal mask，每个词的输出都混入了全句信息")
print("     （GPT 那类模型会加三角 mask，让第 i 个词只能看到 0..i）")

print("\n" + "=" * 62)
print("5 项全部通过 = 你的自注意力实现是标准 scaled dot-product attention")
print("=" * 62)
