# -*- coding: utf-8 -*-
"""
句子情感判断器（基于自注意力）
================================
在你自己的 SelfAttention 基础上，加了一层残差/归一化/前馈，训练一个
字级别的二分类模型：输入一句话 -> 输出「正面/负面 + 置信度」，
并告诉你「是哪几个字在推动这个判断」。

用法：
    python sentiment.py                   # 首次自动训练，然后进入交互输入
    python sentiment.py --retrain         # 强制重新训练
    python sentiment.py -t "这个产品太差了"  # 直接判断一句话，不进交互
    python sentiment.py                 # 交互模式下输入 q 退出
"""
import argparse
import math
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "sentiment_attn.pt")
MAX_LEN = 40


# =====================================================================
# 1. 造训练数据
#    思路：用「主语 + 程度 + 否定 + 情感词 + 语气」拼句子，情感极性由
#    情感词决定；一旦出现否定词，极性翻转。这样模型必须学会「不/没」
#    改变了哪个词的含义 —— 这正是注意力该干的活。
# =====================================================================
POS_WORDS = ["好", "棒", "优秀", "满意", "喜欢", "推荐", "惊喜", "舒服", "划算",
             "完美", "值得", "不错", "给力", "惊艳", "靠谱", "新鲜", "干净",
             "热情", "专业", "方便", "实惠", "好看", "好吃", "好用", "开心",
             "良心", "细致", "快", "耐用", "漂亮"]
NEG_WORDS = ["差", "糟", "垃圾", "失望", "讨厌", "后悔", "难受", "贵", "慢",
             "难用", "坑", "破", "假", "脏", "吵", "敷衍", "油腻", "难吃",
             "崩溃", "糟糕", "无聊", "难看", "生气", "麻烦", "难喝", "卡顿"]
SUBJECTS = ["这个产品", "这家店", "这部电影", "这本书", "这顿饭", "这次旅行",
            "快递", "客服", "服务", "质量", "价格", "包装", "口感", "环境",
            "他", "她", "这个手机", "这个酒店", "物流"]
DEGREE = ["", "很", "非常", "太", "特别", "超级", "真的", "有点", "稍微",
          "极其", "十分", "还算", "挺"]
NEGATION = ["不", "没", "没有", "不怎么", "不太", "一点也不"]
SUFFIX = ["", "。", "！", "，推荐给大家", "，再也不买了", "，下次还会来",
          "，无语了", "，说不过去", "，出乎意料"]
# 转折句式：结论由后半句决定，逼模型学会「但是」后面的内容权重更高
CONTRAST = ["虽然{}但是{}", "虽说{}但{}", "{}，不过{}", "尽管{}，可{}", "{}，就是{}"]


def _clause():
    """生成一个小句，返回 (文本, 极性)"""
    is_pos = random.random() < 0.5
    word = random.choice(POS_WORDS if is_pos else NEG_WORDS)
    label = 1 if is_pos else 0
    neg = random.choice(NEGATION) if random.random() < 0.35 else ""
    if neg:
        label = 1 - label          # 否定 -> 极性翻转
    return random.choice(DEGREE) + neg + word, label


def build_dataset(n=14000):
    seen, data = set(), []
    for _ in range(n):
        if random.random() < 0.3:                  # 30% 转折句
            a, la = _clause()
            b, lb = _clause()
            if la == lb:                           # 确保转折前后极性相反
                continue
            text = random.choice(CONTRAST).format(a, b) + random.choice(SUFFIX)
            label = lb                             # 结论看后半句
        else:
            a, label = _clause()
            # 30% 的样本不加主语，避免模型只会认「主语+评价」的固定套路
            prefix = "" if random.random() < 0.3 else random.choice(SUBJECTS)
            text = prefix + a + random.choice(SUFFIX)
        if text in seen:
            continue
        seen.add(text)
        data.append((text, label))
    random.shuffle(data)
    return data


# =====================================================================
# 2. 词表 & 批处理
# =====================================================================
class Vocab:
    PAD, UNK = 0, 1

    def __init__(self, texts):
        chars = {}
        for t in texts:
            for c in t:
                chars[c] = chars.get(c, 0) + 1
        self.itos = ["<pad>", "<unk>"] + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        ids = [self.stoi.get(c, self.UNK) for c in text][:MAX_LEN]
        return torch.tensor(ids, dtype=torch.long)


def collate(batch, vocab):
    seqs = [vocab.encode(t) for t, _ in batch]
    maxlen = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), maxlen, dtype=torch.long)   # PAD 补 0
    mask = torch.zeros(len(seqs), maxlen, dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
        mask[i, :len(s)] = True
    y = torch.tensor([l for _, l in batch], dtype=torch.long)
    return x, mask, y


# =====================================================================
# 3. 模型：你的 SelfAttention + 残差 + LayerNorm + 前馈
# =====================================================================
class SelfAttention(nn.Module):
    """和 self_attention.py 里同一个东西，多了 mask 参数（屏蔽 PAD）"""

    def __init__(self, dim):
        super().__init__()
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x, mask=None):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim)
        if mask is not None:
            # mask: [B, L] -> [B, 1, L]，屏蔽掉 PAD 列（不允许任何词看 PAD）
            score = score.masked_fill(~mask.unsqueeze(1), float("-inf"))
        weight = F.softmax(score, dim=-1)
        return torch.matmul(weight, V), weight


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Linear(dim * 2, dim))

    def forward(self, x, mask=None):
        a, w = self.attn(self.ln1(x), mask)
        x = x + a                      # 残差：保留原始信息
        x = x + self.ffn(self.ln2(x))
        return x, w


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, dim=64, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=Vocab.PAD)
        self.pos = nn.Embedding(MAX_LEN, dim)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)
        self.dim = dim

    def forward(self, x, mask=None):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos)
        last_w = None
        for blk in self.blocks:
            h, last_w = blk(h, mask)
        h = self.ln(h)
        # 把 PAD 位置排除后做平均池化 -> 得到整句向量
        m = mask.unsqueeze(-1).float() if mask is not None else 1.0
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(pooled), last_w


# =====================================================================
# 4. 训练
# =====================================================================
def train(epochs=32, bs=128, lr=4e-3):
    random.seed(0)
    torch.manual_seed(0)
    data = build_dataset()
    split = int(len(data) * 0.9)
    tr, va = data[:split], data[split:]
    vocab = Vocab([t for t, _ in tr])
    model = SentimentModel(len(vocab))
    print(f"训练样本 {len(tr)} 条 / 验证 {len(va)} 条，词表 {len(vocab)} 字，"
          f"参数量 {sum(p.numel() for p in model.parameters())/1e3:.1f}K")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * (len(tr) // bs))

    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(tr)
        total, correct = 0, 0
        for i in range(0, len(tr) - bs + 1, bs):
            x, mask, y = collate(tr[i:i + bs], vocab)
            logits, _ = model(x, mask)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            correct += (logits.argmax(-1) == y).sum().item()
            total += len(y)
        if ep % 5 == 0 or ep == 1:
            model.eval()
            vc = vt = 0
            with torch.no_grad():
                for i in range(0, len(va), 128):
                    x, mask, y = collate(va[i:i + 128], vocab)
                    logits, _ = model(x, mask)
                    vc += (logits.argmax(-1) == y).sum().item()
                    vt += len(y)
            print(f"  epoch {ep:>2}  train {correct/total:.3f}  val {vc/vt:.3f}")

    torch.save({"state": model.state_dict(), "itos": vocab.itos}, MODEL_PATH)
    print(f"模型已保存 -> {MODEL_PATH}")
    return model, vocab


def load():
    ck = torch.load(MODEL_PATH, weights_only=False)
    vocab = Vocab([""])
    vocab.itos, vocab.stoi = ck["itos"], {c: i for i, c in enumerate(ck["itos"])}
    model = SentimentModel(len(vocab))
    model.load_state_dict(ck["state"])
    model.eval()
    return model, vocab


# =====================================================================
# 5. 预测 + 逐字归因
# =====================================================================
def forward_prob(model, vocab, text):
    """纯前向，返回 (概率, logits, 注意力权重)"""
    if not text:
        return torch.tensor([0.5, 0.5]), torch.zeros(2), torch.zeros(1, 1)
    return forward_ids(model, vocab.encode(text).tolist())


def forward_ids(model, ids):
    x = torch.tensor([ids], dtype=torch.long)
    mask = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad():
        logits, weight = model(x, mask)
    return F.softmax(logits, -1)[0], logits[0], weight[0]


def explain_by_occlusion(model, vocab, text, cls):
    """
    遮挡法：把第 i 个字替换成 <unk>（而不是删掉，避免长度变化带来的干扰），
    看「正负类的 logit 差距」缩小了多少。
    用 logit 差值而不是概率，是因为概率常常饱和在 0/1，删字也不动。
    """
    ids = vocab.encode(text).tolist()
    _, base_logits, _ = forward_ids(model, ids)
    base_margin = (base_logits[cls] - base_logits[1 - cls]).item()
    contrib = []
    for i in range(len(ids)):
        probe = list(ids)
        probe[i] = Vocab.UNK
        _, lg, _ = forward_ids(model, probe)
        margin = (lg[cls] - lg[1 - cls]).item()
        contrib.append(base_margin - margin)       # 正 = 这个字支撑当前结论
    return torch.tensor(contrib)


def show(model, vocab, text):
    text = text.strip()
    if not text:
        return
    prob, logits, weight = forward_prob(model, vocab, text)
    cls = int(prob.argmax())
    contrib = explain_by_occlusion(model, vocab, text, cls)

    label = "正面" if cls == 1 else "负面"
    bar = lambda v, n=24: "█" * max(1, int(v * n))
    print(f"\n输入：{text}")
    print(f"判断：{label}    置信度 {prob[cls].item():.1%}")
    print(f"   负面 {bar(prob[0].item())} {prob[0].item():.3f}")
    print(f"   正面 {bar(prob[1].item())} {prob[1].item():.3f}")

    # 逐字归因：删掉它，结论会动摇多少
    m = contrib.abs().max().clamp(min=1e-6)
    print(f"\n  『哪些字在支撑「{label}」这个判断』")
    order = contrib.abs().argsort(descending=True)
    for idx in order[:min(6, len(text))]:
        ch = text[idx]
        v = (contrib[idx] / m).item()
        if v > 0.15:
            tag, arrow = "支撑", "▲"
        elif v < -0.15:
            tag, arrow = "反对", "▼"
        else:
            tag, arrow = "无关", "·"
        print(f"    {ch}  {arrow} {'■' * max(1, int(abs(v) * 14)):<14} "
              f"{v:+.2f}  {tag}")

    # 注意力都跑到哪去了（看模型把信息汇总到哪些字）
    n = len(text)
    col = weight[:n, :n].sum(0)
    col = col / col.sum()
    top = col.topk(min(3, n))
    print("\n  注意力最集中的字（模型主要把信息汇总到这里）")
    for j, v in zip(top.indices.tolist(), top.values.tolist()):
        print(f"    「{text[j]}」 占比 {v:.1%}")
    if top.values[0].item() > 0.4:
        print("    ⚠ 注意力集中在少数几个字上（attention sink，小模型常见），"
              "所以重要性用遮挡法算，不直接用注意力权重")


# =====================================================================
# 6. 入口
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--text", help="直接判断一句话")
    ap.add_argument("--retrain", action="store_true", help="强制重新训练")
    args = ap.parse_args()

    if args.retrain or not os.path.exists(MODEL_PATH):
        model, vocab = train()
    else:
        model, vocab = load()
        print("已加载训练好的模型")

    if args.text:
        show(model, vocab, args.text)
        return

    print("\n输入一句话试试（输入 q 退出）：")
    while True:
        try:
            s = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if s.strip().lower() in ("q", "quit", "exit"):
            break
        show(model, vocab, s)
    print("再见")


if __name__ == "__main__":
    main()
