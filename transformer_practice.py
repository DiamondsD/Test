import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个注意力头的维度
        
        # 定义线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力计算
        Q: [batch_size, num_heads, seq_len, d_k]
        K/V: [batch_size, num_heads, seq_len, d_k]
        mask: [batch_size, 1, seq_len, seq_len]
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        output = torch.matmul(attn_probs, V)
        return output, attn_probs
        
    def split_heads(self, x):
        """
        将输入张量分割为多个注意力头
        x: [batch_size, seq_len, d_model]
        return: [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        """
        合并多个注意力头的输出
        x: [batch_size, num_heads, seq_len, d_k]
        return: [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # 线性变换 + 分割头
        Q = self.split_heads(self.W_q(Q))  # [batch, num_heads, seq_len, d_k]
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # 计算注意力
        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并头 + 线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output, attn_probs

class PositionWiseFFN(nn.Module):
    """位置前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, src_mask):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, src_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络子层
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 自注意力子层（带因果掩码）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 交叉注意力子层（连接编码器输出）
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        
        # 前馈网络子层
        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)
        x = self.norm3(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1,
                 max_seq_length=100):
        super().__init__()
        self.d_model = d_model
        
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # 编码器
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def generate_src_mask(self, src):
        """生成源序列填充掩码"""
        return (src != 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_len]
    
    def generate_tgt_mask(self, tgt):
        """生成目标序列因果掩码"""
        batch_size, tgt_len = tgt.size()
        # 下三角因果掩码
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
        # 填充掩码
        padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        return causal_mask.unsqueeze(0) & padding_mask  # [batch, 1, tgt_len, tgt_len]
    
    def forward(self, src, tgt):
        # 处理源序列
        src_mask = self.generate_src_mask(src)
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            src_emb = layer(src_emb, src_mask)
            
        # 处理目标序列
        tgt_mask = self.generate_tgt_mask(tgt)
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            tgt_emb = layer(tgt_emb, src_emb, src_mask, tgt_mask)
            
        # 输出预测
        output = self.fc_out(tgt_emb)
        return output

# 测试代码
if __name__ == "__main__":
    # 参数设置
    SRC_VOCAB_SIZE = 5000
    TGT_VOCAB_SIZE = 5000
    BATCH_SIZE = 32
    SEQ_LENGTH = 10
    
    # 初始化模型
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    )
    
    # 生成测试数据
    src = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))  # [batch, src_len]
    tgt = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))  # [batch, tgt_len]
    
    # 前向传播
    output = model(src, tgt)
    print("输出形状:", output.shape)  # 应为 [batch_size, tgt_len, tgt_vocab_size]