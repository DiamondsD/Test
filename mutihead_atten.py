import torch
import torch.nn as nn
import torch.nn.functional as F

class Mutihead_atten(nn.Module):

    def __init__(self, embd_size, heads):
        super(Mutihead_atten, self).__init__()
        self.embd_size = embd_size
        self.heads = heads
        self.embd_dim = embd_size // heads
        assert (self.heads * self.embd_dim == self.embd_size)#确保词嵌入维度可以被头数整除
        self.query = nn.Linear(self.embd_size, self.embd_size, bias=False)
        self.key = nn.Linear(self.embd_size, self.embd_size, bias=False)
        self.values = nn.Linear(self.embd_size, self.embd_size, bias=False)
        self.fc_out = nn.Linear(self.embd_size, self.embd_size)
    
    
    def forward(self, x, mask = None):
        '''
        参数x是输入,形状是(batch_size, seq_lenth, d_model),d_model 是可以学习的超参数,代表的是每个token的词向量维度
        x: 输入张量，形状 (N, seq_length, embed_size)
        mask: 可选掩码，形状 (N, 1, 1, seq_length) 或 (N, 1, seq_length, seq_length)
        '''
        
        Batch_size, seq_length, d_model = x.shape


        #生成Q，K，V 三个矩阵都是从x生成的，使用linear线性操作，是可以进行反向传播的
        Q = self.query(x)  # (Batch_size, seq_length, d_model)
        K = self.key(x)    # (Batch_size, seq_length, d_model)
        V = self.values(x) # (Batch_size, seq_length, d_model)
        

        '''
        拆分多头,需要注意的是,生成qkv是必须在钗头操作之前的,原因有:
        1.如果先拆多头，会让参数量变少，让模型的学习能力变弱 
        2.先拆头的话，每个头都会有一个独立的线性层

        '''
        #拆多头
        Q = Q.view(Batch_size, seq_length, self.heads, self.embd_dim)
        K = K.view(Batch_size, seq_length, self.heads, self.embd_dim)
        V = V.view(Batch_size, seq_length, self.heads, self.embd_dim)

        #变换维度，方便进行attention,size_like(Batch_size, self.heads, seq_length, self.embd_dim),在这里对K进行转置
        Q = Q.permute(0, 2, 1, 3) #(Batch_size, self.heads, seq_length, self.embd_dim)
        K = K.permute(0, 2, 3, 1) #(Batch_size, self.heads, self.embd_dim, seq_length)
        V = V.permute(0, 2, 1, 3) #(Batch_size, self.heads, seq_length, self.embd_dim)

        #计算attention score
        atten_score = torch.matmul(Q, K)

        #进行缩放
        atten_score = atten_score / self.embd_dim ** 0.5

        #进行掩码，如果有掩码
        if mask is not None:
            atten_score = atten_score.masked_fill(mask == 0, float("-1e20"))

        # 归一化，计算权重
        atten_score = F.softmax(atten_score, dim=-1) #

        # 应用注意力到V
        out = torch.matmul(atten_score, V) #(Batch_size, self.heads, seq_length, self.embd_dim)
        out = out.permute(0, 2, 1, 3)   # 变为原来的形状
        out = out.view(Batch_size, seq_length, self.embd_size)

        #最后的线性变换
        out = self.fc_out(out)
        return out




        