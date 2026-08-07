import torch
import math
from torch import nn
# 普通tensor只是数据，即使放入模型，PyTorch也不一定会将其视为需要训练的权重
# nn.Parameter 是一种特殊 Tensor，它告诉 nn.Module：
# 这是模型的可训练参数，需要出现在 model.parameters() 和 state_dict() 中，并在训练时计算梯度
class Linear(nn.Module):
    def __init__(
            self,
            in_features:int,
            out_features:int,
            device = None,
            dtype = None,       
    ):
        super().__init__()
        weight_tensor = torch.empty(
            (out_features,in_features),
            device = device,
            dtype=dtype
        )
        # 神经网络开始训练之前大致围绕0分布，不能太大也不能全为0
        self.weight = nn.Parameter(weight_tensor)
        std_ = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean = 0,
            std = std_ ,
            a = -3 * std_ ,
            b = 3 * std_
        )
 
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return (x @ self.weight.T)

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings:int,
            embedding_dim:int,
            device=None,
            dtype=None,
    ):
        super().__init__()
        weight_tensor = torch.empty(
            (num_embeddings,embedding_dim),
            device=device,
            dtype=dtype,
        )
        self.weight = nn.Parameter(weight_tensor)
        nn.init.trunc_normal_(
            self.weight,
            mean = 0,
            std = 1,
            a = -3,
            b = 3
        )
    def forward(self,token_ids:torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model:int,
            eps:float = 1e-5,
            device = None,
            dtype = None,
    ):
        super().__init__()
        self.eps = eps
        weight_tensor = torch.ones(
            d_model,
            device = device,
            dtype = dtype
        )
        self.weight = nn.Parameter(weight_tensor)
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x_float = x.to(torch.float32)
        squared = x_float ** 2
        mean_square = squared.mean(dim=-1, keepdim=True)
        #math.sqrt只能处理单个数字，不能处理张量
        rms = torch.sqrt(mean_square + self.eps)
        normalized = x_float / rms
        res = normalized * self.weight
        return res.to(in_type)

def silu(x:torch.Tensor) ->torch.Tensor:
    sig = torch.sigmoid(x)
    return x * sig

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model:int,
            d_ff:int,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.w1 = Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w3 = Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2 = Linear(d_ff,d_model,device=device,dtype=dtype)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        content = self.w1(x)
        activated = silu(content)
        gate = self.w3(x)
        gated = activated * gate
        return self.w2(gated)
# RoPE写入位置信息，位置越靠后旋转角度越大，不同维度的旋转速度不同
class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta:float,
            d_k:int,
            max_seq_len:int,
            device=None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        if d_k % 2 != 0:
            raise ValueError("d_k 必须是偶数")
        dimension_indices = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )
        # 每一对共享一个旋转角度，torch.outer用于计算一维向量的外积
        inverse_frequencies = 1.0 / (theta ** (dimension_indices/d_k))
        position_indices = torch.arange(
            0,
            max_seq_len,
            1,
            device = device,
            dtype = torch.float32
        )
        angles = torch.outer(
            position_indices,
            inverse_frequencies
        )
        cosine_values = torch.cos(angles)
        sine_values = torch.sin(angles)
        self.register_buffer(
            "cos_table",
            cosine_values,
            persistent=False,
)
        self.register_buffer(
            "sin_table",
            sine_values,
            persistent=False,
)
        
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor) -> torch.Tensor:
        cos_values = self.cos_table[token_positions]
        sin_values = self.sin_table[token_positions]
        cos_trans = cos_values.to(x.dtype)
        sin_trans = sin_values.to(x.dtype)
        x_even = x[...,0::2]
        x_odd = x[...,1::2]
        rotated_even = x_even * cos_trans - x_odd * sin_trans
        rotated_odd = x_even * sin_trans + x_odd * cos_trans
        stacked = torch.stack([rotated_even,rotated_odd],dim=-1)
        flattened = stacked.flatten(start_dim=-2)
        return flattened
# softmax不能对原始数值直接做exp，容易溢出为inf，因此实现时沿某一维度找到最大值，再令x-max，此时最大值也不过为0
def softmax(
        x:torch.Tensor,
        dim:int,
) -> torch.Tensor:
    max_result = torch.max(
        x,
        dim = dim,
        keepdim = True,
    )
    # 指定dim之后，max函数不仅返回最大值，还会返回最大值的索引
    max_values = max_result.values
    stable_x = x - max_values
    exp_values = torch.exp(stable_x)
    exp_values_sum = exp_values.sum(
        dim = dim,
        keepdim = True,
    )
    prop = exp_values / exp_values_sum
    return prop

def scaled_dot_product_attention(
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        mask:torch.Tensor|None=None    
) -> torch.Tensor:
    d_k = Q.shape[-1]
    K_transposed = K.transpose(-2,-1)
    scores = Q @ K_transposed / math.sqrt(d_k)
    # masked_fill(mask,value)会把mask中为True的位置替换为指定值
    if mask is not None:
        scores = scores.masked_fill(~mask,
                                        float("-inf"),)
    weights = softmax(
        x = scores,
        dim = -1,
    )
    output = weights @ V
    return output
