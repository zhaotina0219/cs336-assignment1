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