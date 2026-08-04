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


