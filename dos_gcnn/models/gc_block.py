"""Graph Convolution Block for DOS-GCNN."""

from typing import Union, Tuple
import torch
from torch import Tensor
from torch.nn import Sequential, Linear
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.nn.conv import MessagePassing


class GC_block(MessagePassing):
    """Graph Convolution Block using message passing."""
    
    def __init__(
        self, 
        channels: Union[int, Tuple[int, int]], 
        dim: int = 0,
        aggr: str = 'mean', 
        **kwargs
    ):
        super(GC_block, self).__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim

        if isinstance(channels, int):
            channels = (channels, channels)

        self.mlp = Sequential(
            Linear(sum(channels) + dim, channels[1]),
            torch.nn.PReLU(),
        )
        self.mlp2 = Sequential(
            Linear(dim, dim),
            torch.nn.PReLU(),
        )

    def forward(
        self, 
        x: Union[Tensor, PairTensor], 
        edge_index: Adj,
        edge_attr: OptTensor = None, 
        size: Size = None
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out += x[1]
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        z = torch.cat([x_i, x_j, self.mlp2(edge_attr)], dim=-1)
        z = self.mlp(z)
        return z

