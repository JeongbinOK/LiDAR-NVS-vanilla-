import torch
import torch.nn as nn
from torch import Tensor

from module import QGS, TBD_, TBD__
class Ours(nn.Module):

    def __init__(self, p2g_cfg, g2g_cfg, g2p_cfg):
        super().__init__()
        self.p2g_cfg = p2g_cfg
        self.g2g_cfg = g2g_cfg
        self.g2p_cfg = g2p_cfg 

        self.p2g_model = QGS(self.p2g_cfg)
        self.g2g_model = TBD_(self.g2g_cfg)
        self.g2p_model = TBD__(self.g2p_cfg) 


    def forward(self, batch, batch_idx):
        return None