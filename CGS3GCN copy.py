import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import encoding, functional, neuron, surrogate, layer, learning
import math

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, tau, theta):
        super().__init__()
        self.W = nn.Sequential(
            layer.Conv1d(in_features, out_features, kernel_size=1, bias=False),
            layer.BatchNorm1d(out_features)
        )
        self.lif = neuron.LIFNode(tau=tau, v_threshold=theta, decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan())
        
    def forward(self, x, adj):
        x = self.lif(x)
        x = torch.matmul(x, adj.transpose(-2, -1))
        x = self.W(x)
        return x
    
class CPG_GCN(nn.Module):
    def __init__(self, hidden, tau, theta):
        super().__init__()
        
        self.agg = GraphConvLayer(hidden, hidden, tau, theta)
        self.high_W = nn.Sequential(
            neuron.LIFNode(tau=tau, v_threshold=theta, decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.Conv1d(hidden, hidden, kernel_size=1, bias=False),
            layer.BatchNorm1d(hidden)
        )
        self.gate = nn.Sequential(
            neuron.LIFNode(tau=tau, v_threshold=theta, decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.Conv1d(hidden*2, hidden, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, adj, prototype, center):
        low = self.agg(x, adj)  # [T, N, hidden, S]
        high = self.high_W(prototype - low)
        cat = torch.cat([center, high], -2)
        x = low + self.gate(cat) * high
        return x

class STCA(nn.Module):
    def __init__(self, T: int, channels: int, reduction: int = 4):
        super().__init__()
        # 利用 T 个时间步的全局脉冲发放率来生成时间-通道混合权重
        self.fc = nn.Sequential(
            nn.Linear(T * channels, (T * channels) // reduction),
            nn.SiLU(),
            nn.Linear((T * channels) // reduction, T * channels),
            nn.Sigmoid()
        )

    def forward(self, x_spike: torch.Tensor, x):
        # x_spike shape: [T, N, C, L^2]
        # x shape: [T, N, C, L^2] 是脉冲前的膜电位
        T, N, C, L2 = x_spike.shape
        spe_spike_rate = x_spike.mean(dim=-1) 
        
        
        flat_rate = spe_spike_rate.transpose(0, 1).reshape(N, T * C)
        spe_attn = self.fc(flat_rate)
        spe_attn = spe_attn.view(N, T, C, 1).transpose(0, 1)
        return x + x * spe_attn

class CGS3GCN(nn.Module):
    def __init__(self, T: int = 3, img_size=15, num_cls=16, input_dim=144, hidden:int = 32, 
                 spa_k: int = 2, spe_k: int = 1, tau :float = 2.0, theta: float = 1.0):
        super().__init__()
        self.T = T 
        self.img_size = img_size
        self.hidden = hidden
        self.input_dim = input_dim
        self.l_squared = img_size * img_size  # l^2, assuming square patches
        self.spa_k = max(spa_k, 1)
        self.spe_k = max(spe_k, 1)
        self.num_cls = num_cls

        self.reduction_bn1 = nn.Sequential(
            layer.Conv1d(input_dim, hidden*2, kernel_size=1, bias=False),
            layer.BatchNorm1d(hidden*2)
        )
        
        self.lif1 = neuron.LIFNode(tau=tau, v_threshold=theta, decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan())
        self.spike_attn = STCA(T=T, channels=hidden*2, reduction=4)
        
        self.reduction_bn2 = nn.Sequential(
            neuron.LIFNode(tau=tau, v_threshold=theta, decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.Conv1d(hidden*2, hidden, kernel_size=1, bias=False),
            layer.BatchNorm1d(hidden)
        )
        self.lif = neuron.LIFNode(tau=tau, v_threshold=theta, decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan())

        self.spa_cghc_layers = nn.ModuleList([CPG_GCN(hidden, tau, theta) for _ in range(self.spa_k)])

        self.center_spec_proj = nn.Sequential(
            layer.Conv1d(3, hidden, kernel_size=1, bias=False),
            layer.BatchNorm1d(hidden),
        )
        self.spec_adj_lif = neuron.LIFNode(tau=tau, v_threshold=theta, decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan())
        self.spe_cghc_layers = nn.ModuleList([CPG_GCN(hidden, tau, theta) for _ in range(self.spe_k)])

        self.fusion_gate = nn.Sequential(
            neuron.LIFNode(tau=tau, v_threshold=theta, decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.Linear(hidden*2, hidden),
            nn.Sigmoid()
        )

        self.out_linear = nn.Linear(hidden, num_cls)

        functional.set_step_mode(self, step_mode='m')
        functional.set_backend(self, backend='cupy')
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d) or isinstance(m, layer.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                   nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d) or isinstance(m, layer.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, layer.BatchNorm1d) or isinstance(m, layer.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        functional.reset_net(self)

        N, l_squared, S = x.shape
        l = int(l_squared ** 0.5)

        x = x.unsqueeze(0).expand(self.T, -1, -1, -1)
        x = x.permute(0, 1, 3, 2)  # [T, N, S, L^2]

        x_seq1 = self.reduction_bn1(x)
        
        x_spike1 = self.lif1(x_seq1)
        x_seq1 = self.spike_attn(x_spike1, x_seq1)
        
        x_seq2 = self.reduction_bn2(x_seq1)
        
        center_idx = l_squared // 2
        center = x_seq2[:, :, :, center_idx:center_idx+1]
        
        
        center_spe = center.expand(-1, -1, -1, self.hidden)
        spe_max = x_seq2.max(-1, keepdim=True)[0]
        spe_mean = x_seq2.mean(-1, keepdim=True)
        center_tokens = torch.cat([center, spe_max, spe_mean], -1)  
        center_tokens = center_tokens.permute(0, 1, 3, 2)

        center_node_feat = self.center_spec_proj(center_tokens)  # [T, N, 特征, 光谱嵌入节点]
        spe_spike = self.spec_adj_lif(center_node_feat)
        spectral_adj = self.AdaptivePearsonAdj(spe_spike)
        spec_gcn = center_node_feat
        for layer in self.spe_cghc_layers:
            spec_gcn = layer(spec_gcn, spectral_adj, center_node_feat, center_spe)
        spec = spec_gcn.mean(-2)
        
        spa_spike = self.lif(x_seq2)
        adj = self.AdaptivePearsonAdj(spa_spike)
        center_spa = center.expand(-1, -1, -1, l_squared)
        spa_gcn = x_seq2
        for layer in self.spa_cghc_layers:
            spa_gcn = layer(spa_gcn, adj, x_seq2, center_spa)       
        spa = spa_gcn.mean(-1)

        gate = self.fusion_gate(torch.cat([spa, spec], -1)) 
        x_center = gate * spa + (1 - gate) * spec 
        
        out = self.out_linear(x_center)
        return out

    def AdaptivePearsonAdj(self, spike: torch.Tensor):
        # spike: [T, N, hidden, L^2]
        # 中心化和归一化
        mu = spike.mean(dim=-2, keepdim=True)
        centered_features = spike - mu
        feature_norm = F.normalize(centered_features, p=2, dim=-2)
        feature_norm_t = feature_norm.transpose(-2, -1)
        sim = torch.matmul(feature_norm_t, feature_norm)
        adj = F.relu(sim)
        deg = adj.sum(-1, keepdim=True)
        deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
        adj_norm = adj * deg_inv_sqrt * deg_inv_sqrt.transpose(-2, -1)
        return adj_norm