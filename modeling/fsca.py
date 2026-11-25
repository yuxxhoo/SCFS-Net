"""Frequency-conditioned cross-modal spatial–channel collaboration."""
import torch
from torch import nn
import torch.nn.functional as F

class ResidualDilatedDenseBlock(nn.Module):
    def __init__(self, dim, growth, dilations=(1, 2, 3)):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = dim
        for dilation in dilations:
            self.layers.append(nn.Conv2d(channels, growth, 3, padding=dilation, dilation=dilation))
            channels += growth
        self.compress = nn.Conv2d(channels, dim, 1, bias=False)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(F.gelu(layer(torch.cat(features, 1))))
        return x + self.compress(torch.cat(features, 1))

class SpatialChannelProjection(nn.Module):
    """The explicit 1x1 channel projection followed by 3x3 depthwise context."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pointwise = nn.Conv2d(in_dim, out_dim, 1, bias=False)
        self.depthwise = nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim, bias=False)

    def forward(self, x):
        return self.depthwise(self.pointwise(x))

class FSCA(nn.Module):
    """Frequency-Conditioned Spatial–Channel Collaborative Attention (FSCA)."""
    def __init__(self, dim, cfg):
        super().__init__()
        s = cfg.MODEL.SCFS
        self.inner = int(s.ATTENTION_DIM)
        self.heads = int(s.NUM_HEADS)
        if self.inner % self.heads or self.inner < self.heads:
            raise ValueError("ATTENTION_DIM must be divisible by NUM_HEADS")
        self.head_dim = self.inner // self.heads
        self.rddb = ResidualDilatedDenseBlock(dim, int(s.RDDB_GROWTH))
        self.q, self.k, self.v = (SpatialChannelProjection(dim, self.inner) for _ in range(3))
        # Per target modality: head temperatures and three source coefficients.
        hidden = max(3 * int(s.NUM_BANDS), 16)
        self.condition = nn.Sequential(nn.Linear(3 * int(s.NUM_BANDS), hidden), nn.GELU(), nn.Linear(hidden, self.heads + 3 * self.heads))
        self.out = nn.Conv2d(self.inner, dim, 1, bias=False)
        self.ffn_a = SpatialChannelProjection(dim, 2 * dim)
        self.ffn_b = SpatialChannelProjection(dim, 2 * dim)
        self.ffn_out = nn.Conv2d(2 * dim, dim, 1, bias=False)
        self.norm = nn.GroupNorm(1, dim)
        self.last_source_weights = None
        self.last_source_probabilities = None
        self.last_temperature = None

    @staticmethod
    def band_descriptor(x, bands):
        spectrum = torch.fft.rfft2(x.float(), norm="ortho").abs().mean(1)
        h, w = spectrum.shape[-2:]
        fy = torch.fft.fftfreq(h, device=x.device).abs()[:, None]
        fx = torch.fft.rfftfreq((w - 1) * 2, device=x.device).abs()[None, :]
        radius = torch.sqrt(fy.square() + fx.square()); radius /= radius.max().clamp_min(1e-6)
        bounds = torch.linspace(0, 1, bands + 1, device=x.device)
        values = []
        for index in range(bands):
            mask = (radius >= bounds[index]) & ((radius <= bounds[index+1]) if index+1 == bands else (radius < bounds[index+1]))
            weight = mask.to(spectrum.dtype)
            values.append(
                (spectrum * weight[None]).sum((-2, -1))
                / weight.sum().clamp_min(1.0)
            )
        return torch.stack(values, -1)

    def forward(self, maps, bands):
        # maps: B x 3 modalities x D x H x W for one receptive-field scale.
        enhanced = torch.stack([self.rddb(maps[:, m]) for m in range(3)], 1)
        q = torch.stack([self.q(enhanced[:, m]) for m in range(3)], 1)
        k = torch.stack([self.k(enhanced[:, m]) for m in range(3)], 1)
        v = torch.stack([self.v(enhanced[:, m]) for m in range(3)], 1)
        descriptor = torch.stack([self.band_descriptor(enhanced[:, m], bands) for m in range(3)], 1)
        outputs, source_stats, temp_stats, source_probabilities = [], [], [], []
        for target in range(3):
            others = [x for x in range(3) if x != target]
            condition = torch.cat((descriptor[:, target]-descriptor[:, others[0]], descriptor[:, target]-descriptor[:, others[1]], descriptor[:, others[0]]-descriptor[:, others[1]]), -1)
            parameters = self.condition(condition)
            temperature = F.softplus(parameters[:, :self.heads]) + 0.05
            source_weight = parameters[:, self.heads:].reshape(-1, self.heads, 3).softmax(-1)
            qt = q[:, target].reshape(-1, self.heads, self.head_dim, q.shape[-2]*q.shape[-1]); qt = F.normalize(qt, dim=-1)
            mixed = 0.0
            for source in range(3):
                ks = k[:, source].reshape(-1, self.heads, self.head_dim, k.shape[-2]*k.shape[-1]); ks = F.normalize(ks, dim=-1)
                vs = v[:, source].reshape(-1, self.heads, self.head_dim, v.shape[-2]*v.shape[-1])
                attention = torch.matmul(qt, ks.transpose(-1, -2)) / temperature[:, :, None, None]
                mixed = mixed + source_weight[:, :, source, None, None] * torch.matmul(attention.softmax(-1), vs)
            mixed = mixed.reshape(-1, self.inner, maps.shape[-2], maps.shape[-1])
            y = enhanced[:, target] + self.out(mixed)
            y = y + self.ffn_out(F.gelu(self.ffn_a(self.norm(y))) * self.ffn_b(self.norm(y)))
            outputs.append(y); source_probabilities.append(source_weight)
            source_stats.append(source_weight.detach().mean(0)); temp_stats.append(temperature.detach().mean(0))
        self.last_source_weights = torch.stack(source_stats)
        self.last_source_probabilities = torch.stack(source_probabilities, dim=1)
        self.last_temperature = torch.stack(temp_stats)
        return torch.stack(outputs, 1)
