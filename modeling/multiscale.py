"""Shared spatial scale states and frequency-conditioned scale routing."""
import torch
from torch import nn
import torch.nn.functional as F


def _grid_size(cfg):
    return (cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0],
            cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1])

class SharedMultiScaleLocalAttention(nn.Module):
    """Fuse three receptive fields with sample-dependent channel attention.

    The output projection is zero initialized.  Consequently this serial block
    is an exact identity at initialization while its output projection receives
    gradients immediately.  RGB/NIR/TIR use the same module and parameters.
    """

    def __init__(self, dim, cfg):
        super().__init__()
        settings = cfg.MODEL.MULTISCALE_LOCAL_ATTENTION
        self.height, self.width = _grid_size(cfg)
        dilations = tuple(int(value) for value in settings.DILATIONS)
        if len(dilations) < 2 or any(value < 1 for value in dilations):
            raise ValueError("DILATIONS must contain at least two positive values")
        self.num_scales = len(dilations)
        self.pre_norm = nn.LayerNorm(dim)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(dim),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                )
                for dilation in dilations
            ]
        )
        hidden = max(dim // int(settings.REDUCTION), 16)
        self.scale_attention = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_scales * dim),
        )
        self.output = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(settings.DROPOUT)
        nn.init.zeros_(self.output.weight)
        self.last_scale_weights = None

    def _routing_bias(self, grid):
        """Optional sample-wise prior over scales (B, S, 1)."""
        return None

    def forward_with_scales(self, tokens):
        """Return the normal output and the unfused scale feature maps.

        Keeping this operation in one method prevents Stage 3 from recomputing
        BatchNorm branches and makes the Stage-2 path numerically unchanged.
        """
        batch, length, dim = tokens.shape
        expected = self.height * self.width
        if length != expected:
            raise ValueError(
                "Token number {} does not match configured grid {}x{} ({}).".format(
                    length, self.height, self.width, expected
                )
            )
        normalized = self.pre_norm(tokens)
        grid = normalized.transpose(1, 2).reshape(
            batch, dim, self.height, self.width
        )
        scales = torch.stack([branch(grid) for branch in self.branches], dim=1)
        context = scales.mean(dim=(1, 3, 4))
        logits = self.scale_attention(context).reshape(
            batch, self.num_scales, dim
        )
        routing_bias = self._routing_bias(grid)
        if routing_bias is not None:
            logits = logits + routing_bias
        weights = logits.softmax(dim=1)
        fused = (scales * weights[:, :, :, None, None]).sum(dim=1)
        correction = self.output(fused).flatten(2).transpose(1, 2)
        self.last_scale_weights = weights.detach().mean(dim=(0, 2))
        return tokens + self.dropout(correction), scales, weights

    def forward(self, tokens):
        output, _, _ = self.forward_with_scales(tokens)
        return output

class FrequencyGuidedMultiScaleLocalAttention(SharedMultiScaleLocalAttention):
    """Use spectral-band energy to route, but never alter, scale features."""

    def __init__(self, dim, cfg):
        super().__init__(dim, cfg)
        settings = cfg.MODEL.FREQUENCY_SCALE_ROUTING
        self.max_logit_bias = float(settings.MAX_LOGIT_BIAS)
        if not 0.0 < self.max_logit_bias <= 1.0:
            raise ValueError("MAX_LOGIT_BIAS must be in (0, 1]")
        # Keep the classifier RNG stream independent of routing initialization.
        with torch.random.fork_rng(devices=[]):
            self.frequency_router = nn.Sequential(
                nn.Linear(self.num_scales, int(settings.HIDDEN_DIM)),
                nn.GELU(),
                nn.Linear(int(settings.HIDDEN_DIM), self.num_scales),
            )
            nn.init.zeros_(self.frequency_router[-1].weight)
            nn.init.zeros_(self.frequency_router[-1].bias)

        fy = torch.fft.fftfreq(self.height).abs()[:, None]
        fx = torch.fft.rfftfreq(self.width).abs()[None, :]
        radius = torch.sqrt(fy.square() + fx.square())
        radius = radius / radius.max().clamp_min(1e-6)
        boundaries = torch.linspace(0, 1, self.num_scales + 1)
        masks = []
        for index in range(self.num_scales):
            if index + 1 == self.num_scales:
                mask = (radius >= boundaries[index]) & (radius <= boundaries[index + 1])
            else:
                mask = (radius >= boundaries[index]) & (radius < boundaries[index + 1])
            masks.append(mask.float())
        self.register_buffer("frequency_masks", torch.stack(masks), persistent=False)
        self.last_band_energy = None
        self.last_frequency_bias = None

    def _routing_bias(self, grid):
        # FFT is a routing statistic only. Computation in fp32 also makes this
        # safe under mixed precision; gradients still reach the input tokens.
        spectrum = torch.fft.rfft2(grid.float(), norm="ortho")
        power = spectrum.abs().square().mean(dim=1)
        masks = self.frequency_masks.to(device=power.device, dtype=power.dtype)
        energy = (power[:, None] * masks[None]).sum(dim=(-2, -1))
        energy = energy / masks.sum(dim=(-2, -1)).clamp_min(1.0)[None]
        energy = torch.log1p(energy)
        energy = (energy - energy.mean(dim=-1, keepdim=True)) / (
            energy.std(dim=-1, keepdim=True, unbiased=False) + 1e-6
        )
        bias = (
            self.max_logit_bias * torch.tanh(self.frequency_router(energy))
        ).unsqueeze(-1)
        self.last_band_energy = energy.detach().mean(dim=0)
        self.last_frequency_bias = bias.detach().mean(dim=(0, 2))
        return bias.to(dtype=grid.dtype)
