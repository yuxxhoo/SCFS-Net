import torch
from torch import nn
import torch.nn.functional as F

from .fsca import FSCA
from .cafd import CAFD
from .mhsr import MHSR
from .multiscale import FrequencyGuidedMultiScaleLocalAttention

class SCFSFusion(nn.Module):
    """Scale-consolidated SCFS without repeated cross-modal attention.

    This implementation keeps all nine
    modality--scale nodes required by the matrix-valued sheaf, but estimates a
    single cross-modal correction from the dynamically consolidated scale
    representation.  The same correction is then written back to the three
    scale states before frequency decomposition and sheaf diffusion.

    All expensive spatial/frequency operations run in ``SCFS.SPATIAL_DIM``;
    only the final identity descriptor is projected back to the CLIP width.
    Unlike the inherited historical routes, this class does not register the
    unused adaptive fusion gates.
    """

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        settings = cfg.MODEL.SCFS
        latent_dim = int(settings.SPATIAL_DIM)
        if latent_dim <= 0 or latent_dim > dim:
            raise ValueError("SCFS SPATIAL_DIM must be in (0, backbone dimension]")

        self.latent_dim = latent_dim
        self.stalk_dim = int(settings.STALK_DIM)
        self.output_scale = float(settings.OUTPUT_SCALE)
        self.dec_weight = float(settings.DECORRELATION_WEIGHT)
        self.energy_weight = float(settings.SHEAF_ENERGY_WEIGHT)
        self.use_fsca = bool(settings.USE_FSCA)
        self.use_cafd_disentanglement = bool(
            settings.USE_CAFD_DISENTANGLEMENT
        )
        self.use_mhsr = bool(settings.USE_MHSR)

        self.input_norm = nn.LayerNorm(dim)
        self.input_down = nn.Linear(dim, latent_dim, bias=False)
        self.multiscale_local = FrequencyGuidedMultiScaleLocalAttention(
            latent_dim, cfg
        )
        self.fsca = FSCA(latent_dim, cfg)
        # Multi-scale depthwise branches already provide the spatial context
        # that RDDB duplicates in the original FSCA route.
        self.fsca.rddb = nn.Identity()
        self.cafd = CAFD(
            latent_dim, self.stalk_dim, cfg
        )
        self.mhsr = MHSR(
            2,
            self.stalk_dim,
            float(settings.DIFFUSION_TAU),
            float(settings.EPS),
        )
        self.compose = nn.Linear(2 * self.stalk_dim, latent_dim, bias=False)
        self.local_pool = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim), nn.GELU()
        )
        self.output_up = nn.Linear(latent_dim, dim, bias=False)
        self.output_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])

        self.auxiliary_loss = None
        self.last_correction_rms = None
        self.last_cross_modal_rms = None
        # Read-only diagnostics used by the paper visualizer.  They are
        # detached so enabling inspection cannot alter gradients or outputs.
        self.last_visualization = None

    def forward(
        self, rgb, nir, tir, rgb_global, nir_global, tir_global, label=None
    ):
        del label
        latent_tokens = [
            self.input_down(self.input_norm(tokens))
            for tokens in (rgb, nir, tir)
        ]

        routed_tokens, scale_maps, scale_weights = [], [], []
        for tokens in latent_tokens:
            routed, maps, weights = self.multiscale_local.forward_with_scales(tokens)
            routed_tokens.append(routed)
            scale_maps.append(maps)
            scale_weights.append(weights)

        # B x M x S x D x H x W.  Consolidate S before the expensive FSCA.
        scale_maps = torch.stack(scale_maps, dim=1)
        weights = torch.stack(scale_weights, dim=1)
        consensus = (scale_maps * weights[..., None, None]).sum(dim=2)
        if self.use_fsca:
            enhanced_consensus = self.fsca(consensus, self.cafd.bands)
        else:
            enhanced_consensus = consensus
        cross_modal_delta = enhanced_consensus - consensus

        # Preserve all three modality--scale states for the original 9-node
        # sheaf while sharing the cross-modal correction across scales.
        enhanced_scales = scale_maps + cross_modal_delta.unsqueeze(2)
        states, decorrelation, completeness = self.cafd(enhanced_scales)
        identity_states = states[:, :2]
        if not self.use_cafd_disentanglement:
            # Retain the same tensor dimensions but deliberately mix the
            # modality state back into both identity carriers.
            identity_states = identity_states + 0.5 * states[:, 2:3]
        identity = identity_states.reshape(
            states.shape[0], 2, 9, self.stalk_dim
        )
        if self.use_mhsr:
            recomposed, energy, _ = self.mhsr(identity)
            identity_increment = recomposed - identity
        else:
            # Direct identity writeback is the topology-free control.
            identity_increment = identity
            energy = identity.new_zeros(())
        delta = identity_increment.transpose(1, 2).reshape(
            states.shape[0], 3, 3, -1
        )
        correction = self.output_scale * self.compose(
            delta.to(enhanced_scales.dtype)
        )

        local_features = []
        for modality in range(3):
            update = correction[:, modality, :, :, None, None]
            update = (update * enhanced_scales[:, modality])
            update = (
                update * weights[:, modality, :, :, None, None]
            ).sum(dim=1)
            refined = routed_tokens[modality] + update.flatten(2).transpose(1, 2)
            pooled = self.local_pool(refined.mean(dim=1))
            local_features.append(self.output_up(pooled))

        features = [
            norm(global_feature + local_feature)
            for norm, global_feature, local_feature in zip(
                self.output_norms,
                (rgb_global, nir_global, tir_global),
                local_features,
            )
        ]
        self.auxiliary_loss = (
            self.dec_weight * decorrelation
            + self.energy_weight * energy
            + completeness
        )
        self.last_correction_rms = correction.detach().square().mean().sqrt()
        self.last_cross_modal_rms = (
            cross_modal_delta.detach().square().mean().sqrt()
        )
        self.last_visualization = {
            "scale_maps": scale_maps.detach(),
            "scale_weights": weights.detach(),
            "consensus": consensus.detach(),
            "enhanced_consensus": enhanced_consensus.detach(),
            "enhanced_scales": enhanced_scales.detach(),
            "states": states.detach(),
            "identity": identity.detach(),
            "recomposed": recomposed.detach() if self.use_mhsr else identity.detach(),
            "correction": correction.detach(),
        }
        return torch.cat(features, dim=-1)

    def get_auxiliary_loss(self):
        if self.auxiliary_loss is None:
            return next(self.parameters()).new_zeros(())
        return self.auxiliary_loss
