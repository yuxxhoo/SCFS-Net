"""Matrix-valued restrictions on the modality–scale hypergraph."""
import torch
from torch import nn
import torch.nn.functional as F

def semantic_incidence():
    h = torch.zeros(9, 6, dtype=torch.bool)
    for modality in range(3):
        h[3 * modality:3 * modality + 3, modality] = True
    for scale in range(3):
        h[scale::3, 3 + scale] = True
    return h

class MHSR(nn.Module):
    """Matrix-Valued Hypergraph Sheaf Recomposition (MHSR)."""

    def __init__(self, bands, stalk_dim, tau, eps):
        super().__init__()
        self.bands, self.d, self.tau, self.eps = bands, stalk_dim, tau, eps
        if stalk_dim < 2 or tau <= 0:
            raise ValueError("require STALK_DIM>=2 and DIFFUSION_TAU>0")
        incidence = semantic_incidence()
        self.register_buffer("incidence", incidence, persistent=True)
        pairs = [(int(v), int(e)) for v, e in incidence.nonzero().tolist()]
        self.register_buffer("incidence_pairs", torch.tensor(pairs), persistent=True)
        eye = torch.eye(stalk_dim)
        self.node_basis_raw = nn.Parameter(eye.repeat(bands, 9, 1, 1) + 0.01 * torch.randn(bands, 9, stalk_dim, stalk_dim))
        self.edge_basis_raw = nn.Parameter(eye.repeat(bands, 6, 1, 1) + 0.01 * torch.randn(bands, 6, stalk_dim, stalk_dim))
        self.diagonal_logits = nn.Parameter(torch.zeros(bands, len(pairs), stalk_dim))
        self.edge_logits = nn.Parameter(torch.zeros(bands, 6))
        self.last_energy = None
        self.last_solve_residual = None
        self.last_holonomy = None

    @staticmethod
    def _orthogonal(raw):
        # Cayley parameterization is exactly orthogonal and has stable
        # derivatives at the identity, unlike square QR with repeated singular
        # structure. Q=(I-S/2)(I+S/2)^-1, S=A-A^T.
        skew = raw.float() - raw.float().transpose(-1, -2)
        identity = torch.eye(raw.shape[-1], device=raw.device, dtype=torch.float32)
        left = identity - 0.5 * skew
        right = identity + 0.5 * skew
        return torch.linalg.solve(right.transpose(-1, -2), left.transpose(-1, -2)).transpose(-1, -2)

    def restrictions(self):
        vq, eq = self._orthogonal(self.node_basis_raw), self._orthogonal(self.edge_basis_raw)
        restrictions = self.node_basis_raw.new_zeros(self.bands, 9, 6, self.d, self.d, dtype=torch.float32)
        diag = F.softplus(self.diagonal_logits.float()) + 0.5
        for index, pair in enumerate(self.incidence_pairs.tolist()):
            v, e = pair
            restrictions[:, v, e] = eq[:, e] @ torch.diag_embed(diag[:, index]) @ vq[:, v].transpose(-1, -2)
        return restrictions

    def differential(self, restrictions, band):
        # Two independent differences per 3-node hyperedge: 12d x 9d.
        delta = restrictions.new_zeros(12 * self.d, 9 * self.d)
        row = 0
        for edge in range(6):
            vertices = self.incidence[:, edge].nonzero().flatten().tolist()
            anchor = vertices[0]
            weight = torch.sqrt(F.softplus(self.edge_logits[band, edge].float()) + self.eps)
            for vertex in vertices[1:]:
                rs = slice(row * self.d, (row + 1) * self.d)
                delta[rs, anchor * self.d:(anchor + 1) * self.d] = weight * restrictions[band, anchor, edge]
                delta[rs, vertex * self.d:(vertex + 1) * self.d] = -weight * restrictions[band, vertex, edge]
                row += 1
        return delta

    def laplacians(self, restrictions=None):
        restrictions = self.restrictions() if restrictions is None else restrictions
        return torch.stack([self.differential(restrictions, k).T @ self.differential(restrictions, k) for k in range(self.bands)])

    def transport(self, restrictions, band, source, target, edge):
        return torch.linalg.solve(restrictions[band, target, edge], restrictions[band, source, edge])

    def holonomy_loss(self, states, restrictions):
        # Non-trivial rectangles alternate a scale hyperedge and a modality
        # hyperedge: (m,s)->(n,s)->(n,t)->(m,t)->(m,s).
        losses = []
        for k in range(self.bands):
            for m, n in ((0, 1), (1, 2), (2, 0)):
                for s, t in ((0, 1), (1, 2), (2, 0)):
                    a, b, c, d = 3*m+s, 3*n+s, 3*n+t, 3*m+t
                    loop = (
                        self.transport(restrictions, k, d, a, m)
                        @ self.transport(restrictions, k, c, d, m)
                        @ self.transport(restrictions, k, b, c, n)
                        @ self.transport(restrictions, k, a, b, 3+s)
                    )
                    reverse = (
                        self.transport(restrictions, k, b, a, 3+s)
                        @ self.transport(restrictions, k, c, b, n)
                        @ self.transport(restrictions, k, d, c, 3+t)
                        @ self.transport(restrictions, k, a, d, m)
                    )
                    x = states[:, k, a].float()
                    losses.extend(((x @ loop.T - x).square().mean(), (x @ reverse.T - x).square().mean()))
        return torch.stack(losses).mean()

    def forward(self, states):
        restrictions = self.restrictions()
        laplacian = self.laplacians(restrictions)
        flat = states.float().reshape(states.shape[0], self.bands, 9 * self.d)
        identity = torch.eye(9 * self.d, device=flat.device, dtype=flat.dtype)
        system = identity[None] + self.tau * laplacian
        solved = torch.linalg.solve(system[None], flat.unsqueeze(-1)).squeeze(-1)
        residual = torch.einsum("kij,bkj->bki", system, solved) - flat
        energy = torch.einsum("bki,kij,bkj->bk", solved, laplacian, solved).mean()
        holonomy = self.holonomy_loss(states, restrictions)
        self.last_energy, self.last_holonomy = energy.detach(), holonomy.detach()
        self.last_solve_residual = residual.square().mean().sqrt().detach()
        return solved.reshape(states.shape), energy, holonomy
