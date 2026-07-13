"""Pure-geometry core for (theta, phi, log r) spherical anchor binning.

This module is the geometry / indexing backbone consumed by
``SphericalQueryHead`` (Step 3). It contains **no** ``nn.Module`` and **no**
learnable state -- only stateless torch tensor ops. Everything is fully
vectorised (no python loops over the number of anchors or tokens) and runs
unchanged on CPU or CUDA; every returned index tensor is ``torch.long``.

Coordinate convention (must match ``builders/common.py:xyz_to_theta_phi_r`` --
imported directly below so the two can never drift):

    r     = ||xyz||                      (clamped to >= 1e-6)
    phi   = atan2(y, x)   in [-pi, pi]
    theta = asin(z / r)   in [-pi/2, pi/2]

Bin grid
--------
The three axes are binned into a regular grid::

    i_theta = floor((theta + pi/2) / d_theta)          -> clamped to [0, n_theta-1]
    i_phi   = floor((phi   + pi)   / d_phi)  mod n_phi  -> [0, n_phi-1]  (+-pi seam safe)
    i_lr    = floor((ln r - ln r_min) / d_logr)         -> [0, n_lr-1]

    n_theta = ceil(pi        / d_theta)
    n_phi   = ceil(2 pi      / d_phi)
    n_lr    = ceil(ln(r_max/r_min) / d_logr)

Every ``idx3`` tensor in this module uses the column order ``[i_theta, i_phi,
i_lr]``. The scalar cell hash is::

    hash = (i_lr * n_theta + i_theta) * n_phi + i_phi

Data flow expected by Step 3
----------------------------
    bins = SphericalBins(dtheta_deg, dphi_deg, dlogr, r_min, r_max)

    # tokens -> cells
    tok_idx3, tok_valid = bins.bin_coords(token_xyz)          # (N,3), (N,)
    tok_hash            = bins.hash(tok_idx3)                  # (N,)  (mask by tok_valid)
    cell_hash, tok2cell = build_cells(tok_hash[tok_valid])    # (U,), (Nv,)

    # anchors = occupied cells; the current SphericalQueryHead background K/V
    # path uses only each anchor's own 1x1x1 cell.
    own_hash             = cell_hash[:, None]                 # (U,1)
    cell_idx, found      = lookup_cells(own_hash.reshape(-1), cell_hash)
    cell_idx = cell_idx.reshape(U, 1)
    found    = found.reshape(U, 1)
    anchor_cell_idx = torch.where(found, cell_idx, cell_idx.new_full((), -1))

    # flatten (anchor, token) membership
    anchor_ids, token_ids, slot_ids = gather_cell_members(
        anchor_cell_idx, tok2cell, cell_hash.numel())

``SphericalBins.neighbor_hashes`` remains available for geometry analysis or
future wider-context experiments; its self cell is slot ``SELF_SLOT`` = 4.
"""
from __future__ import annotations

import math

import torch

# Import the canonical spherical conversion so this module can never drift from
# the feature-builder convention. It is a pure torch function (no nn state).
from .builders.common import xyz_to_theta_phi_r

__all__ = [
    "SphericalBins",
    "build_cells",
    "lookup_cells",
    "gather_cell_members",
]

_HALF_PI = math.pi / 2.0
_TWO_PI = 2.0 * math.pi


class SphericalBins:
    """Regular ``(theta, phi, log r)`` bin grid. All indices are ``long``.

    Stateless value object: it stores only the (python scalar) grid resolution,
    so a single instance is safe to share across devices and threads. Every
    method accepts CPU or CUDA tensors and returns tensors on the input device.

    Column order for every ``idx3`` argument / return is ``[i_theta, i_phi,
    i_lr]``.
    """

    #: number of cells in the 3x3x1 neighbourhood
    N_NEIGHBORS: int = 9
    #: flat slot index of the centre (self) cell in ``neighbor_hashes`` output
    SELF_SLOT: int = 4

    def __init__(self, dtheta_deg: float, dphi_deg: float, dlogr: float,
                 r_min: float, r_max: float):
        assert dtheta_deg > 0 and dphi_deg > 0 and dlogr > 0, "bin steps must be > 0"
        assert 0.0 < r_min < r_max, "require 0 < r_min < r_max"

        self.dtheta = math.radians(float(dtheta_deg))   # radians
        self.dphi = math.radians(float(dphi_deg))       # radians
        self.dlogr = float(dlogr)                       # log-ratio units
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.ln_r_min = math.log(self.r_min)

        self.n_theta = int(math.ceil(math.pi / self.dtheta))
        self.n_phi = int(math.ceil(_TWO_PI / self.dphi))
        self.n_lr = int(math.ceil(math.log(self.r_max / self.r_min) / self.dlogr))
        assert self.n_phi >= 3, (
            "n_phi < 3 makes the +-1 phi neighbours alias under wrap; "
            "use a smaller dphi_deg")

        # (9, 3) neighbour offset table, column order [d_theta, d_phi, d_lr].
        # Built with 'ij' meshgrid + C-order reshape so the flat slot index is
        #   slot = (d_theta+1)*3 + (d_phi+1)
        # hence the self cell (0,0,0) lands at slot 4 (== SELF_SLOT).
        a = torch.tensor([-1, 0, 1], dtype=torch.long)
        r0 = torch.tensor([0], dtype=torch.long)
        dt, dp, dl = torch.meshgrid(a, a, r0, indexing="ij")
        self._offsets_cpu = torch.stack(
            [dt.reshape(-1), dp.reshape(-1), dl.reshape(-1)], dim=1)  # (9,3) long

    # ------------------------------------------------------------------ dims
    @property
    def num_cells(self) -> int:
        """Total number of hashable cells (dense upper bound on the hash)."""
        return self.n_theta * self.n_phi * self.n_lr

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"SphericalBins(n_theta={self.n_theta}, n_phi={self.n_phi}, "
                f"n_lr={self.n_lr}, r=[{self.r_min},{self.r_max}))")

    # ------------------------------------------------------------- binning
    def bin_coords(self, xyz):
        """``(N,3)`` sensor-frame xyz -> ``(idx3 (N,3) long, valid (N,) bool)``.

        ``idx3`` columns are ``[i_theta, i_phi, i_lr]``. ``valid`` is
        ``(r_min <= r) & (r < r_max)``; rows where ``valid`` is False have their
        ``idx3`` zeroed (callers must gate on ``valid``, not on the index value).
        i_theta is clamped to ``[0, n_theta-1]`` (poles), i_phi is wrapped
        ``mod n_phi`` (+-pi seam), i_lr is clamped to ``[0, n_lr-1]``.
        """
        tpr = xyz_to_theta_phi_r(xyz)            # canonical convention; r>=1e-6
        theta = tpr[..., 0]
        phi = tpr[..., 1]
        r = tpr[..., 2]

        i_theta = torch.floor((theta + _HALF_PI) / self.dtheta).to(torch.long)
        i_theta = i_theta.clamp(0, self.n_theta - 1)

        i_phi = torch.floor((phi + math.pi) / self.dphi).to(torch.long)
        i_phi = torch.remainder(i_phi, self.n_phi)           # +-pi seam safe

        i_lr = torch.floor((torch.log(r) - self.ln_r_min) / self.dlogr).to(torch.long)
        i_lr = i_lr.clamp(0, self.n_lr - 1)

        idx3 = torch.stack([i_theta, i_phi, i_lr], dim=-1)   # (...,3)
        valid = (r >= self.r_min) & (r < self.r_max)
        idx3 = torch.where(valid.unsqueeze(-1), idx3, torch.zeros_like(idx3))
        return idx3, valid

    # ----------------------------------------------------------- hash / unhash
    def hash(self, idx3):
        """``(...,3)`` ``[i_theta,i_phi,i_lr]`` -> ``(...,)`` long cell hash.

        hash = ``(i_lr * n_theta + i_theta) * n_phi + i_phi``.
        """
        idx3 = idx3.to(torch.long)
        i_theta = idx3[..., 0]
        i_phi = idx3[..., 1]
        i_lr = idx3[..., 2]
        return (i_lr * self.n_theta + i_theta) * self.n_phi + i_phi

    def unhash(self, h):
        """Inverse of :meth:`hash`: ``(...,)`` long -> ``(...,3)``.

        (Only defined for non-negative hashes; the ``-1`` sentinel used for
        invalid neighbours must be filtered out before calling.)
        """
        h = h.to(torch.long)
        i_phi = torch.remainder(h, self.n_phi)
        rest = torch.div(h, self.n_phi, rounding_mode="floor")
        i_theta = torch.remainder(rest, self.n_theta)
        i_lr = torch.div(rest, self.n_theta, rounding_mode="floor")
        return torch.stack([i_theta, i_phi, i_lr], dim=-1)

    # ------------------------------------------------------------- neighbours
    def neighbor_hashes(self, idx3):
        """``(A,3)`` -> ``(nbr_hash (A,9) long, nbr_valid (A,9) bool)``.

        For each anchor cell, the 3x3x1 = 9 neighbours from every combination
        of ``{-1, 0, +1}`` offsets on ``(i_theta, i_phi)`` and a fixed
        ``i_lr`` offset of 0. This looks left/right in azimuth and up/down in
        elevation, but not the radial bin before/after.

        - ``i_phi`` is **wrapped** ``mod n_phi`` (azimuth is periodic), so phi
          neighbours are always geometrically valid.
        - ``i_theta`` is a **clamped** range: an offset that leaves
          ``[0, n_theta-1]`` yields ``nbr_valid=False`` for that slot (poles
          have fewer live neighbours). ``i_lr`` is unchanged by this stencil.

        Column (slot) order is fixed: ``slot = (d_theta+1)*3 + (d_phi+1)``, so
        the **centre / self cell (offset 0,0,0) is slot
        ``SELF_SLOT`` = 4**. Step 3 uses ``slot == 4`` to recover the
        centre-cell (P0) token subset.

        Invalid slots get ``nbr_hash = -1`` (a sentinel that never matches a
        real cell in :func:`lookup_cells`), so a downstream ``found`` result is
        already ``False`` there; ``nbr_valid`` is returned as well for callers
        that want the geometric mask explicitly.
        """
        idx3 = idx3.to(torch.long)
        offs = self._offsets_cpu.to(device=idx3.device)          # (9,3)

        i_theta = idx3[:, 0:1]                                    # (A,1)
        i_phi = idx3[:, 1:2]
        i_lr = idx3[:, 2:3]

        nt = i_theta + offs[:, 0].view(1, self.N_NEIGHBORS)       # (A,9)
        nl = i_lr + offs[:, 2].view(1, self.N_NEIGHBORS)
        nphi = torch.remainder(
            i_phi + offs[:, 1].view(1, self.N_NEIGHBORS), self.n_phi)

        theta_ok = (nt >= 0) & (nt < self.n_theta)
        lr_ok = (nl >= 0) & (nl < self.n_lr)
        nbr_valid = theta_ok & lr_ok                             # (A,9) bool

        nt_c = nt.clamp(0, self.n_theta - 1)
        nl_c = nl.clamp(0, self.n_lr - 1)
        h = (nl_c * self.n_theta + nt_c) * self.n_phi + nphi     # (A,9)
        nbr_hash = torch.where(nbr_valid, h, h.new_full((), -1))
        return nbr_hash, nbr_valid


# ----------------------------------------------------------------- sparse cells
def build_cells(hashes):
    """``(N,) long`` cell hashes -> ``(cell_hash (U,) long, token2cell (N,) long)``.

    ``cell_hash`` is the **sorted** unique set of hashes (the occupied cells);
    ``token2cell[i]`` is the index into ``cell_hash`` of token ``i``'s cell, i.e.
    ``token2cell in [0, U)``. Safe on empty input (returns two empty tensors).
    """
    hashes = hashes.to(torch.long)
    cell_hash, token2cell = torch.unique(
        hashes, sorted=True, return_inverse=True)
    return cell_hash, token2cell.to(torch.long)


def lookup_cells(query_hash, cell_hash):
    """Exact-match lookup of hashes against the **sorted** occupied-cell set.

    ``query_hash (Q,) long``, ``cell_hash (U,) long`` (sorted ascending, as
    returned by :func:`build_cells`) ->
    ``(cell_idx (Q,) long, found (Q,) bool)``. Where ``found`` is False (query
    absent, e.g. the ``-1`` neighbour sentinel) ``cell_idx`` is an arbitrary
    in-range value -- gate on ``found``. Safe on empty query / empty cell set.
    """
    query_hash = query_hash.to(torch.long)
    cell_hash = cell_hash.to(torch.long)
    U = cell_hash.numel()
    if U == 0:
        z = torch.zeros_like(query_hash)
        return z, torch.zeros_like(query_hash, dtype=torch.bool)
    pos = torch.searchsorted(cell_hash, query_hash)
    pos_c = pos.clamp(max=U - 1)
    found = (pos < U) & (cell_hash[pos_c] == query_hash)
    return pos_c, found


def gather_cell_members(anchor_cell_idx, token2cell, num_cells):
    """Flatten the (anchor, token) membership of every anchor's neighbour cells.

    Parameters
    ----------
    anchor_cell_idx : ``(A, S) long``
        Per-anchor, per-neighbour-slot cell index into the occupied-cell set,
        with ``-1`` for slots that are invalid / not found (as produced by
        masking :func:`lookup_cells` with ``nbr_valid``). Column order is the
        :meth:`SphericalBins.neighbor_hashes` slot order (self cell = slot 4
        for the default 3x3x1 stencil).
    token2cell : ``(N,) long``
        Token -> occupied-cell index (from :func:`build_cells`), in ``[0, num_cells)``.
    num_cells : int
        Number of occupied cells ``U`` (``= cell_hash.numel()``).

    Returns
    -------
    anchor_ids : ``(P,) long``
        Anchor index of each emitted pair. **Guaranteed ascending** (anchors are
        emitted in order 0..A-1).
    token_ids : ``(P,) long``
        Token index of each emitted pair (index into the original ``token2cell``
        / token array).
    slot_ids : ``(P,) long``
        Neighbour-slot (0..S-1) the pair came from.
        (Recommended extension over the bare ``(anchor_ids, token_ids)`` spec:
        lets Step 3 select the centre cell via ``slot_ids == SELF_SLOT``.)

    Ordering contract: pairs are grouped by anchor (ascending), and **within an
    anchor** by neighbour slot (ascending, i.e. the ``neighbor_hashes`` column
    order); the token order inside a single cell block is unspecified. Each
    valid (anchor, slot) cell contributes all of that cell's tokens as a
    contiguous block, so ``P = sum of cell sizes over all valid (anchor, slot)``.

    Fully vectorised (no python loop over anchors/tokens). Safe on empty inputs.
    """
    device = anchor_cell_idx.device
    anchor_cell_idx = anchor_cell_idx.to(torch.long)
    token2cell = token2cell.to(torch.long)
    A = anchor_cell_idx.shape[0]
    N = token2cell.shape[0]
    num_cells = int(num_cells)

    def _empty():
        e = torch.empty(0, dtype=torch.long, device=device)
        return e, e.clone(), e.clone()

    if A == 0 or N == 0 or num_cells == 0:
        return _empty()

    n_slot = anchor_cell_idx.shape[1]

    # --- CSR of tokens per cell: order groups tokens by ascending cell id ---
    order = torch.argsort(token2cell)                                # (N,)
    counts = torch.bincount(token2cell, minlength=num_cells)         # (U,)
    starts = torch.zeros(num_cells, dtype=torch.long, device=device)  # exclusive prefix
    if num_cells > 1:
        starts[1:] = torch.cumsum(counts, dim=0)[:-1]

    # --- flatten anchor grid in (anchor-major, slot-minor) order ---
    flat_cell = anchor_cell_idx.reshape(-1)                          # (A*S,)
    flat_anchor = (torch.arange(A, device=device)
                   .view(A, 1).expand(A, n_slot).reshape(-1))
    flat_slot = (torch.arange(n_slot, device=device)
                 .view(1, n_slot).expand(A, n_slot).reshape(-1))

    keep = flat_cell >= 0
    e_cell = flat_cell[keep]
    e_anchor = flat_anchor[keep]
    e_slot = flat_slot[keep]
    E = e_cell.shape[0]
    if E == 0:
        return _empty()

    e_count = counts[e_cell]                                         # (E,)
    e_start = starts[e_cell]                                         # (E,)

    # --- expand each entry into its contiguous block of tokens ---
    entry_rep = torch.repeat_interleave(
        torch.arange(E, device=device), e_count)                    # (P,)
    P = entry_rep.shape[0]
    if P == 0:
        return _empty()

    block_start = torch.zeros(E, dtype=torch.long, device=device)   # exclusive prefix
    if E > 1:
        block_start[1:] = torch.cumsum(e_count, dim=0)[:-1]
    within = torch.arange(P, device=device) - block_start[entry_rep]

    tok_pos = e_start[entry_rep] + within                           # index into `order`
    token_ids = order[tok_pos]
    anchor_ids = e_anchor[entry_rep]
    slot_ids = e_slot[entry_rep]
    return anchor_ids, token_ids, slot_ids
