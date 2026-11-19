"""Pure-NumPy 8-node hexahedral (H8) orthotropic reference element and LUT builder."""
import numpy as np
from solid import Material, local_stiffness, transform


# Local corner bit order: a = lx + 2*ly + 4*lz for lx, ly, lz in {0, 1}.
HEX_LOCAL_OFFSETS = np.array(
    [[(a >> 0) & 1, (a >> 1) & 1, (a >> 2) & 1] for a in range(8)],
    dtype=np.int32,
)
HEX_PARITY_SIGNS = HEX_LOCAL_OFFSETS * 2 - 1  # -1 or +1


def _validate_pitch(dx, dy, dz):
    p = np.array([dx, dy, dz], dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Hex element dimensions (dx, dy, dz) must be positive finite numbers")
    return float(p[0]), float(p[1]), float(p[2])


def hex_B_at(xi, eta, zeta, dx, dy, dz):
    """Return 6x24 strain-displacement matrix at natural coordinates (xi, eta, zeta) in [-1, 1]^3."""
    dx, dy, dz = _validate_pitch(dx, dy, dz)
    B = np.zeros((6, 24), dtype=np.float64)
    sx = HEX_PARITY_SIGNS[:, 0]
    sy = HEX_PARITY_SIGNS[:, 1]
    sz = HEX_PARITY_SIGNS[:, 2]
    dN_dx = (sx / (4.0 * dx)) * (1.0 + sy * eta) * (1.0 + sz * zeta)
    dN_dy = (sy / (4.0 * dy)) * (1.0 + sx * xi) * (1.0 + sz * zeta)
    dN_dz = (sz / (4.0 * dz)) * (1.0 + sx * xi) * (1.0 + sy * eta)
    B[0, 0::3] = dN_dx
    B[1, 1::3] = dN_dy
    B[2, 2::3] = dN_dz
    B[3, 0::3] = dN_dy
    B[3, 1::3] = dN_dx
    B[4, 1::3] = dN_dz
    B[4, 2::3] = dN_dy
    B[5, 0::3] = dN_dz
    B[5, 2::3] = dN_dx
    return B


def hex_B_matrices(dx, dy, dz):
    """Return (B_gauss, B0) where B_gauss has shape (8, 6, 24) and B0 has shape (6, 24)."""
    dx, dy, dz = _validate_pitch(dx, dy, dz)
    gp = 1.0 / np.sqrt(3.0)
    coords = np.array(
        [
            [sx * gp, sy * gp, sz * gp]
            for sz in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sx in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    B_gauss = np.stack([hex_B_at(xi, eta, zeta, dx, dy, dz) for xi, eta, zeta in coords], axis=0)
    B0 = hex_B_at(0.0, 0.0, 0.0, dx, dy, dz)
    return B_gauss, B0


def strain_transform_batch(angles, build_axis="Z"):
    """Vectorized engineering-strain rotation matrices T of shape (N, 6, 6)."""
    if build_axis not in ("X", "Y", "Z"):
        raise ValueError("build_axis must be 'X', 'Y', or 'Z'")
    angles = np.asarray(angles, dtype=np.float64)
    scalar = angles.ndim == 0
    if scalar:
        angles = angles[None]
    c = np.cos(angles)
    s = np.sin(angles)
    T = np.zeros((len(angles), 6, 6), dtype=np.float64)
    T[:, 0, 0] = c * c
    T[:, 1, 1] = c * c
    T[:, 0, 1] = s * s
    T[:, 1, 0] = s * s
    T[:, 0, 3] = c * s
    T[:, 1, 3] = -c * s
    T[:, 2, 2] = 1.0
    T[:, 3, 0] = -2.0 * c * s
    T[:, 3, 1] = 2.0 * c * s
    T[:, 3, 3] = c * c - s * s
    T[:, 4, 4] = c
    T[:, 4, 5] = -s
    T[:, 5, 4] = s
    T[:, 5, 5] = c
    if build_axis != "Z":
        T = T @ transform(0.0, build_axis)
    return T[0] if scalar else T


def hex_element_stiffness(material, dx, dy, dz, angle_rad=0.0, build_axis="Z"):
    """Return exact 24x24 float64 stiffness matrix for one H8 element."""
    dx, dy, dz = _validate_pitch(dx, dy, dz)
    D_local = local_stiffness(material)
    B_gauss, _ = hex_B_matrices(dx, dy, dz)
    T = strain_transform_batch(float(angle_rad), build_axis)
    D_global = T.T @ D_local @ T
    detJ = (dx * dy * dz) / 8.0
    Ke = np.einsum("gki,kl,glj->ij", B_gauss, D_global, B_gauss) * detJ
    return 0.5 * (Ke + Ke.T)


def angle_to_bin(angle_rad, n_bins=360):
    """Map angle in radians (pi-periodic) to nearest integer bin in [0, n_bins)."""
    wrapped = np.mod(angle_rad, np.pi)
    bins = np.rint(wrapped * (n_bins / np.pi)).astype(np.int64) % n_bins
    return bins


def bin_to_angle(bins, n_bins=360):
    """Map integer bin indices back to radians in [0, pi)."""
    return np.asarray(bins, dtype=np.float64) * (np.pi / n_bins)


def build_stiffness_lut(material, dx, dy, dz, n_bins=360, build_axis="Z"):
    """Build pi-periodic 360-bin H8 stiffness LUT.

    Bin i corresponds to angle i * pi / n_bins so 0, 45, 90, 135 deg are exact.
    Returns (Ke_f64, Ke_f32, D_local_f64).
    """
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    dx, dy, dz = _validate_pitch(dx, dy, dz)
    D_local = local_stiffness(material)
    B_gauss, _ = hex_B_matrices(dx, dy, dz)
    angles = np.arange(n_bins, dtype=np.float64) * (np.pi / n_bins)
    T_all = strain_transform_batch(angles, build_axis)
    D_global = np.einsum("bki,kl,blj->bij", T_all, D_local, T_all)
    detJ = (dx * dy * dz) / 8.0
    Ke_f64 = np.einsum("gki,bkl,glj->bij", B_gauss, D_global, B_gauss) * detJ
    Ke_f64 = 0.5 * (Ke_f64 + Ke_f64.transpose(0, 2, 1))
    Ke_f32 = np.ascontiguousarray(Ke_f64, dtype=np.float32)
    return np.ascontiguousarray(Ke_f64, dtype=np.float64), Ke_f32, np.ascontiguousarray(D_local, dtype=np.float64)


def hex_centroid_stress(u_e, D_local, B0, angles, build_axis="Z"):
    """Compute centroid local and global stresses for a batch of H8 elements.

    Args:
        u_e: (N, 24) element nodal displacement vectors.
        D_local: (6, 6) local orthotropic stiffness matrix.
        B0: (6, 24) centroid strain-displacement matrix.
        angles: (N,) element orientation angles in radians.
        build_axis: 'Z', 'X', or 'Y'.

    Returns:
        local_stress: (N, 6) [s_along, s_trans, s_layer, t_12, t_23, t_13]
        global_stress: (N, 6) [sxx, syy, szz, txy, tyz, txz]
    """
    u_e = np.asarray(u_e, dtype=np.float64)
    eps_global = u_e @ B0.T
    T = strain_transform_batch(angles, build_axis)
    eps_local = np.einsum("eij,ej->ei", T, eps_global)
    local_stress = eps_local @ D_local.T
    global_stress = np.einsum("eji,ej->ei", T, local_stress)
    return local_stress, global_stress
