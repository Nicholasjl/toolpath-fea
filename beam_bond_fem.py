"""Pure-NumPy 6-DOF 3D Timoshenko beam and eccentric 6-DOF interface coupler elements."""
import math
import numpy as np
from solid import Material, local_stiffness


def _beam_material_constants(material: Material):
    """Validate material and extract beam/bond elastic constants."""
    local_stiffness(material)
    e_beam = float(material.e1)
    g_xy = float(material.g12)
    g_xz = float(material.g13)
    g_t = math.sqrt(g_xy * g_xz)
    return e_beam, g_xy, g_xz, g_t


def _section_properties(w, h):
    """Rectangular road cross-section (A, I2_out_of_plane, I3_in_plane, J_torsion)."""
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    A = w * h
    I2 = w * (h ** 3) / 12.0  # bending about local e2 (deflects along Z)
    I3 = h * (w ** 3) / 12.0  # bending about local e3 (deflects in XY)
    a = np.maximum(w, h)
    b = np.minimum(w, h)
    ratio = b / np.maximum(a, 1e-12)
    J = a * (b ** 3) * (1.0 / 3.0 - 0.21 * ratio * (1.0 - (ratio ** 4) / 12.0))
    return A, I2, I3, J


def _beam_local_frames(p0, p1):
    """Orthonormal right-handed local frame R = [e1, e2, e3] (N, 3, 3) and lengths L (N,)."""
    d = p1 - p0
    L = np.linalg.norm(d, axis=1)
    if np.any(L <= 1e-12):
        raise ValueError("Zero-length beam segment encountered")
    e1 = d / L[:, None]
    ref = np.zeros_like(e1)
    vert = np.abs(e1[:, 2]) > 0.99
    ref[~vert, 2] = 1.0
    ref[vert, 1] = 1.0
    e2 = np.cross(ref, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    e3 = np.cross(e1, e2)
    R = np.stack([e1, e2, e3], axis=2)  # columns are e1, e2, e3
    return R, L


def _timoshenko_local_stiffness_batch(L, w, h, material: Material):
    """Return local 12x12 Timoshenko stiffness matrices K_loc (N, 12, 12)."""
    n = len(L)
    E, G_xy, G_xz, G_t = _beam_material_constants(material)
    A, I2, I3, J = _section_properties(w, h)
    kappa = 5.0 / 6.0

    K = np.zeros((n, 12, 12), dtype=np.float64)
    # Axial (u1: 0, 6)
    k_ax = E * A / L
    K[:, 0, 0] = k_ax
    K[:, 6, 6] = k_ax
    K[:, 0, 6] = -k_ax
    K[:, 6, 0] = -k_ax

    # Torsion (th1: 3, 9)
    k_tor = G_t * J / L
    K[:, 3, 3] = k_tor
    K[:, 9, 9] = k_tor
    K[:, 3, 9] = -k_tor
    K[:, 9, 3] = -k_tor

    # Bending in (e1, e2) plane: u2 (1, 7), th3 (5, 11), du2/dx1 = +th3
    phi_y = (12.0 * E * I3) / (kappa * G_xy * A * (L ** 2))
    denom_y = (1.0 + phi_y) * (L ** 3)
    ky_uu = 12.0 * E * I3 / denom_y
    ky_ut = 6.0 * E * I3 * L / denom_y
    ky_tt1 = (4.0 + phi_y) * E * I3 * (L ** 2) / denom_y
    ky_tt2 = (2.0 - phi_y) * E * I3 * (L ** 2) / denom_y
    K[:, 1, 1] = ky_uu
    K[:, 7, 7] = ky_uu
    K[:, 1, 7] = -ky_uu
    K[:, 7, 1] = -ky_uu
    K[:, 1, 5] = ky_ut
    K[:, 5, 1] = ky_ut
    K[:, 1, 11] = ky_ut
    K[:, 11, 1] = ky_ut
    K[:, 7, 5] = -ky_ut
    K[:, 5, 7] = -ky_ut
    K[:, 7, 11] = -ky_ut
    K[:, 11, 7] = -ky_ut
    K[:, 5, 5] = ky_tt1
    K[:, 11, 11] = ky_tt1
    K[:, 5, 11] = ky_tt2
    K[:, 11, 5] = ky_tt2

    # Bending in (e1, e3) plane: u3 (2, 8), th2 (4, 10), du3/dx1 = -th2
    phi_z = (12.0 * E * I2) / (kappa * G_xz * A * (L ** 2))
    denom_z = (1.0 + phi_z) * (L ** 3)
    kz_uu = 12.0 * E * I2 / denom_z
    kz_ut = 6.0 * E * I2 * L / denom_z
    kz_tt1 = (4.0 + phi_z) * E * I2 * (L ** 2) / denom_z
    kz_tt2 = (2.0 - phi_z) * E * I2 * (L ** 2) / denom_z
    K[:, 2, 2] = kz_uu
    K[:, 8, 8] = kz_uu
    K[:, 2, 8] = -kz_uu
    K[:, 8, 2] = -kz_uu
    K[:, 2, 4] = -kz_ut
    K[:, 4, 2] = -kz_ut
    K[:, 2, 10] = -kz_ut
    K[:, 10, 2] = -kz_ut
    K[:, 8, 4] = kz_ut
    K[:, 4, 8] = kz_ut
    K[:, 8, 10] = kz_ut
    K[:, 10, 8] = kz_ut
    K[:, 4, 4] = kz_tt1
    K[:, 10, 10] = kz_tt1
    K[:, 4, 10] = kz_tt2
    K[:, 10, 4] = kz_tt2
    return K


def _rotate_12x12_batch(R, K_loc):
    """Apply K_glob = block_diag(R,R,R,R) @ K_loc @ block_diag(R,R,R,R).T."""
    n = len(R)
    T = np.zeros((n, 12, 12), dtype=np.float64)
    for b in range(4):
        s = b * 3
        T[:, s : s + 3, s : s + 3] = R
    Kg = T @ K_loc @ T.transpose(0, 2, 1)
    return 0.5 * (Kg + Kg.transpose(0, 2, 1))


def timoshenko_beam_stiffness_batch(p0, p1, w, h, material: Material, scale=None):
    """Compute global 12x12 Timoshenko stiffness matrices for a batch of 3D beams."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    w = np.broadcast_to(np.asarray(w, dtype=np.float64), (len(p0),))
    h = np.broadcast_to(np.asarray(h, dtype=np.float64), (len(p0),))
    R, L = _beam_local_frames(p0, p1)
    K_loc = _timoshenko_local_stiffness_batch(L, w, h, material)
    if scale is not None:
        sc = np.asarray(scale, dtype=np.float64)
        K_loc = K_loc * sc[:, None, None]
    return _rotate_12x12_batch(R, K_loc)


def _skew_batch(r):
    """Skew-symmetric matrices [r]_x of shape (N, 3, 3) such that [r]_x @ v = r x v."""
    S = np.zeros((len(r), 3, 3), dtype=np.float64)
    S[:, 0, 1] = -r[:, 2]
    S[:, 0, 2] = r[:, 1]
    S[:, 1, 0] = r[:, 2]
    S[:, 1, 2] = -r[:, 0]
    S[:, 2, 0] = -r[:, 1]
    S[:, 2, 1] = r[:, 0]
    return S


def _bond_kinematics_batch(p_a, p_b, p_c, normal, t1, t2):
    """Build local-jump 6x12 matrices B_bond (N, 6, 12) for eccentric couplers."""
    n = len(p_a)
    Q = np.stack([normal, t1, t2], axis=1)  # rows: normal, t1, t2 -> Q @ v_glob = v_loc
    ra = p_c - p_a
    rb = p_c - p_b
    Sa = _skew_batch(ra)
    Sb = _skew_batch(rb)

    B = np.zeros((n, 6, 12), dtype=np.float64)
    B[:, 0:3, 0:3] = -Q
    B[:, 0:3, 3:6] = Q @ Sa
    B[:, 0:3, 6:9] = Q
    B[:, 0:3, 9:12] = -Q @ Sb
    B[:, 3:6, 3:6] = -Q
    B[:, 3:6, 9:12] = Q
    return B


def eccentric_bond_stiffness_batch(
    p_a,
    p_b,
    p_c,
    normal,
    t1,
    t2,
    area,
    l1,
    l2,
    d_eff,
    e_bond,
    g_bond1,
    g_bond2,
    factor=None,
):
    """Compute global 12x12 stiffness matrices for eccentric 6-DOF interface couplers."""
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    p_c = np.asarray(p_c, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    d_eff = np.maximum(np.asarray(d_eff, dtype=np.float64), 1e-6)
    e_b = np.asarray(e_bond, dtype=np.float64)
    g1 = np.asarray(g_bond1, dtype=np.float64)
    g2 = np.asarray(g_bond2, dtype=np.float64)
    fac = np.ones(len(p_a), dtype=np.float64) if factor is None else np.asarray(factor, dtype=np.float64)

    B = _bond_kinematics_batch(p_a, p_b, p_c, normal, t1, t2)
    g_avg = 0.5 * (g1 + g2)
    diag_d = np.column_stack(
        [
            e_b * area / d_eff,
            g1 * area / d_eff,
            g2 * area / d_eff,
            g_avg * area * (l1 ** 2 + l2 ** 2) / (12.0 * d_eff),
            e_b * area * (l2 ** 2) / (12.0 * d_eff),
            e_b * area * (l1 ** 2) / (12.0 * d_eff),
        ]
    ) * fac[:, None]

    # K = B^T @ diag(diag_d) @ B
    BtD = B.transpose(0, 2, 1) * diag_d[:, None, :]
    K = BtD @ B
    return 0.5 * (K + K.transpose(0, 2, 1))


def beam_internal_stress_batch(p0, p1, w, h, material: Material, u_e, scale=None):
    """Recover axial, bending, shear, and max fiber von Mises stresses (MPa) per beam."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    w = np.broadcast_to(np.asarray(w, dtype=np.float64), (len(p0),))
    h = np.broadcast_to(np.asarray(h, dtype=np.float64), (len(p0),))
    u_e = np.asarray(u_e, dtype=np.float64)
    R, L = _beam_local_frames(p0, p1)
    K_loc = _timoshenko_local_stiffness_batch(L, w, h, material)
    if scale is not None:
        K_loc = K_loc * np.asarray(scale, dtype=np.float64)[:, None, None]

    T = np.zeros((len(R), 12, 12), dtype=np.float64)
    for b in range(4):
        s = b * 3
        T[:, s : s + 3, s : s + 3] = R
    u_loc = np.einsum("eji,ej->ei", T, u_e)
    f_loc = np.einsum("eij,ej->ei", K_loc, u_loc)

    E, _, _, _ = _beam_material_constants(material)
    A, I2, I3, _ = _section_properties(w, h)
    sc = np.ones(len(p0), dtype=np.float64) if scale is None else np.asarray(scale, dtype=np.float64)

    sigma_axial = sc * E * (u_loc[:, 6] - u_loc[:, 0]) / L
    m2_max = np.maximum(np.abs(f_loc[:, 4]), np.abs(f_loc[:, 10]))
    m3_max = np.maximum(np.abs(f_loc[:, 5]), np.abs(f_loc[:, 11]))
    sigma_bend_z = m2_max * (0.5 * h) / I2
    sigma_bend_xy = m3_max * (0.5 * w) / I3
    sigma_fiber_max = np.abs(sigma_axial) + sigma_bend_z + sigma_bend_xy
    tau_xy = 1.5 * np.abs(f_loc[:, 1]) / A
    tau_xz = 1.5 * np.abs(f_loc[:, 2]) / A
    vm_max = np.sqrt(sigma_fiber_max ** 2 + 3.0 * (tau_xy ** 2 + tau_xz ** 2))
    return {
        "sigma_axial": sigma_axial,
        "sigma_bend_z": sigma_bend_z,
        "sigma_bend_xy": sigma_bend_xy,
        "sigma_fiber_max": sigma_fiber_max,
        "von_mises_max": vm_max,
    }


def bond_interface_stress_batch(
    p_a,
    p_b,
    p_c,
    normal,
    t1,
    t2,
    l1,
    l2,
    d_eff,
    e_bond,
    g_bond1,
    g_bond2,
    u_e,
    factor=None,
):
    """Recover normal peel/compression, shear, and peak edge stresses (MPa) on couplers."""
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    p_c = np.asarray(p_c, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    d_eff = np.maximum(np.asarray(d_eff, dtype=np.float64), 1e-6)
    e_b = np.asarray(e_bond, dtype=np.float64)
    g1 = np.asarray(g_bond1, dtype=np.float64)
    g2 = np.asarray(g_bond2, dtype=np.float64)
    u_e = np.asarray(u_e, dtype=np.float64)
    fac = np.ones(len(p_a), dtype=np.float64) if factor is None else np.asarray(factor, dtype=np.float64)

    B = _bond_kinematics_batch(p_a, p_b, p_c, normal, t1, t2)
    jump = np.einsum("eij,ej->ei", B, u_e)

    sigma_n = fac * (e_b / d_eff) * jump[:, 0]
    tau_1 = fac * (g1 / d_eff) * jump[:, 1]
    tau_2 = fac * (g2 / d_eff) * jump[:, 2]
    tau_mag = np.hypot(tau_1, tau_2)

    bend1 = np.abs(fac * (e_b / d_eff) * (0.5 * l2) * jump[:, 4])
    bend2 = np.abs(fac * (e_b / d_eff) * (0.5 * l1) * jump[:, 5])
    sigma_n_peak = sigma_n + bend1 + bend2
    equiv_traction = np.hypot(np.maximum(sigma_n, 0.0), tau_mag)
    equiv_peak_traction = np.hypot(np.maximum(sigma_n_peak, 0.0), tau_mag)
    return {
        "sigma_n": sigma_n,
        "sigma_n_peak": sigma_n_peak,
        "tau": tau_mag,
        "equiv_traction": equiv_traction,
        "equiv_peak_traction": equiv_peak_traction,
    }
