"""mm, N, MPa. CST plane stress with spatially rotated orthotropy."""
from dataclasses import dataclass
import re
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class Settings:
    length: float = 100
    height: float = 20
    thickness: float = 2
    nx: int = 60
    ny: int = 16
    e1: float = 2400
    e2: float = 1200
    g12: float = 600
    nu12: float = .3
    angle: float = 0
    fx: float = 0
    fy: float = -10


def constitutive(e1, e2, g, nu, angle):
    compliance = np.array([[1/e1, -nu/e1, 0], [-nu/e1, 1/e2, 0], [0, 0, 1/g]])
    if np.linalg.eigvalsh(compliance).min() <= 0:
        raise ValueError('材料柔度矩阵不是正定：请检查模量和泊松比')
    local = np.linalg.inv(compliance)
    c, s = np.cos(angle), np.sin(angle)
    # Engineering strain: [exx, eyy, 2exy]. Work conjugacy gives T.T C T.
    transform = np.array([[c*c, s*s, c*s], [s*s, c*c, -c*s], [-2*c*s, 2*c*s, c*c-s*s]])
    return transform.T @ local @ transform, local, transform


def _lines(path):
    with open(path, encoding='utf-8', errors='replace') as source:
        yield from source


def read_gcode(path):
    """Return extrusion segments [x0,y0,x1,y1,z]; never infer arc chords."""
    pos = dict(X=0., Y=0., Z=0., E=0.)
    absolute, e_absolute = True, True
    segments = []
    for line in _lines(path):
        line = re.sub(r'\([^)]*\)', '', line.split(';')[0]).upper()
        words = re.findall(r'([A-Z])\s*([-+]?(?:\d*\.\d+|\d+\.?\d*))', line)
        if not words:
            continue
        commands = [(k, int(float(v))) for k, v in words if k in ('G', 'M')]
        vals = {k: float(v) for k, v in words if k in pos}
        for kind, cmd in commands:
            if kind == 'M':
                if cmd == 82: e_absolute = True
                if cmd == 83: e_absolute = False
                continue
            if cmd == 20: raise ValueError('仅支持毫米 G-code (G21)，不支持英寸 G20')
            if cmd == 90: absolute = True; e_absolute = True
            if cmd == 91: absolute = False; e_absolute = False
            if cmd == 92: pos.update(vals)
            if cmd in (2, 3): raise ValueError('检测到圆弧 G2/G3，请在切片器中禁用圆弧拟合后重新导出')
            if cmd in (0, 1):
                old = pos.copy()
                for axis, value in vals.items():
                    pos[axis] = value if (e_absolute if axis == 'E' else absolute) else pos[axis] + value
                if pos['E'] > old['E'] + 1e-8 and np.hypot(pos['X']-old['X'], pos['Y']-old['Y']) > 1e-8:
                    if abs(pos['Z']-old['Z']) > 1e-5:
                        raise ValueError('检测到非平面挤出路径；此版本仅支持平面层')
                    segments.append([old['X'], old['Y'], pos['X'], pos['Y'], pos['Z']])
    if not segments: raise ValueError('未找到有效的 G0/G1 挤出线段')
    return np.asarray(segments)


def map_angles(centers, segments):
    """Exact nearest-segment direction in XY, bounded temporary memory."""
    if len(segments) > 50000: raise ValueError('单层线段超过 50000，请先裁剪目标零件路径')
    a = segments[:, :2]
    delta = segments[:, 2:4] - a
    squared = (delta*delta).sum(axis=1)
    angles, distances = [], []
    for chunk in np.array_split(centers, max(1, int(np.ceil(len(centers)/32)))):
        diff = chunk[:, None, :] - a
        t = np.clip((diff*delta).sum(axis=2)/squared, 0, 1)
        d2 = ((diff-t[..., None]*delta)**2).sum(axis=2)
        idx = d2.argmin(axis=1)
        angles.extend(np.arctan2(delta[idx, 1], delta[idx, 0]))
        distances.extend(np.sqrt(d2[np.arange(len(chunk)), idx]))
    return np.array(angles), np.array(distances)


def solve(cfg, segments=None):
    values = list(vars(cfg).values())
    if not np.isfinite(values).all(): raise ValueError('输入必须是有限数值')
    if min(cfg.length, cfg.height, cfg.thickness, cfg.e1, cfg.e2, cfg.g12) <= 0:
        raise ValueError('尺寸和模量必须大于零')
    if not (2 <= cfg.nx <= 160 and 2 <= cfg.ny <= 80):
        raise ValueError('网格范围：X 方向 2–160，Y 方向 2–80')
    x, y = np.meshgrid(np.linspace(0, cfg.length, cfg.nx+1), np.linspace(0, cfg.height, cfg.ny+1))
    nodes = np.column_stack([x.ravel(), y.ravel()])
    cells = []
    for j in range(cfg.ny):
        for i in range(cfg.nx):
            a = j*(cfg.nx+1)+i
            b, c, d = a+1, a+cfg.nx+2, a+cfg.nx+1
            cells.extend([[a,b,c], [a,c,d]])
    cells = np.array(cells)
    centers = nodes[cells].mean(axis=1)
    angles = np.full(len(cells), np.deg2rad(cfg.angle))
    distances = np.zeros(len(cells))
    if segments is not None: angles, distances = map_angles(centers, segments)
    rows, cols, data, records = [], [], [], []
    for cell, angle in zip(cells, angles):
        p = nodes[cell]
        area = np.linalg.det(np.column_stack([np.ones(3), p])) / 2
        b = np.roll(p[:,1], -1) - np.roll(p[:,1], -2)
        c = np.roll(p[:,0], -2) - np.roll(p[:,0], -1)
        B = np.zeros((3,6))
        B[0,0::2], B[1,1::2] = b, c
        B[2,0::2], B[2,1::2] = c, b
        B /= 2*area
        D, local, transform = constitutive(cfg.e1, cfg.e2, cfg.g12, cfg.nu12, angle)
        ke = cfg.thickness*area*(B.T @ D @ B)
        dofs = (cell[:,None]*2 + np.arange(2)).ravel()
        rows.extend(np.repeat(dofs,6)); cols.extend(np.tile(dofs,6)); data.extend(ke.ravel())
        records.append((B,D,local,transform,dofs))
    K = coo_matrix((data,(rows,cols)),shape=(nodes.size,nodes.size)).tocsr()
    f = np.zeros(nodes.size)
    right = np.flatnonzero(np.isclose(nodes[:,0],cfg.length))
    weights = np.ones(len(right)); weights[[0,-1]] = .5; weights /= weights.sum()
    f[2*right] = cfg.fx*weights; f[2*right+1] = cfg.fy*weights
    left = np.flatnonzero(np.isclose(nodes[:,0],0))
    fixed = (left[:,None]*2+np.arange(2)).ravel()
    free = np.setdiff1d(np.arange(nodes.size),fixed)
    u = np.zeros(nodes.size)
    u[free] = spsolve(K[free][:,free],f[free])
    if not np.isfinite(u).all(): raise ValueError('求解失败：矩阵奇异或病态')
    stress, local_stress = [], []
    for B,D,local,T,dofs in records:
        strain = B @ u[dofs]
        stress.append(D @ strain); local_stress.append(local @ T @ strain)
    stress, local_stress = np.array(stress), np.array(local_stress)
    sx, sy, tau = stress.T
    vm = np.sqrt(sx*sx - sx*sy + sy*sy + 3*tau*tau)
    residual = K@u-f
    relative_residual = np.linalg.norm(residual[free])/max(np.linalg.norm(f[free]),1e-20)
    return dict(nodes=nodes, cells=cells, displacement=u.reshape(-1,2), stress=stress,
                local_stress=local_stress, von_mises=vm, angles=np.rad2deg(angles),
                path_distance=distances, reaction=residual.reshape(-1,2)[left].sum(axis=0),
                relative_residual=float(relative_residual), energy=float(.5*u@f))
