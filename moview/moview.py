from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from moview.fchk_parser import FCHKParser, FCHKWavefunction
from moview.molden_parser import MoldenParser

from moview.utils import BOHR_TO_ANG, HARTREE_TO_EV
from moview.utils import (
    CARTESIAN_POWERS, COVALENT_RADII, SPHERICAL_TO_CARTESIAN
)
from moview.utils import n_shell_functions
from moview.utils import Shell

# User-adjustable defaults. Set MOVIEW_CACHE_DIR to override CACHE_DIR without editing this file.
CACHE_DIR = Path(
    os.environ.get("MOVIEW_CACHE_DIR") or Path(tempfile.gettempdir()) / "fchk_orbital_viewer_cache"
).expanduser()
DEFAULT_PREFETCH_WORKERS = 12

(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))


def _odd_double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    out = 1
    for value in range(n, 0, -2):
        out *= value
    return out


def primitive_norm(lx: int, ly: int, lz: int, alpha: float) -> float:
    """Cartesian primitive Gaussian normalization."""
    lsum = lx + ly + lz
    prefactor = (2.0 * alpha / math.pi) ** 0.75
    numerator = (4.0 * alpha) ** lsum
    denom = (
        _odd_double_factorial(2 * lx - 1)
        * _odd_double_factorial(2 * ly - 1)
        * _odd_double_factorial(2 * lz - 1)
    )
    return prefactor * math.sqrt(numerator / denom)


def _primitive_cartesian_value(
    lx: int,
    ly: int,
    lz: int,
    alpha: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    exp_term: np.ndarray,
) -> np.ndarray:
    out = primitive_norm(lx, ly, lz, alpha) * exp_term
    if lx:
        out = out * (x**lx)
    if ly:
        out = out * (y**ly)
    if lz:
        out = out * (z**lz)
    return out


def _primitive_cartesian_value32(
    lx: int,
    ly: int,
    lz: int,
    alpha: np.float32,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    exp_term: np.ndarray,
) -> np.ndarray:
    out = np.float32(primitive_norm(lx, ly, lz, float(alpha))) * exp_term
    if lx:
        out = out * (x**lx)
    if ly:
        out = out * (y**ly)
    if lz:
        out = out * (z**lz)
    return out


def evaluate_shell(shell: Shell, points: np.ndarray) -> list[np.ndarray]:
    rel = points - shell.center
    x = rel[:, 0]
    y = rel[:, 1]
    z = rel[:, 2]
    r2 = x * x + y * y + z * z
    st = shell.shell_type

    if st == -1:
        comps = [np.zeros(points.shape[0], dtype=np.float64) for _ in range(4)]
        powers = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for pidx, alpha in enumerate(shell.exponents):
            exp_term = np.exp(-float(alpha) * r2)
            coeffs = [float(shell.coefficients[pidx])]
            p_coeff = (
                float(shell.sp_coefficients[pidx])
                if shell.sp_coefficients is not None
                else float(shell.coefficients[pidx])
            )
            coeffs.extend([p_coeff, p_coeff, p_coeff])
            for comp_idx, (lx, ly, lz) in enumerate(powers):
                comps[comp_idx] += coeffs[comp_idx] * _primitive_cartesian_value(
                    lx, ly, lz, float(alpha), x, y, z, exp_term
                )
        return comps

    angular_momentum = abs(st)
    if angular_momentum not in CARTESIAN_POWERS:
        raise NotImplementedError(f"Unsupported shell type {st}; only S through H shells are supported")
    powers = CARTESIAN_POWERS[angular_momentum]
    if st >= 0:
        matrix = np.eye(len(powers), dtype=np.float64)
    else:
        matrix = SPHERICAL_TO_CARTESIAN[angular_momentum]

    comps = [np.zeros(points.shape[0], dtype=np.float64) for _ in range(matrix.shape[1])]
    nonzero_by_cart = [np.flatnonzero(np.abs(matrix[row]) > 1.0e-14) for row in range(matrix.shape[0])]
    for pidx, alpha in enumerate(shell.exponents):
        alpha_f = float(alpha)
        coeff = float(shell.coefficients[pidx])
        exp_term = np.exp(-alpha_f * r2)
        for cart_idx, (lx, ly, lz) in enumerate(powers):
            comp_indices = nonzero_by_cart[cart_idx]
            if comp_indices.size == 0:
                continue
            term = coeff * _primitive_cartesian_value(lx, ly, lz, alpha_f, x, y, z, exp_term)
            for comp_idx in comp_indices:
                comps[int(comp_idx)] += matrix[cart_idx, int(comp_idx)] * term
    return comps


def evaluate_shell_float32(shell: Shell, points: np.ndarray) -> list[np.ndarray]:
    points32 = points.astype(np.float32, copy=False)
    center = shell.center.astype(np.float32, copy=False)
    rel = points32 - center
    x = rel[:, 0]
    y = rel[:, 1]
    z = rel[:, 2]
    r2 = x * x + y * y + z * z
    st = shell.shell_type

    if st == -1:
        comps = [np.zeros(points32.shape[0], dtype=np.float32) for _ in range(4)]
        powers = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        exponents = shell.exponents.astype(np.float32, copy=False)
        coefficients = shell.coefficients.astype(np.float32, copy=False)
        sp_coefficients = (
            shell.sp_coefficients.astype(np.float32, copy=False)
            if shell.sp_coefficients is not None
            else coefficients
        )
        for pidx, alpha in enumerate(exponents):
            exp_term = np.exp(-alpha * r2)
            coeffs = [coefficients[pidx], sp_coefficients[pidx], sp_coefficients[pidx], sp_coefficients[pidx]]
            for comp_idx, (lx, ly, lz) in enumerate(powers):
                comps[comp_idx] += coeffs[comp_idx] * _primitive_cartesian_value32(
                    lx, ly, lz, alpha, x, y, z, exp_term
                )
        return comps

    angular_momentum = abs(st)
    if angular_momentum not in CARTESIAN_POWERS:
        raise NotImplementedError(f"Unsupported shell type {st}; only S through H shells are supported")
    powers = CARTESIAN_POWERS[angular_momentum]
    if st >= 0:
        matrix = np.eye(len(powers), dtype=np.float32)
    else:
        matrix = SPHERICAL_TO_CARTESIAN[angular_momentum].astype(np.float32, copy=False)

    comps = [np.zeros(points32.shape[0], dtype=np.float32) for _ in range(matrix.shape[1])]
    nonzero_by_cart = [np.flatnonzero(np.abs(matrix[row]) > 1.0e-14) for row in range(matrix.shape[0])]
    exponents = shell.exponents.astype(np.float32, copy=False)
    coefficients = shell.coefficients.astype(np.float32, copy=False)
    for pidx, alpha in enumerate(exponents):
        coeff = coefficients[pidx]
        exp_term = np.exp(-alpha * r2)
        for cart_idx, (lx, ly, lz) in enumerate(powers):
            comp_indices = nonzero_by_cart[cart_idx]
            if comp_indices.size == 0:
                continue
            term = coeff * _primitive_cartesian_value32(lx, ly, lz, alpha, x, y, z, exp_term)
            for comp_idx in comp_indices:
                comps[int(comp_idx)] += matrix[cart_idx, int(comp_idx)] * term
    return comps


def evaluate_mos(
    wavefunction: FCHKWavefunction,
    spin: str,
    orbital_indices0: Iterable[int],
    points: np.ndarray,
    coeff_cutoff: float = 1.0e-10,
) -> np.ndarray:
    indices = np.array(list(orbital_indices0), dtype=np.int64)
    if indices.size == 0:
        return np.empty((0, points.shape[0]), dtype=np.float64)

    coeff_matrix = wavefunction.coefficients(spin)
    if int(indices.min()) < 0 or int(indices.max()) >= coeff_matrix.shape[0]:
        raise IndexError(f"Orbital index is outside 1..{coeff_matrix.shape[0]}")

    selected_coeffs = coeff_matrix[indices]
    values = np.zeros((indices.size, points.shape[0]), dtype=np.float64)
    basis_index = 0
    for shell in wavefunction.shells:
        nfunc = n_shell_functions(shell.shell_type)
        shell_coeffs = selected_coeffs[:, basis_index : basis_index + nfunc]
        basis_index += nfunc
        active_components = np.max(np.abs(shell_coeffs), axis=0) >= coeff_cutoff
        if not np.any(active_components):
            continue
        components = evaluate_shell(shell, points)
        component_matrix = np.vstack(
            [components[int(comp_idx)] for comp_idx in np.flatnonzero(active_components)]
        )
        values += shell_coeffs[:, active_components] @ component_matrix
    return values


def make_grid(
    wavefunction: FCHKWavefunction,
    grid_size: int,
    margin_bohr: float,
    max_points: int = 900_000,
) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray, np.ndarray]:
    coords = wavefunction.coordinates_bohr
    low = coords.min(axis=0) - margin_bohr
    high = coords.max(axis=0) + margin_bohr
    lengths = np.maximum(high - low, 1.0)
    longest = float(lengths.max())
    spacing_value = longest / max(grid_size - 1, 1)
    counts = np.maximum(8, np.ceil(lengths / spacing_value).astype(int) + 1)
    total = int(np.prod(counts))
    if total > max_points:
        scale = (max_points / total) ** (1.0 / 3.0)
        counts = np.maximum(8, np.floor(counts * scale).astype(int))
        spacing_value = longest / max(int(counts.max()) - 1, 1)
    axes = [low[i] + np.arange(counts[i]) * spacing_value for i in range(3)]
    xg, yg, zg = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    points = np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))
    return points, tuple(int(v) for v in counts), np.array([spacing_value] * 3, dtype=np.float64), low


@dataclass
class OrbitalGrid:
    spin: str
    orbital_index0: int
    grid_size: int
    margin_bohr: float
    values: np.ndarray
    shape: tuple[int, int, int]
    spacing: np.ndarray
    origin: np.ndarray
    auto_iso: float


@dataclass
class BasisGrid:
    grid_size: int
    margin_bohr: float
    shape: tuple[int, int, int]
    spacing: np.ndarray
    origin: np.ndarray
    basis_values: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.basis_values.nbytes)


@dataclass
class SurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def auto_isovalue(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 0.03
    vmax = float(finite.max())
    if vmax <= 0.0:
        return 0.03
    p99 = float(np.percentile(finite, 99.25))
    return max(1.0e-5, min(0.75 * vmax, 0.42 * p99))


def compute_basis_grid(
    wavefunction: FCHKWavefunction,
    grid_size: int,
    margin_bohr: float,
    workers: int = 1,
) -> BasisGrid:
    points, shape, spacing, origin = make_grid(wavefunction, grid_size, margin_bohr)
    points32 = points.astype(np.float32, copy=False)
    basis_values = np.empty((wavefunction.n_basis, points32.shape[0]), dtype=np.float32)

    shell_jobs: list[tuple[int, Shell]] = []
    basis_index = 0
    for shell in wavefunction.shells:
        shell_jobs.append((basis_index, shell))
        basis_index += n_shell_functions(shell.shell_type)

    def fill_shell(job: tuple[int, Shell]) -> None:
        start, shell = job
        components = evaluate_shell_float32(shell, points32)
        for offset, component in enumerate(components):
            basis_values[start + offset] = component

    worker_count = max(1, int(workers))
    if worker_count > 1 and len(shell_jobs) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(fill_shell, shell_jobs))
    else:
        for job in shell_jobs:
            fill_shell(job)

    return BasisGrid(
        grid_size=grid_size,
        margin_bohr=margin_bohr,
        shape=shape,
        spacing=spacing,
        origin=origin,
        basis_values=basis_values,
    )


def compute_orbital_grids(
    wavefunction: FCHKWavefunction,
    spin: str,
    orbital_indices0: Iterable[int],
    grid_size: int,
    margin_bohr: float,
    basis_grid: BasisGrid | None = None,
) -> list[OrbitalGrid]:
    indices = list(orbital_indices0)
    if not indices:
        return []

    coeff_matrix = wavefunction.coefficients(spin)
    index_array = np.asarray(indices, dtype=np.int64)
    if int(index_array.min()) < 0 or int(index_array.max()) >= coeff_matrix.shape[0]:
        raise IndexError(f"Orbital index is outside 1..{coeff_matrix.shape[0]}")

    selected_coeffs = coeff_matrix[index_array].astype(np.float32, copy=False)

    if basis_grid is not None:
        # Fast path: reuse precomputed basis function values
        value_rows = selected_coeffs @ basis_grid.basis_values
        shape = basis_grid.shape
        spacing = basis_grid.spacing
        origin = basis_grid.origin
        grid_size_out = basis_grid.grid_size
        margin_out = basis_grid.margin_bohr
    else:
        # Normal path: compute grid and evaluate shells
        points, shape, spacing, origin = make_grid(wavefunction, grid_size, margin_bohr)
        value_rows = evaluate_mos(wavefunction, spin, indices, points)
        grid_size_out = grid_size
        margin_out = margin_bohr

    grids: list[OrbitalGrid] = []
    for row_idx, orbital_index0 in enumerate(indices):
        values = value_rows[row_idx].reshape(shape).copy()
        grids.append(
            OrbitalGrid(
                spin=spin,
                orbital_index0=orbital_index0,
                grid_size=grid_size_out,
                margin_bohr=margin_out,
                values=values,
                shape=shape,
                spacing=spacing,
                origin=origin,
                auto_iso=auto_isovalue(values),
            )
        )
    return grids


def _empty_mesh() -> SurfaceMesh:
    return SurfaceMesh(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int64))


def smooth_surface_mesh(mesh: SurfaceMesh, iterations: int = 0, relaxation: float = 0.18) -> SurfaceMesh:
    if mesh.vertices.shape[0] < 4 or mesh.faces.shape[0] < 4 or iterations <= 0:
        return mesh
    vertices = mesh.vertices.astype(np.float32, copy=True)
    faces = mesh.faces.astype(np.int64, copy=False)
    edges = np.vstack(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        )
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    src = np.concatenate((edges[:, 0], edges[:, 1]))
    dst = np.concatenate((edges[:, 1], edges[:, 0]))
    degree = np.bincount(src, minlength=vertices.shape[0]).astype(np.float32)
    movable = degree > 0
    for _ in range(iterations):
        neighbor_sum = np.zeros_like(vertices)
        np.add.at(neighbor_sum, src, vertices[dst])
        averaged = vertices.copy()
        averaged[movable] = neighbor_sum[movable] / degree[movable, None]
        vertices[movable] = (1.0 - relaxation) * vertices[movable] + relaxation * averaged[movable]
    return SurfaceMesh(vertices, mesh.faces)


def _marching_mesh(field: np.ndarray, level: float, origin: np.ndarray, spacing: np.ndarray) -> SurfaceMesh:
    if not np.isfinite(level):
        return _empty_mesh()
    vmin = float(np.nanmin(field))
    vmax = float(np.nanmax(field))
    if not (vmin < level < vmax):
        return _empty_mesh()
    try:
        from skimage import measure
    except ModuleNotFoundError as exc:  # pragma: no cover - user-facing dependency guard
        print(
            "Missing dependency: scikit-image\n"
            "Install with:\n"
            "pip install numpy scikit-image pyqtgraph PyQt6 PyOpenGL",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    vertices, faces, _normals, _values = measure.marching_cubes(
        field,
        level=level,
        spacing=tuple(float(v) for v in spacing),
        allow_degenerate=False,
    )
    vertices = (vertices + origin) * BOHR_TO_ANG
    return smooth_surface_mesh(
        SurfaceMesh(vertices.astype(np.float32, copy=False), faces.astype(np.int64, copy=False))
    )


def extract_isosurfaces(grid: OrbitalGrid, iso: float) -> tuple[SurfaceMesh, SurfaceMesh, float]:
    level = grid.auto_iso if iso <= 0 else float(abs(iso))
    return (
        _marching_mesh(grid.values, level, grid.origin, grid.spacing),
        _marching_mesh(grid.values, -level, grid.origin, grid.spacing),
        level,
    )


def surface_for_orbital(
    wavefunction: FCHKWavefunction,
    spin: str,
    orbital_index0: int,
    grid_size: int,
    iso: float,
    margin_bohr: float,
) -> tuple[SurfaceMesh, SurfaceMesh, float, tuple[int, int, int]]:
    grid = compute_orbital_grids(wavefunction, spin, [orbital_index0], grid_size, margin_bohr)[0]
    pos, neg, level = extract_isosurfaces(grid, iso)
    return pos, neg, level, grid.shape


def detect_wavefunction_format(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(262_144)
    lower = sample.lower()
    if "[molden format]" in lower or ("[atoms]" in lower and "[gto]" in lower):
        return "molden"
    if "number of basis functions" in lower or "alpha mo coefficients" in lower:
        return "fchk"
    name = path.name.lower()
    if name.endswith((".fchk", ".fch")):
        return "fchk"
    if "molden" in name or name.endswith((".molden", ".mol")):
        return "molden"
    raise ValueError(f"Could not determine wavefunction file type for {path}")


def parse_wavefunction(
    path: Path, 
    file_format: str | None = None
) -> FCHKWavefunction:
    resolved_format = (file_format or detect_wavefunction_format(path)).lower()
    if resolved_format == "fchk":
        return FCHKParser(path).parse()
    if resolved_format == "molden":
        return MoldenParser(path).parse()
    raise ValueError(f"Unsupported wavefunction file type: {resolved_format}")


def compute_bonds(atomic_numbers: np.ndarray, coords_ang: np.ndarray) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for i in range(len(atomic_numbers)):
        zi = int(atomic_numbers[i])
        if zi == 0:
            continue
        ri = COVALENT_RADII.get(zi, 0.75)
        for j in range(i + 1, len(atomic_numbers)):
            zj = int(atomic_numbers[j])
            if zj == 0:
                continue
            rj = COVALENT_RADII.get(zj, 0.75)
            cutoff = 1.22 * (ri + rj) + 0.16
            dist = float(np.linalg.norm(coords_ang[i] - coords_ang[j]))
            if 0.25 < dist < cutoff:
                bonds.append((i, j))
    return bonds


def run_batch(args: argparse.Namespace) -> int:
    input_path = getattr(args, "input", None) or getattr(args, "fchk", None)
    wf = parse_wavefunction(Path(input_path), getattr(args, "file_format", None))
    orbital_index0 = args.orbital - 1
    pos, neg, level, shape = surface_for_orbital(wf, args.spin, orbital_index0, args.grid, args.iso, args.margin)
    energy = wf.energies(args.spin)[orbital_index0]
    occ = wf.occupation(args.spin, orbital_index0)
    print(f"file: {wf.path}")
    print(f"format: {wf.source_format}")
    print(f"atoms: {len(wf.atomic_numbers)}  basis: {wf.n_basis}")
    print(f"spin: {args.spin}  orbital: {args.orbital}  occupation: {occ:g}")
    print(f"energy: {energy:.8f} Eh  {energy * HARTREE_TO_EV:.4f} eV")
    print(f"grid: {shape}  isovalue: {level:.6g}")
    print(f"positive triangles: {pos.n_faces}")
    print(f"negative triangles: {neg.n_faces}")
    return 0
