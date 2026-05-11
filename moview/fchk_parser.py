
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from moview.utils import BOHR_TO_ANG
from moview.utils import _as_float, n_shell_functions
from moview.utils import Shell


def _section_header(line: str) -> tuple[str, str] | None:
    match = re.match(r"^\s*\[([^\]]+)\]\s*(.*)$", line)
    if match is None:
        return None
    return match.group(1).strip().lower(), match.group(2).strip()

@dataclass
class FCHKWavefunction:
    path: Path
    title: str
    method_line: str
    atomic_numbers: np.ndarray
    coordinates_bohr: np.ndarray
    n_alpha: int
    n_beta: int
    n_basis: int
    shell_types: np.ndarray
    shell_to_atom: np.ndarray
    shells: list[Shell]
    alpha_energies: np.ndarray
    beta_energies: np.ndarray | None
    alpha_coefficients: np.ndarray
    beta_coefficients: np.ndarray | None
    alpha_occupations: np.ndarray | None = None
    beta_occupations: np.ndarray | None = None
    source_format: str = "fchk"

    @property
    def coordinates_angstrom(self) -> np.ndarray:
        return self.coordinates_bohr * BOHR_TO_ANG

    @property
    def is_unrestricted(self) -> bool:
        return self.beta_coefficients is not None

    def energies(self, spin: str) -> np.ndarray:
        if spin == "beta" and self.beta_energies is not None:
            return self.beta_energies
        return self.alpha_energies

    def coefficients(self, spin: str) -> np.ndarray:
        if spin == "beta" and self.beta_coefficients is not None:
            return self.beta_coefficients
        return self.alpha_coefficients

    def occupation(self, spin: str, orbital_index0: int) -> float:
        if spin == "beta" and self.beta_occupations is not None:
            return float(self.beta_occupations[orbital_index0])
        if spin != "beta" and self.alpha_occupations is not None:
            return float(self.alpha_occupations[orbital_index0])
        if self.beta_coefficients is None:
            if orbital_index0 < self.n_beta:
                return 2.0
            if orbital_index0 < self.n_alpha:
                return 1.0
            return 0.0
        if spin == "alpha":
            return 1.0 if orbital_index0 < self.n_alpha else 0.0
        return 1.0 if orbital_index0 < self.n_beta else 0.0

    def default_orbital(self, spin: str) -> int:
        count = self.n_beta if spin == "beta" else self.n_alpha
        return min(max(0, count - 1), len(self.energies(spin)) - 1)

    def lumo_orbital(self, spin: str) -> int:
        energies = self.energies(spin)
        for idx in range(len(energies)):
            if self.occupation(spin, idx) <= 0:
                return idx
        return len(energies) - 1



class FCHKParser:
    HEADER_RE = re.compile(r"^(?P<label>.*?)\s+(?P<kind>[IRC])\s+(?:N=\s*(?P<n>\d+)|(?P<value>.*?))\s*$")

    def __init__(self, path: Path):
        self.path = path
        self.title = path.name
        self.method_line = ""
        self.fields: dict[str, object] = {}

    def parse(self) -> FCHKWavefunction:
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()

        if lines:
            self.title = lines[0].strip()
        if len(lines) > 1:
            self.method_line = lines[1].strip()

        i = 2 if len(lines) > 1 else 0
        while i < len(lines):
            line = lines[i].rstrip("\n")
            match = self.HEADER_RE.match(line)
            if not match:
                i += 1
                continue

            label = match.group("label").strip()
            kind = match.group("kind")
            n_text = match.group("n")
            value = match.group("value") or ""
            i += 1

            if n_text is None:
                parts = value.split()
                if kind == "I":
                    self.fields[label] = int(parts[-1])
                elif kind == "R":
                    self.fields[label] = _as_float(parts[-1])
                else:
                    self.fields[label] = value.strip()
                continue

            n_items = int(n_text)
            if kind == "C":
                self.fields[label] = []
                continue

            block_lines: list[str] = []
            token_count = 0
            while i < len(lines) and token_count < n_items:
                block_lines.append(lines[i])
                token_count += len(lines[i].split())
                i += 1
            if token_count < n_items:
                raise ValueError(f"Field {label!r} ended early: expected {n_items}, got {token_count}")
            block_text = "".join(block_lines)
            if kind == "I":
                array = np.fromstring(block_text, dtype=np.int32, sep=" ", count=n_items)
            else:
                array = np.fromstring(
                    block_text.replace("D", "E").replace("d", "E"),
                    dtype=np.float64,
                    sep=" ",
                    count=n_items,
                )
            if array.size != n_items:
                raise ValueError(f"Field {label!r} ended early: expected {n_items}, got {array.size}")
            self.fields[label] = array

        return self._build_wavefunction()

    def _field(self, *names: str):
        for name in names:
            if name in self.fields:
                return self.fields[name]
        joined = ", ".join(names)
        raise KeyError(f"Missing required fchk field: {joined}")

    def _optional_array(self, *names: str) -> np.ndarray | None:
        for name in names:
            value = self.fields.get(name)
            if isinstance(value, np.ndarray):
                return value
        return None

    @staticmethod
    def _reshape_coefficients(raw: np.ndarray, n_basis: int, label: str) -> np.ndarray:
        if raw.size % n_basis != 0:
            raise ValueError(f"{label} length {raw.size} is not divisible by basis count {n_basis}")
        return raw.reshape((raw.size // n_basis, n_basis))

    @staticmethod
    def _fit_energies(energies: np.ndarray, count: int) -> np.ndarray:
        if energies.size == count:
            return energies
        if energies.size > count:
            return energies[:count].copy()
        out = np.full(count, np.nan, dtype=np.float64)
        out[: energies.size] = energies
        return out

    def _build_wavefunction(self) -> FCHKWavefunction:
        atomic_numbers = np.asarray(self._field("Atomic numbers"), dtype=np.int32)
        coords = np.asarray(self._field("Current cartesian coordinates"), dtype=np.float64).reshape(-1, 3)
        n_alpha = int(self._field("Number of alpha electrons"))
        n_beta = int(self._field("Number of beta electrons"))
        n_basis = int(self._field("Number of basis functions"))
        shell_types = np.asarray(self._field("Shell types"), dtype=np.int32)
        nprim = np.asarray(self._field("Number of primitives per shell"), dtype=np.int32)
        shell_to_atom = np.asarray(self._field("Shell to atom map"), dtype=np.int32) - 1
        exponents = np.asarray(self._field("Primitive exponents"), dtype=np.float64)
        coefficients = np.asarray(self._field("Contraction coefficients"), dtype=np.float64)
        sp_coefficients = self._optional_array("P(S=P) Contraction coefficients")
        shell_coords_arr = self._optional_array("Coordinates of each shell")
        if shell_coords_arr is None:
            shell_coords = coords[shell_to_atom]
        else:
            shell_coords = np.asarray(shell_coords_arr, dtype=np.float64).reshape(-1, 3)

        shells: list[Shell] = []
        prim_start = 0
        for idx, shell_type in enumerate(shell_types):
            count = int(nprim[idx])
            prim_slice = slice(prim_start, prim_start + count)
            p_coeffs = None
            if shell_type == -1 and sp_coefficients is not None:
                p_coeffs = sp_coefficients[prim_slice].copy()
            shells.append(
                Shell(
                    shell_type=int(shell_type),
                    center=shell_coords[idx].copy(),
                    exponents=exponents[prim_slice].copy(),
                    coefficients=coefficients[prim_slice].copy(),
                    sp_coefficients=p_coeffs,
                )
            )
            prim_start += count

        expected_basis = sum(n_shell_functions(int(st)) for st in shell_types)
        if expected_basis != n_basis:
            raise ValueError(f"Basis count mismatch: fchk says {n_basis}, shell layout gives {expected_basis}")

        alpha_energies = np.asarray(
            self._field("Alpha Orbital Energies", "alpha orbital energies", "orbital energies"),
            dtype=np.float64,
        )
        beta_energies = self._optional_array("Beta Orbital Energies", "beta orbital energies")
        alpha_raw = np.asarray(
            self._field("Alpha MO coefficients", "alpha MO coefficients", "MO coefficients"),
            dtype=np.float64,
        )
        beta_raw = self._optional_array("Beta MO coefficients", "beta MO coefficients")

        alpha_coefficients = self._reshape_coefficients(alpha_raw, n_basis, "Alpha MO coefficients")
        beta_coefficients = (
            self._reshape_coefficients(np.asarray(beta_raw, dtype=np.float64), n_basis, "Beta MO coefficients")
            if beta_raw is not None
            else None
        )
        alpha_energies = self._fit_energies(alpha_energies, alpha_coefficients.shape[0])
        if beta_coefficients is not None and beta_energies is not None:
            beta_energies = self._fit_energies(np.asarray(beta_energies, dtype=np.float64), beta_coefficients.shape[0])

        return FCHKWavefunction(
            path=self.path,
            title=self.title,
            method_line=self.method_line,
            atomic_numbers=atomic_numbers,
            coordinates_bohr=coords,
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_basis=n_basis,
            shell_types=shell_types,
            shell_to_atom=shell_to_atom,
            shells=shells,
            alpha_energies=alpha_energies,
            beta_energies=beta_energies,
            alpha_coefficients=alpha_coefficients,
            beta_coefficients=beta_coefficients,
        )