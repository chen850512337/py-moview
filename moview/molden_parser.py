
from pathlib import Path

import numpy as np


from moview.fchk_parser import FCHKWavefunction
from moview.fchk_parser import _section_header

from moview.utils import SYMBOL_TO_ATOMIC_NUMBER, BOHR_TO_ANG
from moview.utils import n_shell_functions, _as_float
from moview.utils import Shell



class MoldenParser:
    SHELL_LABELS = {
        "s": (0, 1),
        "p": (1, 3),
        "d": (2, 6),
        "f": (3, 10),
        "g": (4, 15),
        "h": (5, 21),
        "sp": (-1, 4),
    }

    def __init__(self, path: Path):
        self.path = path
        self.title = path.name
        self.lines: list[str] = []
        self.sections: dict[str, tuple[int, int, str]] = {}
        self.spherical_d = False
        self.spherical_f = False
        self.spherical_g = False
        self.spherical_h = False

    def parse(self) -> FCHKWavefunction:
        self.lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        self.sections = self._build_sections()
        self.title = self._parse_title()
        atomic_numbers, coords_bohr = self._parse_atoms()
        shells = self._parse_gto(coords_bohr)
        n_basis = sum(n_shell_functions(shell.shell_type) for shell in shells)
        alpha_energies, alpha_occ, alpha_coeffs, beta_energies, beta_occ, beta_coeffs = self._parse_mo(n_basis)
        n_alpha = int(np.count_nonzero(alpha_occ > 1.0e-8))
        n_beta = int(np.count_nonzero(beta_occ > 1.0e-8)) if beta_occ is not None else n_alpha
        return FCHKWavefunction(
            path=self.path,
            title=self.title,
            method_line="Molden",
            atomic_numbers=atomic_numbers,
            coordinates_bohr=coords_bohr,
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_basis=n_basis,
            shell_types=np.asarray([shell.shell_type for shell in shells], dtype=np.int32),
            shell_to_atom=np.asarray([int(np.argmin(np.linalg.norm(coords_bohr - shell.center, axis=1))) for shell in shells], dtype=np.int32),
            shells=shells,
            alpha_energies=alpha_energies,
            beta_energies=beta_energies,
            alpha_coefficients=alpha_coeffs,
            beta_coefficients=beta_coeffs,
            alpha_occupations=alpha_occ,
            beta_occupations=beta_occ,
            source_format="molden",
        )

    def _build_sections(self) -> dict[str, tuple[int, int, str]]:
        starts: list[tuple[str, int, str]] = []
        for idx, line in enumerate(self.lines):
            header = _section_header(line)
            if header is None:
                continue
            name, suffix = header
            starts.append((name, idx, suffix))
            if name == "5d":
                self.spherical_d = True
            elif name == "6d":
                self.spherical_d = False
            elif name == "7f":
                self.spherical_f = True
            elif name == "10f":
                self.spherical_f = False
            elif name == "9g":
                self.spherical_g = True
            elif name == "15g":
                self.spherical_g = False
            elif name == "11h":
                self.spherical_h = True
            elif name == "21h":
                self.spherical_h = False
        sections: dict[str, tuple[int, int, str]] = {}
        for pos, (name, start, suffix) in enumerate(starts):
            end = starts[pos + 1][1] if pos + 1 < len(starts) else len(self.lines)
            sections.setdefault(name, (start + 1, end, suffix))
        return sections

    def _parse_title(self) -> str:
        title_section = self.sections.get("title")
        if title_section is None:
            return self.path.name
        start, end, _suffix = title_section
        for line in self.lines[start:end]:
            text = line.strip()
            if text:
                return text
        return self.path.name

    def _parse_atoms(self) -> tuple[np.ndarray, np.ndarray]:
        section = self.sections.get("atoms")
        if section is None:
            raise ValueError("Molden file is missing [Atoms]")
        start, end, suffix = section
        unit = suffix.upper()
        atomic_numbers: list[int] = []
        coords: list[tuple[float, float, float]] = []
        for line in self.lines[start:end]:
            parts = line.split()
            if len(parts) < 6:
                continue
            symbol = parts[0].upper()
            try:
                atomic_number = int(parts[2])
            except ValueError:
                atomic_number = SYMBOL_TO_ATOMIC_NUMBER.get(symbol)
                if atomic_number is None:
                    raise ValueError(f"Unknown Molden atom symbol: {parts[0]!r}") from None
            atomic_numbers.append(atomic_number)
            coords.append((_as_float(parts[3]), _as_float(parts[4]), _as_float(parts[5])))
        if not atomic_numbers:
            raise ValueError("Molden [Atoms] section contains no atoms")
        coords_arr = np.asarray(coords, dtype=np.float64)
        if "AU" not in unit:
            coords_arr = coords_arr / BOHR_TO_ANG
        return np.asarray(atomic_numbers, dtype=np.int32), coords_arr

    def _molden_shell_type(self, label: str) -> int:
        base_type, _count = self.SHELL_LABELS[label]
        if label == "d" and self.spherical_d:
            return -2
        if label == "f" and self.spherical_f:
            return -3
        if label == "g" and self.spherical_g:
            return -4
        if label == "h" and self.spherical_h:
            return -5
        return base_type

    def _parse_gto(self, coords_bohr: np.ndarray) -> list[Shell]:
        section = self.sections.get("gto")
        if section is None:
            raise ValueError("Molden file is missing [GTO]")
        start, end, _suffix = section
        shells: list[Shell] = []
        atom_index0: int | None = None
        i = start
        while i < end:
            line = self.lines[i].strip()
            i += 1
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].lstrip("+-").isdigit():
                atom_index0 = int(parts[0]) - 1
                if atom_index0 < 0 or atom_index0 >= coords_bohr.shape[0]:
                    raise ValueError(f"Molden [GTO] atom index out of range: {parts[0]}")
                continue
            label = parts[0].lower()
            if atom_index0 is None or label not in self.SHELL_LABELS or len(parts) < 2:
                continue
            n_prim = int(parts[1])
            scale = _as_float(parts[2]) if len(parts) >= 3 else 1.0
            exponents: list[float] = []
            coefficients: list[float] = []
            sp_coefficients: list[float] | None = [] if label == "sp" else None
            for _ in range(n_prim):
                if i >= end:
                    raise ValueError("Molden [GTO] ended inside a shell")
                prim_parts = self.lines[i].split()
                i += 1
                if len(prim_parts) < 2:
                    raise ValueError("Invalid Molden primitive line")
                exponents.append(_as_float(prim_parts[0]))
                coefficients.append(scale * _as_float(prim_parts[1]))
                if sp_coefficients is not None:
                    if len(prim_parts) < 3:
                        raise ValueError("Molden sp shell primitive is missing p coefficient")
                    sp_coefficients.append(scale * _as_float(prim_parts[2]))
            shells.append(
                Shell(
                    shell_type=self._molden_shell_type(label),
                    center=coords_bohr[atom_index0].copy(),
                    exponents=np.asarray(exponents, dtype=np.float64),
                    coefficients=np.asarray(coefficients, dtype=np.float64),
                    sp_coefficients=(
                        np.asarray(sp_coefficients, dtype=np.float64)
                        if sp_coefficients is not None
                        else None
                    ),
                )
            )
        if not shells:
            raise ValueError("Molden [GTO] section contains no basis shells")
        return shells

    def _finish_mo_block(
        self,
        block: dict[str, object],
        n_basis: int,
        alpha_rows: list[np.ndarray],
        alpha_energies: list[float],
        alpha_occ: list[float],
        beta_rows: list[np.ndarray],
        beta_energies: list[float],
        beta_occ: list[float],
    ) -> None:
        coeff_pairs = block.get("coefficients")
        if not coeff_pairs:
            return
        row = np.zeros(n_basis, dtype=np.float64)
        for basis_idx, coeff in coeff_pairs:  # type: ignore[union-attr]
            if 1 <= basis_idx <= n_basis:
                row[basis_idx - 1] = coeff
        spin = str(block.get("spin", "alpha")).lower()
        energy = float(block.get("energy", np.nan))
        occupation = float(block.get("occupation", 0.0))
        if spin.startswith("beta"):
            beta_rows.append(row)
            beta_energies.append(energy)
            beta_occ.append(occupation)
        else:
            alpha_rows.append(row)
            alpha_energies.append(energy)
            alpha_occ.append(occupation)

    def _parse_mo(
        self,
        n_basis: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        section = self.sections.get("mo")
        if section is None:
            raise ValueError("Molden file is missing [MO]")
        start, end, _suffix = section
        alpha_rows: list[np.ndarray] = []
        alpha_energies: list[float] = []
        alpha_occ: list[float] = []
        beta_rows: list[np.ndarray] = []
        beta_energies: list[float] = []
        beta_occ: list[float] = []
        block: dict[str, object] = {}
        for line in self.lines[start:end]:
            text = line.strip()
            if not text:
                continue
            lower = text.lower()
            if lower.startswith("sym="):
                self._finish_mo_block(
                    block, n_basis, alpha_rows, alpha_energies, alpha_occ, beta_rows, beta_energies, beta_occ
                )
                block = {"coefficients": []}
            elif lower.startswith("ene="):
                block["energy"] = _as_float(text.split("=", 1)[1])
            elif lower.startswith("spin="):
                block["spin"] = text.split("=", 1)[1].strip()
            elif lower.startswith("occup="):
                block["occupation"] = _as_float(text.split("=", 1)[1])
            else:
                parts = text.split()
                if len(parts) >= 2 and parts[0].lstrip("+-").isdigit():
                    block.setdefault("coefficients", []).append((int(parts[0]), _as_float(parts[1])))  # type: ignore[union-attr]
        self._finish_mo_block(
            block, n_basis, alpha_rows, alpha_energies, alpha_occ, beta_rows, beta_energies, beta_occ
        )
        if not alpha_rows:
            raise ValueError("Molden [MO] section contains no alpha orbitals")
        alpha_coeffs = np.vstack(alpha_rows)
        beta_coeffs = np.vstack(beta_rows) if beta_rows else None
        return (
            np.asarray(alpha_energies, dtype=np.float64),
            np.asarray(alpha_occ, dtype=np.float64),
            alpha_coeffs,
            np.asarray(beta_energies, dtype=np.float64) if beta_rows else None,
            np.asarray(beta_occ, dtype=np.float64) if beta_rows else None,
            beta_coeffs,
        )