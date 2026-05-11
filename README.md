# Py-MOview
This repo fork from Py-MOview.
Py-MOview is a lightweight Python/OpenGL viewer for molecular orbital isosurfaces. It is designed as a cross-platform alternative for quickly previewing molecular orbitals directly from wavefunction files, especially on macOS where some GUI functions of Multiwfn are not available.

The program currently supports Gaussian formatted checkpoint files (`.fchk/.fch`) and Molden-format wavefunction files. It provides an interactive GUI for viewing orbital isosurfaces, orbital energies, occupations, and alpha/beta spin orbitals.

## Features

- Read Gaussian `.fchk/.fch` files
- Read Molden-format wavefunction files
- Visualize molecular orbital isosurfaces with OpenGL
- Display orbital energies and occupations
- Support alpha/beta spin orbitals
- HOMO/LUMO quick selection
- Adjustable isovalue, grid size, and box margin
- Multi-orbital comparison in the same window
- Basic keyboard shortcuts for orbital switching and view rotation
- Pre-rendering/cache mechanism for faster orbital switching

## Installation


```bash
conda create -n moview python=3.12
conda activate moview
pip install -e ./
```

## Usage

Run directly with Python:

```bash
moviewer molecule.fchk
```

## Common options

```bash
--grid <int>
```

Set the rendering grid size. The default value is 56. Larger values give smoother and more detailed isosurfaces but require more computation. A value around 81 is usually sufficient for high-quality visualization.

```bash
--iso <float>
```

Set the isovalue. The default value is 0, which enables automatic isovalue selection. For typical molecular orbitals, an isovalue around 0.05 may be reasonable, depending on the system.

```bash
--margin <float>
```

Set the box margin in bohr. The default value is 4.0. Increasing this value may help prevent distant isosurfaces from being truncated, but it also increases computational cost.

```bash
--prefetch-workers <int>
```

Set the number of worker threads used for basis-grid construction and pre-rendering. The default value is 12. On most machines, using roughly 2–3 workers per CPU core is a reasonable starting point.

## GUI controls

The left panel contains file information, rendering settings, orbital selection, and comparison controls.

Main controls:

- **Open file**: load a new wavefunction file
- **Render**: render the currently selected orbital
- **Spin**: select alpha or beta orbitals when both are available
- **Grid**: set the rendering grid size
- **Margin**: set the grid margin in bohr
- **Isovalue**: adjust the orbital isosurface value
- **Compare**: render multiple orbitals side by side
- **Synchronized rotation**: synchronize or separate the views in comparison mode
- **Corner axes**: show or hide the coordinate axes
- **Orbital list**: double-click an orbital to render it

For natural orbital files, the occupation column may correspond to natural orbital occupation numbers rather than orbital energies.

## Keyboard shortcuts

| Key | Function |
| --- | --- |
| `A` | Previous orbital |
| `D` | Next orbital |
| Arrow keys | Rotate the view |
| `C` + click atom | Set rotation center |

## Notes

This program was initially developed and tested on macOS. Other operating systems may work, but they have not been systematically tested. Please report platform-specific bugs through Issues.

The default atom colors are based on GaussView-style element colors. The RGB values can be modified directly in the source code if needed.

The current implementation is intended for fast visual inspection of molecular orbitals rather than final publication-quality rendering. For high-resolution final figures, exporting cube files and rendering them in specialized visualization software such as VMD may still be preferable.

## Known limitations

- Currently tested mainly on macOS
- Input support is limited to `.fchk/.fch` and Molden-format files
- Cube-file visualization is not currently implemented
- Some uncommon basis-set or Molden conventions may require further testing

## Contributing

Contributions are welcome. Especially for new file-format support or substantial GUI changes.

When reporting a bug, please include:

- Operating system
- Python version
- Installation method
- Input file format
- Command used to launch the program
- Error message or screenshot
- A minimal reproducible example, if possible
