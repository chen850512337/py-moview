import argparse

from moview.moview import run_batch
from moview.gui import run_gui

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read fchk/Molden files and view molecular orbital "
                    "isosurfaces with OpenGL."
    )
    parser.add_argument(
        "input", nargs="?", help="Path to .fchk/.fch/Molden file"
    )
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--fchk", dest="file_format", action="store_const", const="fchk", 
        help="Treat input as Gaussian fchk"
    )
    format_group.add_argument(
        "--molden", dest="file_format", action="store_const", const="molden", 
        help="Treat input as Molden"
    )
    parser.add_argument(
        "--batch", action="store_true", 
        help="Run a non-GUI parse/evaluate/surface test"
    )
    parser.add_argument(
        "--spin", choices=["alpha", "beta"], default="alpha"
    )
    parser.add_argument(
        "--orbital", type=int, default=1, 
        help="1-based orbital index for --batch"
    )
    parser.add_argument(
        "--grid", type=int, default=64, 
        help="Approximate grid points along longest axis"
    )
    parser.add_argument(
        "--iso", type=float, default=0.05, 
        help="Isovalue, default: 0.05."
    )
    parser.add_argument(
        "--margin", type=float, default=4.0, 
        help="Box margin in bohr"
    )
    parser.add_argument(
        "--prefetch-workers", type=int, default=4, 
        help="Basis-grid/pre-render worker threads"
    )
    parser.add_argument(
        "--no-auto-render", action="store_true", 
        help="Do not automatically render HOMO after opening the GUI"
    )
    args = parser.parse_args()

    if args.batch:
        if not args.input:
            parser.error("--batch requires an input path")
        return run_batch(args)
    
    return run_gui(
        args.input,
        args.grid,
        args.iso,
        args.margin,
        args.prefetch_workers,
        auto_render=not args.no_auto_render,
        file_format=args.file_format,
    )


if __name__ == "__main__":
    raise SystemExit(main())