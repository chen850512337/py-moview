from __future__ import annotations

import re
import sys
import math
from pathlib import Path
from collections import OrderedDict
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from moview.fchk_parser import FCHKWavefunction
from moview.moview import OrbitalGrid, SurfaceMesh, BasisGrid
from moview.moview import (
    parse_wavefunction, compute_basis_grid, extract_isosurfaces,
    compute_orbital_grids, compute_bonds
)
from moview.utils import (
    ELEMENT_COLORS, PERIODIC_SYMBOLS, BOHR_TO_ANG, COVALENT_RADII,
    HARTREE_TO_EV
)
from moview.utils import atom_symbol

# Keep a copy of the embedded default element colors before the OpenGL viewer
# optionally overrides them from a colocated gview_color.tcl file.
FALLBACK_ELEMENT_COLORS = dict(ELEMENT_COLORS)


_GUI_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import shaders as pg_shaders
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing dependency guard
    _GUI_IMPORT_ERROR = exc

    class _MissingQtWidgets:
        QWidget = object
        QMainWindow = object

    class _MissingGL:
        GLViewWidget = object
        GLMeshItem = object
        MeshData = object

    QtCore = QtGui = pg = pg_shaders = None
    QtWidgets = _MissingQtWidgets()
    gl = _MissingGL()


def _require_gui_dependencies() -> None:
    if _GUI_IMPORT_ERROR is None:
        return
    missing = _GUI_IMPORT_ERROR.name or "PyQtGraph/OpenGL dependency"
    print(
        f"Missing dependency: {missing}\n"
        "Install with:\n"
        "pip install numpy scikit-image pyqtgraph PyQt6 PyOpenGL",
        file=sys.stderr,
    )
    raise SystemExit(1) from _GUI_IMPORT_ERROR

VMD_ROTATE_DEG_PER_PIXEL = 1.0 / 3.0
OPENGL_SURFACE_FACE_LIMIT = 160_000
PREFETCH_OCCUPIED_BACK = 12
PREFETCH_VIRTUAL_FORWARD = 12
PREFETCH_BATCH_SIZE = 24
CORE_PREFETCH_OCCUPIED_BACK = 30
CORE_PREFETCH_VIRTUAL_FORWARD = 15
LOW_PREFETCH_GRID = 56
MAX_COMPARE_ORBITALS = 9
MOLECULE_SPHERE_ROWS = 10
MOLECULE_SPHERE_COLS = 16
MOLECULE_BOND_COLS = 14
MOLECULE_BOND_RADIUS = 0.055
SCENE_BACKGROUND_HEX = "#f8fafc"
FOG_FLAT_SHADER = "orbitalFogFlat"
FOG_SHADED_SHADER = "orbitalFogShaded"
FOG_STRENGTH = 0.74

DISPLAY_SYMBOL_TO_ATOMIC_NUMBER = {symbol: z for z, symbol in enumerate(PERIODIC_SYMBOLS) if symbol}


def axis_rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == "x":
        return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=np.float64)
    if axis == "y":
        return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float64)
    if axis == "z":
        return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    raise ValueError(f"Unknown rotation axis: {axis}")


def default_scene_rotation() -> np.ndarray:
    return axis_rotation_matrix("x", -24.0) @ axis_rotation_matrix("y", 34.0)


def rgb_from_hex(color: str) -> np.ndarray:
    text = color.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected #rrggbb color, got {color!r}")
    return np.array([int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)


SCENE_BACKGROUND_RGB = tuple(float(value) for value in rgb_from_hex(SCENE_BACKGROUND_HEX))


def load_gview_element_colors(path: Path | None = None) -> dict[int, tuple[float, float, float]]:
    color_path = path or Path(__file__).with_name("gview_color.tcl")
    color_ids: dict[int, tuple[float, float, float]] = {}
    element_colors: dict[int, tuple[float, float, float]] = {}
    try:
        lines = color_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return dict(FALLBACK_ELEMENT_COLORS)
    for line in lines:
        parts = line.split()
        if len(parts) >= 7 and parts[:3] == ["color", "change", "rgb"]:
            try:
                rgb = (float(parts[4]), float(parts[5]), float(parts[6]))
                max_channel = max(rgb)
                if max_channel > 100.0:
                    rgb = tuple(value / 255.0 for value in rgb)
                elif max_channel > 1.0:
                    rgb = tuple(value / 100.0 for value in rgb)
                color_ids[int(parts[3])] = tuple(max(0.0, min(1.0, value)) for value in rgb)
            except ValueError:
                continue
        elif len(parts) >= 4 and parts[:2] == ["color", "Element"]:
            z = DISPLAY_SYMBOL_TO_ATOMIC_NUMBER.get(parts[2])
            if z is None:
                continue
            try:
                color_id = int(parts[3])
            except ValueError:
                continue
            rgb = color_ids.get(color_id)
            if rgb is not None:
                element_colors[z] = rgb
    if not element_colors:
        return dict(FALLBACK_ELEMENT_COLORS)
    merged = dict(FALLBACK_ELEMENT_COLORS)
    merged.update(element_colors)
    return merged


ELEMENT_COLORS = load_gview_element_colors()


def _register_fog_shaders() -> None:
    if FOG_FLAT_SHADER in pg_shaders.ShaderProgram.names:
        return
    fog_uniform = {
        "u_fog": [
            SCENE_BACKGROUND_RGB[0],
            SCENE_BACKGROUND_RGB[1],
            SCENE_BACKGROUND_RGB[2],
            0.0,
            1.0,
            FOG_STRENGTH,
        ]
    }
    pg_shaders.ShaderProgram(
        FOG_FLAT_SHADER,
        [
            pg_shaders.VertexShader(
                """
                uniform mat4 u_mvp;
                attribute vec4 a_position;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying float v_eye_depth;
                void main() {
                    gl_Position = u_mvp * a_position;
                    v_color = a_color;
                    v_eye_depth = max(gl_Position.w, 0.0);
                }
                """
            ),
            pg_shaders.FragmentShader(
                """
                #ifdef GL_ES
                precision mediump float;
                #endif
                uniform float u_fog[6];
                varying vec4 v_color;
                varying float v_eye_depth;
                void main() {
                    float fog = smoothstep(u_fog[3], u_fog[4], v_eye_depth) * u_fog[5];
                    vec3 bg = vec3(u_fog[0], u_fog[1], u_fog[2]);
                    gl_FragColor = vec4(mix(v_color.rgb, bg, fog), v_color.a);
                }
                """
            ),
        ],
        uniforms=fog_uniform,
    )
    pg_shaders.ShaderProgram(
        FOG_SHADED_SHADER,
        [
            pg_shaders.VertexShader(
                """
                uniform mat4 u_mvp;
                uniform mat3 u_normal;
                attribute vec4 a_position;
                attribute vec3 a_normal;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying vec3 v_normal;
                varying float v_eye_depth;
                void main() {
                    gl_Position = u_mvp * a_position;
                    v_normal = normalize(u_normal * a_normal);
                    v_color = a_color;
                    v_eye_depth = max(gl_Position.w, 0.0);
                }
                """
            ),
            pg_shaders.FragmentShader(
                """
                #ifdef GL_ES
                precision mediump float;
                #endif
                uniform float u_fog[6];
                varying vec4 v_color;
                varying vec3 v_normal;
                varying float v_eye_depth;
                void main() {
                    vec3 light_dir = normalize(vec3(0.65, -0.85, -1.0));
                    float diffuse = max(dot(v_normal, light_dir), 0.0);
                    float highlight = pow(diffuse, 12.0);
                    vec3 lit = v_color.rgb * (0.46 + 0.54 * diffuse) + vec3(1.0) * (0.10 * highlight);
                    float fog = smoothstep(u_fog[3], u_fog[4], v_eye_depth) * u_fog[5];
                    vec3 bg = vec3(u_fog[0], u_fog[1], u_fog[2]);
                    gl_FragColor = vec4(mix(clamp(lit, 0.0, 1.0), bg, fog), v_color.a);
                }
                """
            ),
        ],
        uniforms=fog_uniform,
    )


def update_fog_shader_params(distance: float, radius: float) -> None:
    _register_fog_shaders()
    radius = max(float(radius), 0.1)
    start = max(0.01, float(distance) - 0.20 * radius)
    end = max(start + 0.10 * radius, float(distance) + 1.05 * radius)
    fog = [
        SCENE_BACKGROUND_RGB[0],
        SCENE_BACKGROUND_RGB[1],
        SCENE_BACKGROUND_RGB[2],
        start,
        end,
        FOG_STRENGTH,
    ]
    pg_shaders.ShaderProgram.names[FOG_FLAT_SHADER]["u_fog"] = fog
    pg_shaders.ShaderProgram.names[FOG_SHADED_SHADER]["u_fog"] = fog


def glass_facecolors(triangles: np.ndarray, color: str) -> np.ndarray:
    if triangles.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    base = rgb_from_hex(color)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths > 0] /= lengths[lengths > 0, None]
    light_dir = np.array((-0.34, -0.46, 0.82), dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    view_dir = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    lambert = np.abs(normals @ light_dir)
    rim = np.power(1.0 - np.clip(np.abs(normals @ view_dir), 0.0, 1.0), 0.55)
    highlight = np.power(np.clip(normals @ light_dir, 0.0, 1.0), 10.0)
    shade = 0.50 + 0.36 * lambert
    rgb = base[None, :] * shade[:, None]
    rgb += (1.0 - base[None, :]) * (0.22 * rim[:, None] + 0.34 * highlight[:, None])
    return np.column_stack((np.clip(rgb, 0.0, 1.0), np.full(triangles.shape[0], 0.62))).astype(np.float32)


def glass_edgecolor(color: str) -> tuple[float, float, float, float]:
    base = rgb_from_hex(color)
    edge_rgb = np.clip(base * 0.20, 0.0, 1.0)
    return (float(edge_rgb[0]), float(edge_rgb[1]), float(edge_rgb[2]), 0.46)


def grid_box_points(grid: OrbitalGrid) -> np.ndarray:
    high = grid.origin + grid.spacing * (np.array(grid.shape, dtype=np.float64) - 1.0)
    corners = np.array(
        [
            [grid.origin[0], grid.origin[1], grid.origin[2]],
            [high[0], grid.origin[1], grid.origin[2]],
            [grid.origin[0], high[1], grid.origin[2]],
            [grid.origin[0], grid.origin[1], high[2]],
            [high[0], high[1], grid.origin[2]],
            [high[0], grid.origin[1], high[2]],
            [grid.origin[0], high[1], high[2]],
            [high[0], high[1], high[2]],
        ],
        dtype=np.float64,
    )
    return corners * BOHR_TO_ANG


def mesh_faces(mesh: SurfaceMesh, face_limit: int = OPENGL_SURFACE_FACE_LIMIT) -> np.ndarray:
    if mesh.n_faces == 0:
        return np.empty((0, 3), dtype=np.uint32)
    stride = max(1, int(math.ceil(mesh.faces.shape[0] / face_limit)))
    return mesh.faces[::stride].astype(np.uint32, copy=False)


def mesh_item(mesh: SurfaceMesh, color: str) -> gl.GLMeshItem:
    faces = mesh_faces(mesh)
    triangles = mesh.vertices[faces] if faces.size else np.empty((0, 3, 3), dtype=np.float32)
    mesh_data = gl.MeshData(
        vertexes=mesh.vertices.astype(np.float32, copy=False),
        faces=faces,
        faceColors=glass_facecolors(triangles, color),
    )
    return gl.GLMeshItem(
        meshdata=mesh_data,
        smooth=False,
        drawFaces=True,
        drawEdges=False,
        edgeColor=glass_edgecolor(color),
        computeNormals=False,
        shader=FOG_FLAT_SHADER,
        glOptions="translucent",
    )


def atom_display_radius(atomic_number: int) -> float:
    covalent = COVALENT_RADII.get(int(atomic_number), 0.75)
    return max(0.13, min(0.33, 0.34 * covalent))


def _rgba_rows(rgb: tuple[float, float, float], count: int, alpha: float = 1.0) -> np.ndarray:
    rgba = np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.float32)
    return np.tile(rgba, (count, 1))


def _rotation_from_z_axis(direction: np.ndarray) -> np.ndarray:
    z_axis = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    dot = float(np.clip(z_axis @ unit, -1.0, 1.0))
    if dot > 0.999999:
        return np.eye(3, dtype=np.float64)
    if dot < -0.999999:
        return np.array(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), dtype=np.float64)
    cross = np.cross(z_axis, unit)
    skew = np.array(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        ),
        dtype=np.float64,
    )
    sin2 = float(cross @ cross)
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / sin2)


def cylinder_between(start: np.ndarray, end: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-8:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)
    base = gl.MeshData.cylinder(rows=1, cols=MOLECULE_BOND_COLS, radius=[radius, radius], length=length)
    rotation = _rotation_from_z_axis(vector)
    vertices = base.vertexes().astype(np.float64, copy=False) @ rotation.T + start
    return vertices.astype(np.float32, copy=False), base.faces().astype(np.uint32, copy=False)


def molecule_mesh_item(wavefunction: FCHKWavefunction, bonds: list[tuple[int, int]]) -> gl.GLMeshItem:
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    vertex_color_parts: list[np.ndarray] = []
    offset = 0

    coords = wavefunction.coordinates_angstrom.astype(np.float64, copy=False)
    for atom_idx, atomic_number in enumerate(wavefunction.atomic_numbers):
        z = int(atomic_number)
        if z == 0:
            continue
        color = ELEMENT_COLORS.get(z, (0.55, 0.58, 0.64))
        sphere = gl.MeshData.sphere(
            rows=MOLECULE_SPHERE_ROWS,
            cols=MOLECULE_SPHERE_COLS,
            radius=atom_display_radius(z),
        )
        vertices = sphere.vertexes().astype(np.float32, copy=True)
        vertices += coords[atom_idx].astype(np.float32, copy=False)
        faces = sphere.faces().astype(np.uint32, copy=False)
        vertices_parts.append(vertices)
        faces_parts.append(faces + offset)
        vertex_color_parts.append(_rgba_rows(color, vertices.shape[0]))
        offset += vertices.shape[0]

    bond_color = (0.42, 0.45, 0.50)
    for i, j in bonds:
        vertices, faces = cylinder_between(coords[i], coords[j], MOLECULE_BOND_RADIUS)
        if vertices.size == 0:
            continue
        vertices_parts.append(vertices)
        faces_parts.append(faces + offset)
        vertex_color_parts.append(_rgba_rows(bond_color, vertices.shape[0], alpha=0.96))
        offset += vertices.shape[0]

    if not vertices_parts:
        return gl.GLMeshItem()

    mesh_data = gl.MeshData(
        vertexes=np.vstack(vertices_parts).astype(np.float32, copy=False),
        faces=np.vstack(faces_parts).astype(np.uint32, copy=False),
        vertexColors=np.vstack(vertex_color_parts).astype(np.float32, copy=False),
    )
    return gl.GLMeshItem(
        meshdata=mesh_data,
        smooth=True,
        drawFaces=True,
        drawEdges=False,
        shader=FOG_SHADED_SHADER,
        glOptions="opaque",
    )


class CornerAxesWidget(QtWidgets.QWidget):
    def __init__(self, owner: "OpenGLViewer", slot: "SceneSlot | None" = None):
        super().__init__()
        self.owner = owner
        self.slot = slot
        self.setFixedSize(150, 150)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if getattr(self.owner, "corner_check", None) is not None and not self.owner.corner_check.isChecked():
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(SCENE_BACKGROUND_HEX))
        painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1"), 1.0))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)

        rotation = self.slot.scene_rotation if self.slot is not None else np.eye(3, dtype=np.float64)
        axes = np.eye(3, dtype=np.float64) @ rotation.T
        colors = (QtGui.QColor("#ef4444"), QtGui.QColor("#22c55e"), QtGui.QColor("#3b82f6"))
        labels = ("X", "Y", "Z")
        center = QtCore.QPointF(self.width() * 0.5, self.height() * 0.54)
        scale = min(self.width(), self.height()) * 0.34

        projected: list[tuple[float, int, np.ndarray]] = []
        for idx, vec in enumerate(axes):
            screen = np.array((vec[0] - 0.38 * vec[1], -vec[2] + 0.38 * vec[1]), dtype=np.float64)
            projected.append((float(vec[1]), idx, screen))
        painter.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Weight.Bold))
        for _depth, idx, screen in sorted(projected, key=lambda item: item[0]):
            length = float(np.linalg.norm(screen))
            if length < 1.0e-6:
                continue
            unit = screen / length
            end = center + QtCore.QPointF(float(screen[0] * scale), float(screen[1] * scale))
            painter.setPen(QtGui.QPen(colors[idx], 2.4, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
            painter.drawLine(center, end)

            arrow_base = end - QtCore.QPointF(float(unit[0] * 10.0), float(unit[1] * 10.0))
            normal = np.array((-unit[1], unit[0]), dtype=np.float64)
            points = [
                end,
                arrow_base + QtCore.QPointF(float(normal[0] * 4.5), float(normal[1] * 4.5)),
                arrow_base - QtCore.QPointF(float(normal[0] * 4.5), float(normal[1] * 4.5)),
            ]
            painter.setBrush(QtGui.QBrush(colors[idx]))
            painter.drawPolygon(QtGui.QPolygonF(points))

            label_pos = end + QtCore.QPointF(float(unit[0] * 9.0), float(unit[1] * 9.0))
            painter.setPen(QtGui.QPen(colors[idx], 1.0))
            painter.drawText(QtCore.QRectF(label_pos.x() - 8, label_pos.y() - 8, 16, 16), QtCore.Qt.AlignmentFlag.AlignCenter, labels[idx])


class ViewHost(QtWidgets.QWidget):
    def __init__(self, owner: "OpenGLViewer", slot: "SceneSlot | None" = None):
        super().__init__()
        self.slot = slot
        self.main_view = OrbitalGLView(owner, slot)
        self.corner_view = CornerAxesWidget(owner, slot)
        self.main_view.setParent(self)
        self.corner_view.setParent(self)
        self.corner_view.raise_()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.main_view.setGeometry(self.rect())
        size = min(150, max(110, self.width() // 6))
        margin = 18
        self.corner_view.setGeometry(self.width() - size - margin, self.height() - size - margin, size, size)


class OrbitalGLView(gl.GLViewWidget):
    def __init__(self, owner: "OpenGLViewer", slot: "SceneSlot | None" = None):
        super().__init__()
        self.owner = owner
        self.slot = slot
        self.setBackgroundColor(SCENE_BACKGROUND_HEX)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setCameraParams(distance=12.0, elevation=90.0, azimuth=-90.0, fov=18.0)
        self._dragging = False
        self._drag_mode = "rotate"
        self._last_pos: QtCore.QPointF | None = None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.owner.set_active_slot(self.slot)
        if self.owner.center_pick_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.owner.pick_rotation_center(self.slot, event.position())
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_mode = "rotate"
        elif event.button() in (QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton):
            self._drag_mode = "roll"
        else:
            event.ignore()
            return
        self._dragging = True
        self._last_pos = event.position()
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._dragging or self._last_pos is None:
            event.ignore()
            return
        pos = event.position()
        dx = float(pos.x() - self._last_pos.x())
        dy = float(pos.y() - self._last_pos.y())
        self._last_pos = pos
        self.owner.apply_mouse_rotation(self.slot, dx, dy, self._drag_mode)
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._dragging = False
        self._last_pos = None
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
            self.owner.zoom_view(1.12 if delta > 0 else 1.0 / 1.12, slot=self.slot)
        event.accept()


class SceneSlot:
    def __init__(self, owner: "OpenGLViewer", index: int):
        self.owner = owner
        self.index = index
        self.frame = QtWidgets.QWidget()
        self.frame.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.layout = QtWidgets.QVBoxLayout(self.frame)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.title_label = QtWidgets.QLabel("")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 10pt; color: #111827;")
        self.view_host = ViewHost(owner, self)
        self.view_host.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.view = self.view_host.main_view
        self.corner_view = self.view_host.corner_view
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.view_host, stretch=1)
        self.reset_state()

    def reset_state(self) -> None:
        self.scene_rotation = default_scene_rotation()
        self.base_center: np.ndarray | None = None
        self.base_radius = 1.0
        self.view_center: np.ndarray | None = None
        self.view_zoom = 1.0
        self.center_atom_idx: int | None = None
        self.surface_items: list[object] = []
        self.molecule_items: list[object] = []
        self.corner_items: list[object] = []
        self.scene_limit_arrays: list[np.ndarray] = []
        self.orbital_index0: int | None = None
        self.grid: OrbitalGrid | None = None
        self.level: float | None = None


class OpenGLViewer(QtWidgets.QMainWindow):
    def __init__(
        self,
        fchk_path: str | None,
        default_grid: int,
        default_iso: float,
        default_margin: float,
        prefetch_workers: int,
        auto_render: bool,
        file_format: str | None = None,
    ):
        super().__init__()
        self.setWindowTitle("Oribital Viewer - OpenGL")
        self.resize(1320, 860)
        self.setMinimumSize(1040, 680)

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prefetch_workers = max(1, int(prefetch_workers))
        self.prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self.wf: FCHKWavefunction | None = None
        self.bonds: list[tuple[int, int]] = []
        self.basis_cache: OrderedDict[tuple[int, float], BasisGrid] = OrderedDict()
        self.basis_cache_limit_bytes = 1_900 * 1024 * 1024
        self.basis_cache_lock = threading.RLock()
        self.grid_cache: OrbitalGrid | None = None
        self.current_pos: SurfaceMesh | None = None
        self.current_neg: SurfaceMesh | None = None
        self.render_cache: OrderedDict[
            tuple[str, int, int, float], tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]
        ] = OrderedDict()
        self.cache_limit = 90
        self.job_token = 0
        self.iso_timer: QtCore.QTimer | None = None
        self.orbital_timer: QtCore.QTimer | None = None
        self.future: Future | None = None
        self.prefetch_futures: dict[Future, tuple[tuple[int, int], ...]] = {}
        self.prefetch_queue: list[tuple[int, int]] = []
        self.prefetch_token = 0
        self.prefetch_total = 0
        self.prefetch_done = 0
        self.prefetch_spin: str | None = None
        self.prefetch_grid_size: int | None = None
        self.prefetch_margin: float | None = None
        self.slots: list[SceneSlot] = []
        self.visible_slot_count = 0
        self.primary_slot: SceneSlot | None = None
        self.active_slot: SceneSlot | None = None
        self.scene_rotation = default_scene_rotation()
        self.base_center: np.ndarray | None = None
        self.base_radius = 1.0
        self.view_center: np.ndarray | None = None
        self.view_zoom = 1.0
        self.center_pick_mode = False
        self.center_atom_idx: int | None = None
        self.surface_items: list[object] = []
        self.molecule_items: list[object] = []
        self.corner_items: list[object] = []
        self.scene_limit_arrays: list[np.ndarray] = []
        self.iso_upper = 0.12
        self.shortcuts: list[QtGui.QShortcut] = []

        self._build_ui(default_grid, default_iso, default_margin)
        self._install_shortcuts()
        if fchk_path:
            QtCore.QTimer.singleShot(
                80,
                lambda: self.load_fchk(fchk_path, auto_render=auto_render, file_format=file_format),
            )

    def _build_ui(self, default_grid: int, default_iso: float, default_margin: float) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f6fb; color: #111827; font-family: Arial; font-size: 12pt; }
            QFrame#side { background: #ffffff; }
            QLabel#header { font-size: 16pt; font-weight: 700; background: #ffffff; }
            QLabel.muted { color: #4b5563; background: #ffffff; }
            QPushButton { padding: 6px 10px; }
            QPushButton#accent { background: #2563eb; color: white; border: 1px solid #1d4ed8; border-radius: 4px; }
            QTreeWidget { background: white; alternate-background-color: #f8fafc; border: 1px solid #d1d5db; }
            QHeaderView::section { background: #eef2ff; color: #111827; padding: 4px; border: 0; }
            """
        )
        shell = QtWidgets.QWidget()
        root_layout = QtWidgets.QHBoxLayout(shell)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(shell)

        side = QtWidgets.QFrame(objectName="side")
        side.setFixedWidth(390)
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(18, 16, 18, 16)
        side_layout.setSpacing(10)
        root_layout.addWidget(side)

        header = QtWidgets.QLabel("Oribital Viewer")
        header.setObjectName("header")
        side_layout.addWidget(header)
        self.file_label = QtWidgets.QLabel("No wavefunction file loaded")
        self.file_label.setProperty("class", "muted")
        self.file_label.setWordWrap(True)
        side_layout.addWidget(self.file_label)

        button_row = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Open file")
        self.open_button.setObjectName("accent")
        self.open_button.clicked.connect(self.open_file)
        self.render_button = QtWidgets.QPushButton("Render")
        self.render_button.clicked.connect(self.render_selected)
        button_row.addWidget(self.open_button)
        button_row.addWidget(self.render_button)
        button_row.addStretch(1)
        side_layout.addLayout(button_row)

        meta_row = QtWidgets.QHBoxLayout()
        self.atom_label = QtWidgets.QLabel("Atoms: -")
        self.basis_label = QtWidgets.QLabel("Basis: -")
        self.atom_label.setProperty("class", "muted")
        self.basis_label.setProperty("class", "muted")
        meta_row.addWidget(self.atom_label)
        meta_row.addSpacing(18)
        meta_row.addWidget(self.basis_label)
        meta_row.addStretch(1)
        side_layout.addLayout(meta_row)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.spin_combo = QtWidgets.QComboBox()
        self.spin_combo.addItem("alpha")
        self.spin_combo.currentTextChanged.connect(lambda _text: self.populate_orbitals())
        self.grid_spin = QtWidgets.QSpinBox()
        self.grid_spin.setRange(56, 81)
        self.grid_spin.setSingleStep(2)
        self.grid_spin.setValue(max(56, min(81, int(default_grid))))
        self.grid_spin.valueChanged.connect(lambda _value: self.cancel_prefetch_work())
        self.margin_spin = QtWidgets.QDoubleSpinBox()
        self.margin_spin.setRange(1.0, 12.0)
        self.margin_spin.setSingleStep(0.5)
        self.margin_spin.setDecimals(2)
        self.margin_spin.setValue(default_margin)
        self.margin_spin.valueChanged.connect(lambda _value: self.cancel_prefetch_work())
        form.addRow("Spin", self.spin_combo)
        form.addRow("Grid", self.grid_spin)
        form.addRow("Margin / bohr", self.margin_spin)
        side_layout.addLayout(form)

        iso_header = QtWidgets.QHBoxLayout()
        iso_header.addWidget(QtWidgets.QLabel("Isovalue"))
        side_layout.addLayout(iso_header)
        self.iso_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.iso_slider.setRange(1, 1000)
        self.iso_slider.valueChanged.connect(self.on_iso_drag)
        side_layout.addWidget(self.iso_slider)
        iso_entry_row = QtWidgets.QHBoxLayout()
        self.iso_entry = QtWidgets.QLineEdit(f"{max(0.0001, default_iso if default_iso > 0 else 0.05):.5f}")
        self.iso_entry.returnPressed.connect(self.apply_iso_entry)
        self.apply_iso_button = QtWidgets.QPushButton("Apply")
        self.apply_iso_button.clicked.connect(self.apply_iso_entry)
        iso_entry_row.addWidget(self.iso_entry)
        iso_entry_row.addWidget(self.apply_iso_button)
        side_layout.addLayout(iso_entry_row)
        self.set_iso_slider_value(max(0.0001, default_iso if default_iso > 0 else 0.05), update_text=False)

        quick_row = QtWidgets.QHBoxLayout()
        self.homo_button = QtWidgets.QPushButton("HOMO")
        self.homo_button.clicked.connect(lambda: self.select_frontier("homo"))
        self.lumo_button = QtWidgets.QPushButton("LUMO")
        self.lumo_button.clicked.connect(lambda: self.select_frontier("lumo"))
        self.reset_button = QtWidgets.QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        quick_row.addWidget(self.homo_button)
        quick_row.addWidget(self.lumo_button)
        quick_row.addWidget(self.reset_button)
        side_layout.addLayout(quick_row)

        compare_row = QtWidgets.QHBoxLayout()
        compare_row.addWidget(QtWidgets.QLabel("Compare"))
        self.compare_entry = QtWidgets.QLineEdit()
        self.compare_entry.setPlaceholderText("1-based MOs, e.g. 45,46,47")
        self.compare_entry.returnPressed.connect(self.render_selected)
        compare_row.addWidget(self.compare_entry, stretch=1)
        side_layout.addLayout(compare_row)

        self.sync_views_check = QtWidgets.QCheckBox("Synchronized rotation")
        self.sync_views_check.setChecked(True)
        self.sync_views_check.stateChanged.connect(self.on_sync_views_changed)
        side_layout.addWidget(self.sync_views_check)

        zoom_row = QtWidgets.QHBoxLayout()
        zoom_row.addWidget(QtWidgets.QLabel("Zoom"))
        self.zoom_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(35, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider)
        zoom_row.addWidget(self.zoom_slider)
        side_layout.addLayout(zoom_row)

        self.corner_check = QtWidgets.QCheckBox("Display axes")
        self.corner_check.setChecked(True)
        self.corner_check.stateChanged.connect(lambda _state: self.update_corner_axes())
        side_layout.addWidget(self.corner_check)

        self.orbital_info_label = QtWidgets.QLabel("Select an orbital")
        self.orbital_info_label.setWordWrap(True)
        self.orbital_info_label.setProperty("class", "muted")
        side_layout.addWidget(self.orbital_info_label)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Orb", "Occ", "Energy / Eh", "eV"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.itemSelectionChanged.connect(self.update_orbital_info)
        self.tree.itemDoubleClicked.connect(lambda _item, _col: self.render_selected())
        self.tree.setColumnWidth(0, 58)
        self.tree.setColumnWidth(1, 58)
        self.tree.setColumnWidth(2, 112)
        side_layout.addWidget(self.tree, stretch=1)

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("class", "muted")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        side_layout.addWidget(self.status_label)
        side_layout.addWidget(self.progress)

        canvas_panel = QtWidgets.QWidget()
        canvas_layout = QtWidgets.QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(8, 8, 10, 8)
        canvas_layout.setSpacing(6)
        self.scene_title = QtWidgets.QLabel("Open a wavefunction file")
        self.scene_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scene_title.setStyleSheet("font-size: 12pt; color: #111827;")
        canvas_layout.addWidget(self.scene_title)
        self.scene_grid_widget = QtWidgets.QWidget()
        self.scene_grid = QtWidgets.QGridLayout(self.scene_grid_widget)
        self.scene_grid.setContentsMargins(0, 0, 0, 0)
        self.scene_grid.setSpacing(8)
        canvas_layout.addWidget(self.scene_grid_widget, stretch=1)
        root_layout.addWidget(canvas_panel, stretch=1)
        self.configure_scene_slots(1)
        self.draw_empty()

    def comparison_positions(self, count: int) -> list[tuple[int, int, int, int]]:
        if count <= 1:
            return [(0, 0, 1, 1)]
        if count == 2:
            return [(0, 0, 1, 1), (0, 1, 1, 1)]
        if count == 3:
            return [(0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 1, 1)]
        cols = 2 if count <= 4 else 3
        return [(idx // cols, idx % cols, 1, 1) for idx in range(count)]

    def configure_scene_slots(self, count: int) -> None:
        count = max(1, min(MAX_COMPARE_ORBITALS, int(count)))
        self.clear_all_gl_items()
        self.scene_grid_widget.setUpdatesEnabled(False)
        while len(self.slots) < count:
            self.slots.append(SceneSlot(self, len(self.slots)))
        try:
            while self.scene_grid.count():
                item = self.scene_grid.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
            for idx, slot in enumerate(self.slots):
                slot.frame.setVisible(idx < count)
            positions = self.comparison_positions(count)
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            for slot, (row, col, row_span, col_span) in zip(self.slots[:count], positions):
                self.scene_grid.addWidget(slot.frame, row, col, row_span, col_span)
                used_rows.update(range(row, row + row_span))
                used_cols.update(range(col, col + col_span))
            for row in range(3):
                self.scene_grid.setRowStretch(row, 1 if row in used_rows else 0)
                self.scene_grid.setRowMinimumHeight(row, 0)
            for col in range(3):
                self.scene_grid.setColumnStretch(col, 1 if col in used_cols else 0)
                self.scene_grid.setColumnMinimumWidth(col, 0)
        finally:
            self.scene_grid_widget.setUpdatesEnabled(True)
        self.visible_slot_count = count
        self.primary_slot = self.slots[0]
        if self.active_slot not in self.slots[:count]:
            self.active_slot = self.primary_slot
        self.view_host = self.primary_slot.view_host
        self.view = self.primary_slot.view
        self.corner_view = self.primary_slot.corner_view

    def clear_all_gl_items(self) -> None:
        for slot in self.slots:
            self.clear_scene(slot)
            self.clear_corner(slot)

    def visible_slots(self) -> list[SceneSlot]:
        return self.slots[: self.visible_slot_count]

    def slot_or_primary(self, slot: SceneSlot | None = None) -> SceneSlot:
        if slot is not None:
            return slot
        if self.active_slot is not None:
            return self.active_slot
        assert self.primary_slot is not None
        return self.primary_slot

    def set_active_slot(self, slot: SceneSlot | None) -> None:
        if slot is not None:
            self.active_slot = slot

    def sync_rotation_enabled(self) -> bool:
        return getattr(self, "sync_views_check", None) is not None and self.sync_views_check.isChecked()

    def target_slots(self, slot: SceneSlot | None = None) -> list[SceneSlot]:
        if self.sync_rotation_enabled():
            return self.visible_slots()
        return [self.slot_or_primary(slot)]

    def on_sync_views_changed(self, _state: int) -> None:
        if not self.sync_rotation_enabled() or not self.visible_slots():
            return
        reference = self.visible_slots()[0]
        for slot in self.visible_slots()[1:]:
            slot.scene_rotation = reference.scene_rotation.copy()
            slot.view_zoom = reference.view_zoom
            slot.view_center = None if reference.view_center is None else reference.view_center.copy()
        self.update_scene_transform()

    def _install_shortcuts(self) -> None:
        bindings = [
            ("A", lambda: self.select_relative_orbital(-1)),
            ("D", lambda: self.select_relative_orbital(1)),
            ("C", self.enable_center_pick),
            ("Left", lambda: self.rotate_shortcut("y", -7.0)),
            ("Right", lambda: self.rotate_shortcut("y", 7.0)),
            ("Up", lambda: self.rotate_shortcut("x", 6.0)),
            ("Down", lambda: self.rotate_shortcut("x", -6.0)),
            ("+", lambda: self.zoom_view(1.15)),
            ("=", lambda: self.zoom_view(1.15)),
            ("-", lambda: self.zoom_view(1.0 / 1.15)),
        ]
        for sequence, callback in bindings:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(lambda cb=callback: self.trigger_shortcut(cb))
            self.shortcuts.append(shortcut)

    def focus_accepts_text(self) -> bool:
        focus = QtWidgets.QApplication.focusWidget()
        parent = focus.parentWidget() if focus is not None else None
        while parent is not None:
            if isinstance(parent, QtWidgets.QAbstractSpinBox):
                return False
            parent = parent.parentWidget()
        return isinstance(
            focus,
            (
                QtWidgets.QLineEdit,
                QtWidgets.QTextEdit,
                QtWidgets.QPlainTextEdit,
            ),
        )

    def trigger_shortcut(self, callback) -> None:
        if self.focus_accepts_text():
            return
        callback()

    def rotate_shortcut(self, axis: str, angle_deg: float) -> None:
        self.apply_scene_rotation(angle_deg, axis)
        self.update_scene_transform()

    def open_file(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open wavefunction file",
            str(Path.cwd()),
            "Wavefunction files (*.fchk *.fch *.molden *.molden.input *.input);;"
            "Gaussian fchk (*.fchk *.fch);;Molden (*.molden *.molden.input *.input);;All files (*)",
        )
        if path:
            self.load_fchk(path, auto_render=True)

    def load_fchk(self, path: str, auto_render: bool, file_format: str | None = None) -> None:
        try:
            self.set_busy("Parsing wavefunction file...")
            QtWidgets.QApplication.processEvents()
            self.wf = parse_wavefunction(Path(path), file_format)
            self.cache_limit = max(240, min(1400, len(self.wf.alpha_energies) + 80))
            self.bonds = compute_bonds(self.wf.atomic_numbers, self.wf.coordinates_angstrom)
            self.grid_cache = None
            self.current_pos = None
            self.current_neg = None
            self.render_cache.clear()
            with self.basis_cache_lock:
                self.basis_cache.clear()
            self.cancel_prefetch_work()
            self.center_atom_idx = None
            self.base_center = None
            self.view_center = None
            self.scene_rotation = default_scene_rotation()
            self.clear_scene()
            for slot in self.slots:
                slot.reset_state()
            self.configure_scene_slots(1)
            self.view_zoom = 1.0
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(100)
            self.zoom_slider.blockSignals(False)
            self.update_base_view(self.wf.coordinates_angstrom, self.primary_slot)
            self.file_label.setText(f"{self.wf.path.name} ({self.wf.source_format})")
            self.atom_label.setText(f"Atoms: {len(self.wf.atomic_numbers)}")
            self.basis_label.setText(f"Basis: {self.wf.n_basis}")
            self.spin_combo.blockSignals(True)
            self.spin_combo.clear()
            self.spin_combo.addItems(["alpha", "beta"] if self.wf.is_unrestricted else ["alpha"])
            self.spin_combo.setCurrentText("alpha")
            self.spin_combo.blockSignals(False)
            self.populate_orbitals()
            self.draw_scene()
            self.set_ready("Loaded. Double-click an orbital or press Render.")
            if auto_render:
                QtCore.QTimer.singleShot(120, self.render_selected)
            QtCore.QTimer.singleShot(900, self.start_prefetch_common_orbitals)
        except Exception as exc:
            self.set_ready("Load failed")
            QtWidgets.QMessageBox.critical(self, "Failed to load wavefunction file", str(exc))

    def populate_orbitals(self) -> None:
        if self.wf is None:
            return
        spin = self.spin_combo.currentText()
        self.tree.clear()
        for idx, energy in enumerate(self.wf.energies(spin)):
            occ = self.wf.occupation(spin, idx)
            eh = "nan" if not np.isfinite(energy) else f"{energy:.6f}"
            ev = "nan" if not np.isfinite(energy) else f"{energy * HARTREE_TO_EV:.3f}"
            item = QtWidgets.QTreeWidgetItem([str(idx + 1), f"{occ:g}", eh, ev])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, idx)
            self.tree.addTopLevelItem(item)
        self.select_orbital(self.wf.default_orbital(spin))
        self.grid_cache = None
        self.current_pos = None
        self.current_neg = None
        self.cancel_prefetch_work()
        self.draw_scene()
        QtCore.QTimer.singleShot(500, self.start_prefetch_common_orbitals)

    def selected_orbital(self) -> int | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        value = items[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
        return int(value)

    def select_orbital(self, idx: int) -> None:
        if self.wf is None:
            return
        idx = max(0, min(idx, len(self.wf.energies(self.spin_combo.currentText())) - 1))
        item = self.tree.topLevelItem(idx)
        if item is None:
            return
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
        self.update_orbital_info()

    def select_frontier(self, which: str) -> None:
        if self.wf is None:
            return
        spin = self.spin_combo.currentText()
        idx = self.wf.default_orbital(spin) if which == "homo" else self.wf.lumo_orbital(spin)
        self.select_orbital(idx)
        self.render_selected()

    def update_orbital_info(self) -> None:
        if self.wf is None:
            self.orbital_info_label.setText("Select an orbital")
            return
        idx = self.selected_orbital()
        if idx is None:
            return
        spin = self.spin_combo.currentText()
        energy = self.wf.energies(spin)[idx]
        occ = self.wf.occupation(spin, idx)
        energy_text = "nan" if not np.isfinite(energy) else f"{energy:.6f} Eh / {energy * HARTREE_TO_EV:.3f} eV"
        self.orbital_info_label.setText(f"{spin} MO {idx + 1}: occ {occ:g}, E {energy_text}")

    def parse_compare_indices(self) -> list[int] | None:
        if self.wf is None:
            return None
        spin = self.spin_combo.currentText()
        n_orb = len(self.wf.energies(spin))
        text = self.compare_entry.text().strip()
        if not text:
            idx = self.selected_orbital()
            return [] if idx is None else [idx]
        indices: list[int] = []
        for token in re.split(r"[\s,;]+", text):
            if not token:
                continue
            if re.fullmatch(r"\d+\s*-\s*\d+", token):
                start_text, end_text = re.split(r"\s*-\s*", token)
                start = int(start_text)
                end = int(end_text)
                step = 1 if end >= start else -1
                indices.extend(range(start - 1, end - 1 + step, step))
            else:
                try:
                    indices.append(int(token) - 1)
                except ValueError:
                    QtWidgets.QMessageBox.critical(self, "Invalid comparison list", f"Invalid orbital token: {token!r}")
                    return None
        deduped: list[int] = []
        for idx in indices:
            if idx < 0 or idx >= n_orb:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid comparison list",
                    f"Orbital {idx + 1} is outside 1..{n_orb}.",
                )
                return None
            if idx not in deduped:
                deduped.append(idx)
        if len(deduped) > MAX_COMPARE_ORBITALS:
            QtWidgets.QMessageBox.critical(
                self,
                "Too many orbitals",
                f"Compare at most {MAX_COMPARE_ORBITALS} orbitals at once.",
            )
            return None
        return deduped

    def core_prefetch_bounds(self, spin: str) -> tuple[int, int]:
        assert self.wf is not None
        homo = self.wf.default_orbital(spin)
        lumo = self.wf.lumo_orbital(spin)
        n_orb = len(self.wf.energies(spin))
        return max(0, homo - CORE_PREFETCH_OCCUPIED_BACK), min(n_orb - 1, lumo + CORE_PREFETCH_VIRTUAL_FORWARD)

    def display_grid_for_orbital(self, spin: str, idx: int) -> int:
        return max(56, min(81, int(self.grid_spin.value())))

    def render_selected(self) -> None:
        if self.wf is None:
            return
        indices = self.parse_compare_indices()
        if not indices:
            return
        self.render_orbitals(indices)

    def render_orbitals(self, indices: list[int]) -> None:
        assert self.wf is not None
        margin = float(self.margin_spin.value())
        spin = self.spin_combo.currentText()
        iso = self.iso_value()
        self.configure_scene_slots(len(indices))
        for slot, idx in zip(self.visible_slots(), indices):
            slot.orbital_index0 = idx
            slot.title_label.setText(f"{spin} MO {idx + 1} | loading")
            self.clear_surfaces(slot)
            self.ensure_molecule_model(slot)
            self.update_scene_transform(slot)
        self.grid_cache = None
        ready: list[tuple[int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]] = []
        preview_ready: list[tuple[int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]] = []
        extract_from_cache: list[tuple[int, OrbitalGrid]] = []
        compute_specs: list[tuple[int, int, int]] = []
        for slot, idx in zip(self.visible_slots(), indices):
            grid_size = self.display_grid_for_orbital(spin, idx)
            key = self.cache_key(spin, idx, grid_size, margin)
            cached = self.render_cache.get(key)
            if cached is not None:
                self.render_cache.move_to_end(key)
                cached_grid, _cached_pos, _cached_neg, cached_level = cached
                if iso <= 0 or math.isclose(abs(iso), cached_level, rel_tol=1.0e-8, abs_tol=1.0e-10):
                    ready.append((slot.index, cached))
                else:
                    extract_from_cache.append((slot.index, cached_grid))
                continue
            low_key = self.cache_key(spin, idx, LOW_PREFETCH_GRID, margin)
            low_cached = self.render_cache.get(low_key)
            if low_cached is not None and LOW_PREFETCH_GRID < grid_size:
                self.render_cache.move_to_end(low_key)
                low_grid, _low_pos, _low_neg, low_level = low_cached
                if iso <= 0 or math.isclose(abs(iso), low_level, rel_tol=1.0e-8, abs_tol=1.0e-10):
                    preview_ready.append((slot.index, low_cached))
                else:
                    preview_ready.append((slot.index, self._extract_from_cached_grid(low_grid, iso)))
            compute_specs.append((slot.index, idx, grid_size))
        if not extract_from_cache and not compute_specs:
            self._finish_multi_render(ready, from_cache=True)
            return
        if preview_ready:
            self._finish_multi_render(ready + preview_ready, from_cache=True, restart_prefetch=False)
        self.submit_job(
            f"Rendering {len(indices)} orbital view(s)...",
            lambda: self._compute_multi_job(spin, extract_from_cache, compute_specs, margin, iso),
            lambda computed: self._finish_multi_render(ready + computed, from_cache=not compute_specs),
        )

    def _extract_from_cached_grid(self, grid: OrbitalGrid, iso: float) -> tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]:
        pos, neg, level = extract_isosurfaces(grid, iso)
        return grid, pos, neg, level

    def _compute_full_job(
        self,
        spin: str,
        idx: int,
        grid_size: int,
        margin: float,
        iso: float,
    ) -> tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]:
        assert self.wf is not None
        basis_grid = self.get_basis_grid(grid_size, margin)
        [grid] = compute_orbital_grids(self.wf, spin, [idx], grid_size, margin, basis_grid)
        pos, neg, level = extract_isosurfaces(grid, iso)
        return grid, pos, neg, level

    def _compute_multi_job(
        self,
        spin: str,
        extract_from_cache: list[tuple[int, OrbitalGrid]],
        compute_specs: list[tuple[int, int, int]],
        margin: float,
        iso: float,
    ) -> list[tuple[int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]]:
        assert self.wf is not None
        out: list[tuple[int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]] = []
        for slot_index, grid in extract_from_cache:
            pos, neg, level = extract_isosurfaces(grid, iso)
            out.append((slot_index, (grid, pos, neg, level)))
        by_grid: dict[int, list[tuple[int, int]]] = {}
        for slot_index, idx, grid_size in compute_specs:
            by_grid.setdefault(grid_size, []).append((slot_index, idx))
        for grid_size, specs in by_grid.items():
            indices = [idx for _slot_index, idx in specs]
            slot_by_idx = {idx: slot_index for slot_index, idx in specs}
            basis_grid = self.get_basis_grid(grid_size, margin)
            for grid in compute_orbital_grids(self.wf, spin, indices, grid_size, margin, basis_grid):
                pos, neg, level = extract_isosurfaces(grid, iso)
                out.append((slot_by_idx[grid.orbital_index0], (grid, pos, neg, level)))
        return out

    def _finish_full_render(
        self,
        result: tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float],
        from_cache: bool = False,
    ) -> None:
        grid, pos, neg, level = result
        self.grid_cache = grid
        self.current_pos = pos
        self.current_neg = neg
        self.touch_cache((grid.spin, grid.orbital_index0, grid.grid_size, round(grid.margin_bohr, 6)), result)
        self.set_iso_slider_value(level, update_text=True)
        self.update_iso_scale(grid)
        self.draw_scene(pos, neg, grid, level)
        prefix = "Cached" if from_cache else "Rendered"
        self.set_ready(f"{prefix} {pos.n_faces} positive and {neg.n_faces} negative triangles.")
        self.slot_or_primary().view.setFocus()
        QtCore.QTimer.singleShot(1200, self.start_prefetch_common_orbitals)

    def _finish_multi_render(
        self,
        results: list[tuple[int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]],
        from_cache: bool = False,
        restart_prefetch: bool = True,
    ) -> None:
        if not results:
            self.set_ready("No orbital rendered.")
            return
        results = sorted(results, key=lambda item: item[0])
        first_grid: OrbitalGrid | None = None
        total_pos = 0
        total_neg = 0
        for slot_index, result in results:
            if slot_index >= len(self.slots):
                continue
            grid, pos, neg, level = result
            slot = self.slots[slot_index]
            self.touch_cache(self.cache_key(grid.spin, grid.orbital_index0, grid.grid_size, grid.margin_bohr), result)
            self.draw_scene(pos, neg, grid, level, slot=slot)
            total_pos += pos.n_faces
            total_neg += neg.n_faces
            if first_grid is None:
                first_grid = grid
                self.grid_cache = grid
                self.current_pos = pos
                self.current_neg = neg
                self.set_iso_slider_value(level, update_text=True)
                self.update_iso_scale(grid)
        n_rendered = len(results)
        if n_rendered == 1 and first_grid is not None:
            prefix = "Cached" if from_cache else "Rendered"
            self.set_ready(f"{prefix} {total_pos} positive and {total_neg} negative triangles.")
        else:
            prefix = "Cached" if from_cache else "Rendered"
            self.set_ready(f"{prefix} {n_rendered} orbital views.")
        self.slot_or_primary().view.setFocus()
        if restart_prefetch:
            QtCore.QTimer.singleShot(1200, self.start_prefetch_common_orbitals)

    def update_iso_scale(self, grid: OrbitalGrid) -> None:
        vmax = float(np.nanmax(np.abs(grid.values)))
        self.iso_upper = max(grid.auto_iso * 4.0, vmax * 0.75, 0.001)
        self.set_iso_slider_value(float(self.iso_entry.text()), update_text=False)

    def iso_value(self) -> float:
        return max(1.0e-8, self.iso_slider.value() / 1000.0 * self.iso_upper)

    def set_iso_slider_value(self, value: float, update_text: bool) -> None:
        value = max(1.0e-8, min(float(value), self.iso_upper))
        slider_value = max(1, min(1000, int(round(value / self.iso_upper * 1000.0))))
        self.iso_slider.blockSignals(True)
        self.iso_slider.setValue(slider_value)
        self.iso_slider.blockSignals(False)
        if update_text:
            self.iso_entry.setText(f"{value:.5f}")

    def on_iso_drag(self, _value: int) -> None:
        self.iso_entry.setText(f"{self.iso_value():.5f}")
        if self.grid_cache is None:
            return
        if self.iso_timer is not None:
            self.iso_timer.stop()
        self.iso_timer = QtCore.QTimer(self)
        self.iso_timer.setSingleShot(True)
        self.iso_timer.timeout.connect(self.render_cached_iso)
        self.iso_timer.start(260)

    def apply_iso_entry(self) -> None:
        try:
            value = abs(float(self.iso_entry.text()))
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Invalid isovalue", "Isovalue must be numeric.")
            return
        self.set_iso_slider_value(value, update_text=True)
        self.render_cached_iso()

    def render_cached_iso(self) -> None:
        cached_slots = [(slot.index, slot.grid) for slot in self.visible_slots() if slot.grid is not None]
        if len(cached_slots) > 1:
            level = self.iso_value()
            self.submit_job(
                "Updating comparison isosurfaces...",
                lambda: self._compute_multi_job(self.spin_combo.currentText(), cached_slots, [], float(self.margin_spin.value()), level),
                lambda results: self._finish_multi_render(results, from_cache=True),
                show_progress=False,
            )
            return
        if self.grid_cache is None:
            return
        grid = self.grid_cache
        level = self.iso_value()
        self.submit_job(
            "Updating isosurface...",
            lambda: (*extract_isosurfaces(grid, level), grid),
            self._finish_cached_render,
            show_progress=False,
        )

    def _finish_cached_render(self, result: tuple[SurfaceMesh, SurfaceMesh, float, OrbitalGrid]) -> None:
        pos, neg, level, grid = result
        self.current_pos = pos
        self.current_neg = neg
        self.set_iso_slider_value(level, update_text=True)
        self.draw_scene(pos, neg, grid, level)
        self.set_ready(f"Rendered {pos.n_faces} positive and {neg.n_faces} negative triangles.")
        QtCore.QTimer.singleShot(1200, self.start_prefetch_common_orbitals)

    def submit_job(self, status: str, work, finish, show_progress: bool = True) -> None:
        self.job_token += 1
        token = self.job_token
        self.status_label.setText(status)
        if show_progress:
            self.progress.setRange(0, 0)
            self.progress.setFormat("Rendering...")
        if self.future is not None and not self.future.done():
            self.future.cancel()
        future = self.executor.submit(work)
        self.future = future
        QtCore.QTimer.singleShot(80, lambda: self.poll_future(token, future, finish))

    def poll_future(self, token: int, future: Future, finish) -> None:
        if not future.done():
            QtCore.QTimer.singleShot(80, lambda: self.poll_future(token, future, finish))
            return
        if token != self.job_token:
            return
        self.progress.setRange(0, 1)
        self.progress.setFormat("")
        self.progress.setValue(0)
        try:
            finish(future.result())
        except Exception as exc:
            self.set_ready("Render failed")
            QtWidgets.QMessageBox.critical(self, "Render failed", str(exc))

    def basis_key(self, grid_size: int, margin: float) -> tuple[int, float]:
        return (int(grid_size), round(float(margin), 6))

    def basis_cache_bytes_locked(self) -> int:
        return sum(grid.nbytes for grid in self.basis_cache.values())

    def prune_basis_cache_locked(self) -> None:
        while len(self.basis_cache) > 1 and self.basis_cache_bytes_locked() > self.basis_cache_limit_bytes:
            self.basis_cache.popitem(last=False)

    def has_basis_grid(self, grid_size: int, margin: float) -> bool:
        key = self.basis_key(grid_size, margin)
        with self.basis_cache_lock:
            return key in self.basis_cache

    def get_basis_grid(self, grid_size: int, margin: float) -> BasisGrid:
        assert self.wf is not None
        key = self.basis_key(grid_size, margin)
        with self.basis_cache_lock:
            cached = self.basis_cache.get(key)
            if cached is not None:
                self.basis_cache.move_to_end(key)
                return cached
            basis_grid = compute_basis_grid(
                self.wf,
                int(grid_size),
                float(margin),
                workers=max(1, self.prefetch_workers),
            )
            self.basis_cache[key] = basis_grid
            self.basis_cache.move_to_end(key)
            self.prune_basis_cache_locked()
            return basis_grid

    def current_cache_key(self, spin: str, idx: int) -> tuple[str, int, int, float]:
        return self.cache_key(spin, idx, int(self.grid_spin.value()), float(self.margin_spin.value()))

    def cache_key(self, spin: str, idx: int, grid_size: int, margin: float) -> tuple[str, int, int, float]:
        return (spin, idx, int(grid_size), round(float(margin), 6))

    def touch_cache(
        self,
        key: tuple[str, int, int, float],
        value: tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float],
    ) -> None:
        self.render_cache[key] = value
        self.render_cache.move_to_end(key)
        while len(self.render_cache) > self.cache_limit:
            self.render_cache.popitem(last=False)

    def cancel_prefetch_work(self) -> None:
        self.prefetch_token += 1
        self.prefetch_queue.clear()
        self.prefetch_total = 0
        self.prefetch_done = 0
        self.prefetch_spin = None
        self.prefetch_grid_size = None
        self.prefetch_margin = None
        for future in list(self.prefetch_futures):
            future.cancel()
        self.prefetch_futures.clear()

    def homo_centered_prefetch_order(self, homo: int, n_orb: int) -> list[int]:
        indices: list[int] = [homo]
        max_delta = max(homo, n_orb - 1 - homo)
        for delta in range(1, max_delta + 1):
            occupied_idx = homo - delta
            virtual_idx = homo + delta
            if occupied_idx >= 0:
                indices.append(occupied_idx)
            if virtual_idx < n_orb:
                indices.append(virtual_idx)
        return indices

    def prefetch_tasks(
        self,
        spin: str,
        homo: int,
        n_orb: int,
        core_grid_size: int,
    ) -> list[tuple[int, int]]:
        low_idx, high_idx = self.core_prefetch_bounds(spin)
        order = self.homo_centered_prefetch_order(homo, n_orb)
        selected = self.selected_orbital()
        if selected is not None and low_idx <= selected <= high_idx:
            order = [selected] + [idx for idx in order if idx != selected]
        return [(idx, int(core_grid_size)) for idx in order if low_idx <= idx <= high_idx]

    def update_prefetch_progress(self, text: str | None = None) -> None:
        if self.prefetch_total <= 0:
            return
        self.progress.setRange(0, self.prefetch_total)
        self.progress.setFormat("Pre-render %v/%m")
        self.progress.setValue(min(self.prefetch_done, self.prefetch_total))
        if text is not None:
            self.status_label.setText(text)

    def prefetch_future_for(
        self,
        spin: str,
        idx: int,
        grid_size: int,
        margin: float,
    ) -> Future | None:
        if (
            self.prefetch_spin != spin
            or self.prefetch_margin is None
            or not math.isclose(self.prefetch_margin, margin, rel_tol=0.0, abs_tol=1.0e-9)
        ):
            return None
        for future, tasks in self.prefetch_futures.items():
            if (idx, grid_size) in tasks and not future.cancelled():
                return future
        return None

    def wait_for_prefetch_render(
        self,
        future: Future,
        spin: str,
        idx: int,
        grid_size: int,
        margin: float,
        iso: float,
    ) -> None:
        self.job_token += 1
        token = self.job_token
        if self.future is not None and not self.future.done():
            self.future.cancel()
        self.status_label.setText(f"Waiting for pre-rendered MO {idx + 1}...")
        self.progress.setRange(0, 0)
        self.progress.setFormat("Waiting...")
        QtCore.QTimer.singleShot(
            80,
            lambda: self.poll_prefetch_render(token, future, spin, idx, grid_size, margin, iso),
        )

    def poll_prefetch_render(
        self,
        token: int,
        future: Future,
        spin: str,
        idx: int,
        grid_size: int,
        margin: float,
        iso: float,
    ) -> None:
        if not future.done():
            QtCore.QTimer.singleShot(
                80,
                lambda: self.poll_prefetch_render(token, future, spin, idx, grid_size, margin, iso),
            )
            return
        if token != self.job_token:
            return
        selected_result: tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float] | None = None
        count_prefetch_progress = future in self.prefetch_futures
        try:
            for result_idx, result_grid_size, result in future.result():
                key = self.cache_key(spin, result_idx, result_grid_size, margin)
                self.touch_cache(key, result)
                if count_prefetch_progress:
                    self.prefetch_done += 1
                self.maybe_apply_prefetch_result(spin, result_idx, result_grid_size, result)
                if result_idx == idx and result_grid_size == grid_size:
                    selected_result = result
        except Exception as exc:
            self.cancel_prefetch_work()
            self.set_ready("Render failed")
            QtWidgets.QMessageBox.critical(self, "Render failed", str(exc))
            return
        self.prefetch_futures.pop(future, None)
        if count_prefetch_progress:
            self.update_prefetch_progress(
                f"Pre-render cache: {self.prefetch_done}/{self.prefetch_total} orbitals ready."
            )
        self.progress.setRange(0, 1)
        self.progress.setFormat("")
        self.progress.setValue(0)
        if selected_result is None:
            self.submit_job(
                "Evaluating MO field and extracting isosurfaces...",
                lambda: self._compute_full_job(spin, idx, grid_size, margin, iso),
                self._finish_full_render,
            )
            return
        grid, _pos, _neg, cached_level = selected_result
        if iso <= 0 or math.isclose(abs(iso), cached_level, rel_tol=1.0e-8, abs_tol=1.0e-10):
            self._finish_full_render(selected_result, from_cache=True)
            return
        self.submit_job(
            "Updating isosurface from pre-rendered MO grid...",
            lambda: self._extract_from_cached_grid(grid, iso),
            self._finish_full_render,
            show_progress=False,
        )

    def maybe_apply_prefetch_result(
        self,
        spin: str,
        idx: int,
        result_grid_size: int,
        result: tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float],
    ) -> None:
        if self.wf is None or spin != self.spin_combo.currentText():
            return
        desired_grid_size = self.display_grid_for_orbital(spin, idx)
        if result_grid_size > desired_grid_size:
            return
        if result_grid_size != desired_grid_size and result_grid_size != LOW_PREFETCH_GRID:
            return
        grid, pos, neg, level = result
        if not math.isclose(abs(self.iso_value()), level, rel_tol=1.0e-8, abs_tol=1.0e-10):
            return
        updated_primary = False
        for slot in self.visible_slots():
            if slot.orbital_index0 != idx:
                continue
            current_grid_size = slot.grid.grid_size if slot.grid is not None else 0
            if current_grid_size >= result_grid_size:
                continue
            self.draw_scene(pos, neg, grid, level, slot=slot)
            if slot is self.primary_slot:
                updated_primary = True
        if updated_primary:
            self.grid_cache = grid
            self.current_pos = pos
            self.current_neg = neg

    def start_prefetch_common_orbitals(self) -> None:
        if self.wf is None:
            return
        if self.future is not None and not self.future.done():
            QtCore.QTimer.singleShot(1000, self.start_prefetch_common_orbitals)
            return
        spin = self.spin_combo.currentText()
        grid_size = int(self.grid_spin.value())
        margin = float(self.margin_spin.value())
        if not self.has_basis_grid(grid_size, margin):
            return
        if self.prefetch_queue or self.prefetch_futures:
            same_prefetch = (
                self.prefetch_spin == spin
                and self.prefetch_grid_size == grid_size
                and self.prefetch_margin is not None
                and math.isclose(self.prefetch_margin, margin, rel_tol=0.0, abs_tol=1.0e-9)
            )
            if same_prefetch:
                return
            self.cancel_prefetch_work()
        self.prefetch_token += 1
        token = self.prefetch_token
        homo = self.wf.default_orbital(spin)
        n_orb = len(self.wf.energies(spin))
        tasks = self.prefetch_tasks(spin, homo, n_orb, grid_size)
        self.prefetch_queue = [
            task
            for task in tasks
            if self.cache_key(spin, task[0], task[1], margin) not in self.render_cache
        ]
        self.prefetch_total = len(self.prefetch_queue)
        self.prefetch_done = 0
        for future in list(self.prefetch_futures):
            future.cancel()
        self.prefetch_futures.clear()
        self.prefetch_spin = spin
        self.prefetch_grid_size = grid_size
        self.prefetch_margin = margin
        if self.prefetch_total:
            self.update_prefetch_progress(
                f"Pre-render cache: 0/{self.prefetch_total} queued at grid {grid_size}."
            )
            self.pump_prefetch(token, spin, grid_size, margin)
        else:
            self.progress.setRange(0, 1)
            self.progress.setFormat("")
            self.progress.setValue(0)

    def pump_prefetch(self, token: int, spin: str, grid_size: int, margin: float) -> None:
        if token != self.prefetch_token or self.wf is None:
            return
        done_futures = [future for future in self.prefetch_futures if future.done()]
        for future in done_futures:
            try:
                for idx, result_grid_size, result in future.result():
                    key = self.cache_key(spin, idx, result_grid_size, margin)
                    self.touch_cache(key, result)
                    self.prefetch_done += 1
                    self.maybe_apply_prefetch_result(spin, idx, result_grid_size, result)
            except Exception:
                pass
            self.prefetch_futures.pop(future, None)
            self.update_prefetch_progress(
                f"Pre-render cache: {self.prefetch_done}/{self.prefetch_total} orbitals ready."
            )
        if self.future is not None and not self.future.done():
            QtCore.QTimer.singleShot(250, lambda: self.pump_prefetch(token, spin, grid_size, margin))
            return
        while self.prefetch_queue and len(self.prefetch_futures) < 1:
            batch_indices: list[tuple[int, int]] = []
            skipped_cached = False
            while self.prefetch_queue and len(batch_indices) < PREFETCH_BATCH_SIZE:
                task = self.prefetch_queue.pop(0)
                if self.cache_key(spin, task[0], task[1], margin) in self.render_cache:
                    self.prefetch_done += 1
                    skipped_cached = True
                    continue
                batch_indices.append(task)
            if skipped_cached:
                self.update_prefetch_progress(
                    f"Pre-render cache: {self.prefetch_done}/{self.prefetch_total} orbitals ready."
                )
            if not batch_indices:
                continue
            future = self.prefetch_executor.submit(
                self.prefetch_batch,
                spin,
                batch_indices,
                margin,
            )
            self.prefetch_futures[future] = tuple(batch_indices)
        if not self.prefetch_queue and not self.prefetch_futures:
            if self.prefetch_total:
                self.update_prefetch_progress(
                    f"Pre-render cache complete: {self.prefetch_done}/{self.prefetch_total} orbitals."
                )
            return
        QtCore.QTimer.singleShot(180, lambda: self.pump_prefetch(token, spin, grid_size, margin))

    def prefetch_batch(
        self,
        spin: str,
        tasks: list[tuple[int, int]],
        margin: float,
    ) -> list[tuple[int, int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]]:
        assert self.wf is not None
        out: list[tuple[int, int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]] = []
        by_grid: dict[int, list[int]] = {}
        for idx, grid_size in tasks:
            by_grid.setdefault(grid_size, []).append(idx)
        for grid_size, indices in by_grid.items():
            basis_grid = self.get_basis_grid(grid_size, margin)
            for grid in compute_orbital_grids(self.wf, spin, indices, grid_size, margin, basis_grid):
                pos, neg, level = extract_isosurfaces(grid, 0.0)
                out.append((grid.orbital_index0, grid.grid_size, (grid, pos, neg, level)))
        return out

    def clear_scene(self, slot: SceneSlot | None = None) -> None:
        slots = self.slots if slot is None else [slot]
        for scene_slot in slots:
            self.clear_surfaces(scene_slot)
            for item in scene_slot.molecule_items:
                try:
                    scene_slot.view.removeItem(item)
                except Exception:
                    pass
            scene_slot.molecule_items = []

    def clear_surfaces(self, slot: SceneSlot | None = None) -> None:
        scene_slot = self.slot_or_primary(slot)
        for item in scene_slot.surface_items:
            try:
                scene_slot.view.removeItem(item)
            except Exception:
                pass
        scene_slot.surface_items = []

    def clear_corner(self, slot: SceneSlot | None = None) -> None:
        scene_slot = self.slot_or_primary(slot)
        if hasattr(scene_slot.corner_view, "removeItem"):
            for item in scene_slot.corner_items:
                try:
                    scene_slot.corner_view.removeItem(item)
                except Exception:
                    pass
        scene_slot.corner_items = []
        scene_slot.corner_view.update()

    def add_surface_item(self, slot: SceneSlot, item) -> None:
        slot.view.addItem(item)
        slot.surface_items.append(item)

    def add_molecule_item(self, slot: SceneSlot, item) -> None:
        slot.view.addItem(item)
        slot.molecule_items.append(item)

    def ensure_molecule_model(self, slot: SceneSlot | None = None) -> None:
        scene_slot = self.slot_or_primary(slot)
        if self.wf is None or scene_slot.molecule_items:
            return
        self.add_molecule_item(scene_slot, molecule_mesh_item(self.wf, self.bonds))

    def draw_empty(self) -> None:
        self.configure_scene_slots(1)
        self.clear_scene()
        self.scene_title.setText("Open a wavefunction file")
        for slot in self.visible_slots():
            slot.title_label.setText("")
            slot.scene_limit_arrays = []
            self.update_camera(slot)
        self.update_corner_axes()

    def draw_scene(
        self,
        pos: SurfaceMesh | None = None,
        neg: SurfaceMesh | None = None,
        grid: OrbitalGrid | None = None,
        level: float | None = None,
        slot: SceneSlot | None = None,
    ) -> None:
        scene_slot = self.slot_or_primary(slot)
        self.clear_surfaces(scene_slot)
        points_for_limits: list[np.ndarray] = []

        if pos is not None and pos.n_faces:
            self.add_surface_item(scene_slot, mesh_item(pos, "#ef3b2c"))
        if neg is not None and neg.n_faces:
            self.add_surface_item(scene_slot, mesh_item(neg, "#2563eb"))

        if self.wf is not None:
            coords = self.wf.coordinates_angstrom
            points_for_limits.append(grid_box_points(grid) if grid is not None else coords)
            self.ensure_molecule_model(scene_slot)

        if self.wf is None:
            title = "Open a wavefunction file"
        elif grid is None:
            title = f"{self.wf.path.name} | Atoms {len(self.wf.atomic_numbers)} | Basis {self.wf.n_basis}"
        else:
            energy = self.wf.energies(grid.spin)[grid.orbital_index0]
            occ = self.wf.occupation(grid.spin, grid.orbital_index0)
            energy_text = "nan" if not np.isfinite(energy) else f"{energy:.6f} Eh ({energy * HARTREE_TO_EV:.3f} eV)"
            iso_text = "auto" if level is None else f"+/-{level:.5g}"
            title = (
                f"{grid.spin} MO {grid.orbital_index0 + 1} | occ {occ:g} | E {energy_text}\n"
                f"iso {iso_text} | grid {grid.shape} | red +psi, blue -psi"
            )
        scene_slot.title_label.setText(title)
        if scene_slot is self.primary_slot:
            if self.wf is not None:
                self.scene_title.setText("")
            else:
                self.scene_title.setText("Open a wavefunction file")
        scene_slot.scene_limit_arrays = [arr.copy() for arr in points_for_limits]
        if points_for_limits:
            self.update_base_view(np.vstack([arr.reshape(-1, 3) for arr in points_for_limits if arr.size]), scene_slot)
        scene_slot.grid = grid
        scene_slot.level = level
        self.update_scene_transform(scene_slot)

    def rotation_center(self, slot: SceneSlot | None = None) -> np.ndarray:
        scene_slot = self.slot_or_primary(slot)
        if scene_slot.view_center is not None:
            return scene_slot.view_center
        if scene_slot.base_center is not None:
            return scene_slot.base_center
        return np.zeros(3, dtype=np.float64)

    def scene_transform(self, slot: SceneSlot | None = None) -> pg.Transform3D:
        scene_slot = self.slot_or_primary(slot)
        center = self.rotation_center(scene_slot)
        rotation = scene_slot.scene_rotation
        translation = center - rotation @ center
        return pg.Transform3D(
            [
                [rotation[0, 0], rotation[0, 1], rotation[0, 2], translation[0]],
                [rotation[1, 0], rotation[1, 1], rotation[1, 2], translation[1]],
                [rotation[2, 0], rotation[2, 1], rotation[2, 2], translation[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def transform_points(self, points: np.ndarray, slot: SceneSlot | None = None) -> np.ndarray:
        if points.size == 0:
            return points.reshape((-1, 3)).astype(np.float64, copy=False)
        scene_slot = self.slot_or_primary(slot)
        center = self.rotation_center(scene_slot)
        pts = points.reshape((-1, 3)).astype(np.float64, copy=False)
        return (pts - center) @ scene_slot.scene_rotation.T + center

    def update_scene_transform(self, slot: SceneSlot | None = None) -> None:
        slots = self.visible_slots() if slot is None else [slot]
        for scene_slot in slots:
            transform = self.scene_transform(scene_slot)
            for item in scene_slot.molecule_items + scene_slot.surface_items:
                try:
                    item.setTransform(transform)
                except Exception:
                    pass
            self.update_camera(scene_slot)
            self.update_corner_axes(scene_slot)
            scene_slot.view.update()

    def apply_scene_rotation(self, angle_deg: float, axis: str, slot: SceneSlot | None = None) -> None:
        if abs(angle_deg) < 1.0e-9:
            return
        matrix = axis_rotation_matrix(axis, angle_deg)
        for scene_slot in self.target_slots(slot):
            scene_slot.scene_rotation = matrix @ scene_slot.scene_rotation

    def apply_mouse_rotation(self, slot: SceneSlot | None, dx: float, dy: float, drag_mode: str) -> None:
        target = self.slot_or_primary(slot)
        if drag_mode == "roll":
            self.apply_scene_rotation(dx * VMD_ROTATE_DEG_PER_PIXEL, "z", target)
        else:
            self.apply_scene_rotation(dy * VMD_ROTATE_DEG_PER_PIXEL, "x", target)
            self.apply_scene_rotation(dx * VMD_ROTATE_DEG_PER_PIXEL, "y", target)
        self.update_scene_transform(None if self.sync_rotation_enabled() else target)

    def update_base_view(self, points: np.ndarray, slot: SceneSlot | None = None) -> None:
        if points.size == 0:
            return
        scene_slot = self.slot_or_primary(slot)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        scene_slot.base_center = 0.5 * (mins + maxs)
        scene_slot.base_radius = max(1.0, 0.56 * float((maxs - mins).max()))
        if scene_slot.view_center is None:
            scene_slot.view_center = scene_slot.base_center.copy()
        if scene_slot is self.primary_slot:
            self.base_center = scene_slot.base_center
            self.base_radius = scene_slot.base_radius
            self.view_center = scene_slot.view_center

    def update_camera(self, slot: SceneSlot | None = None) -> None:
        scene_slot = self.slot_or_primary(slot)
        center = scene_slot.view_center if scene_slot.view_center is not None else scene_slot.base_center
        if center is None:
            center = np.zeros(3, dtype=np.float64)
        radius = max(0.1, scene_slot.base_radius / max(scene_slot.view_zoom, 0.05))
        fov = 18.0
        distance = max(4.0, radius / math.tan(math.radians(fov) * 0.5) * 1.22)
        qcenter = QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2]))
        update_fog_shader_params(distance, radius)
        scene_slot.view.setCameraParams(center=qcenter, distance=distance, elevation=90.0, azimuth=-90.0, fov=fov)

    def update_corner_axes(self, slot: SceneSlot | None = None) -> None:
        slots = self.visible_slots() if slot is None else [slot]
        for scene_slot in slots:
            self._update_corner_axes_for_slot(scene_slot)

    def _update_corner_axes_for_slot(self, slot: SceneSlot) -> None:
        slot.corner_view.setVisible(self.corner_check.isChecked())
        slot.corner_view.update()

    def reset_view(self) -> None:
        for slot in self.visible_slots():
            slot.scene_rotation = default_scene_rotation()
            slot.view_zoom = 1.0
            slot.view_center = None if slot.base_center is None else slot.base_center.copy()
            slot.center_atom_idx = None
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(100)
        self.zoom_slider.blockSignals(False)
        self.center_atom_idx = None
        self.update_scene_transform()

    def zoom_view(self, factor: float, slot: SceneSlot | None = None) -> None:
        targets = self.target_slots(slot)
        for scene_slot in targets:
            scene_slot.view_zoom = max(0.35, min(3.0, scene_slot.view_zoom * factor))
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(round(targets[0].view_zoom * 100.0)))
        self.zoom_slider.blockSignals(False)
        for scene_slot in targets:
            self.update_camera(scene_slot)

    def on_zoom_slider(self, value: int) -> None:
        zoom = max(0.35, min(3.0, value / 100.0))
        for slot in self.target_slots():
            slot.view_zoom = zoom
            self.update_camera(slot)

    def pick_rotation_center(self, slot: SceneSlot | None, pos: QtCore.QPointF) -> None:
        self.center_pick_mode = False
        if self.wf is None:
            return
        scene_slot = self.slot_or_primary(slot)
        idx = self.nearest_atom_at_pos(scene_slot, pos)
        if idx is None:
            self.set_ready("No atom near click. Press C and try again.")
            return
        scene_slot.center_atom_idx = idx
        scene_slot.view_center = self.wf.coordinates_angstrom[idx].astype(np.float64, copy=True)
        if self.sync_rotation_enabled():
            for target in self.visible_slots():
                target.center_atom_idx = idx
                target.view_center = scene_slot.view_center.copy()
        self.center_atom_idx = idx
        symbol = atom_symbol(int(self.wf.atomic_numbers[idx]))
        self.set_ready(f"Rotation center: {symbol}{idx + 1}")
        self.update_scene_transform(None if self.sync_rotation_enabled() else scene_slot)

    def nearest_atom_at_pos(self, slot: SceneSlot, pos: QtCore.QPointF) -> int | None:
        assert self.wf is not None
        coords = self.transform_points(self.wf.coordinates_angstrom, slot)
        try:
            width = max(1, slot.view.width())
            height = max(1, slot.view.height())
            viewport = slot.view.getViewport()
            projection = slot.view.projectionMatrix((0, 0, width, height), viewport)
            view_matrix = slot.view.viewMatrix()
            matrix = projection * view_matrix
            xy = []
            for coord in coords:
                mapped = matrix.map(QtGui.QVector3D(float(coord[0]), float(coord[1]), float(coord[2])))
                sx = (mapped.x() + 1.0) * 0.5 * width
                sy = (1.0 - mapped.y()) * 0.5 * height
                xy.append((sx, sy))
            xy_arr = np.asarray(xy, dtype=np.float64)
        except Exception:
            center = slot.view_center if slot.view_center is not None else np.zeros(3, dtype=np.float64)
            scale = min(slot.view.width(), slot.view.height()) / max(slot.base_radius * 2.4, 1.0)
            xy_arr = np.column_stack(
                (
                    slot.view.width() * 0.5 + (coords[:, 0] - center[0]) * scale,
                    slot.view.height() * 0.5 - (coords[:, 1] - center[1]) * scale,
                )
            )
        target = np.array([float(pos.x()), float(pos.y())], dtype=np.float64)
        distances = np.linalg.norm(xy_arr - target, axis=1)
        idx = int(np.argmin(distances))
        return idx if float(distances[idx]) <= 36.0 else None

    def enable_center_pick(self) -> None:
        self.center_pick_mode = True
        self.set_ready("Center pick: click an atom in the preview.")

    def select_relative_orbital(self, delta: int) -> None:
        if self.wf is None:
            return
        idx = self.selected_orbital()
        if idx is None:
            idx = self.wf.default_orbital(self.spin_combo.currentText())
        self.select_orbital(idx + delta)
        if self.orbital_timer is not None:
            self.orbital_timer.stop()
        self.orbital_timer = QtCore.QTimer(self)
        self.orbital_timer.setSingleShot(True)
        self.orbital_timer.timeout.connect(self.render_selected)
        self.orbital_timer.start(140)
        self.slot_or_primary().view.setFocus()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self.focus_accepts_text():
            super().keyPressEvent(event)
            return
        key = event.key()
        if key == QtCore.Qt.Key.Key_A:
            self.select_relative_orbital(-1)
        elif key == QtCore.Qt.Key.Key_D:
            self.select_relative_orbital(1)
        elif key == QtCore.Qt.Key.Key_C:
            self.enable_center_pick()
        elif key == QtCore.Qt.Key.Key_Left:
            self.apply_scene_rotation(-7.0, "y")
            self.update_scene_transform()
        elif key == QtCore.Qt.Key.Key_Right:
            self.apply_scene_rotation(7.0, "y")
            self.update_scene_transform()
        elif key == QtCore.Qt.Key.Key_Up:
            self.apply_scene_rotation(6.0, "x")
            self.update_scene_transform()
        elif key == QtCore.Qt.Key.Key_Down:
            self.apply_scene_rotation(-6.0, "x")
            self.update_scene_transform()
        elif key in (QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal):
            self.zoom_view(1.15)
        elif key == QtCore.Qt.Key.Key_Minus:
            self.zoom_view(1.0 / 1.15)
        else:
            super().keyPressEvent(event)

    def set_busy(self, text: str) -> None:
        self.status_label.setText(text)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Working...")

    def set_ready(self, text: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setFormat("")
        self.progress.setValue(0)
        self.status_label.setText(text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.job_token += 1
        self.prefetch_token += 1
        if self.iso_timer is not None:
            self.iso_timer.stop()
        if self.orbital_timer is not None:
            self.orbital_timer.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.prefetch_executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def run_gui(
    fchk_path: str | None,
    default_grid: int,
    default_iso: float,
    default_margin: float,
    prefetch_workers: int,
    auto_render: bool,
    file_format: str | None = None,
) -> int:
    _require_gui_dependencies()
    app = QtWidgets.QApplication.instance()
    owned_app = app is None
    if app is None:
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
        app = QtWidgets.QApplication(sys.argv[:1])
    window = OpenGLViewer(
        fchk_path,
        default_grid,
        default_iso,
        default_margin,
        prefetch_workers,
        auto_render,
        file_format=file_format,
    )
    window.show()
    return int(app.exec()) if owned_app else 0
