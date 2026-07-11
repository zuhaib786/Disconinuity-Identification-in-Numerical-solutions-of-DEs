"""Validated triangular meshes and cell-adjacency graphs for 2D solvers."""

import numpy as np
from scipy.spatial import Delaunay, QhullError


class TriangleMesh:
    """Affine triangular mesh with deterministic face topology.

    Cells are normalized to counter-clockwise vertex order. Local face ``f``
    is the directed edge ``cells[:, f] -> cells[:, (f + 1) % 3]``.
    """

    def __init__(self, points, cells):
        points = np.asarray(points, dtype=float)
        raw_cells = np.asarray(cells)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("points must have shape (Nv, 2) with Nv >= 3")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("duplicate point coordinates are not supported")
        if raw_cells.ndim != 2 or raw_cells.shape[1] != 3 or len(raw_cells) == 0:
            raise ValueError("cells must have shape (K, 3) with K >= 1")
        if not np.issubdtype(raw_cells.dtype, np.integer):
            if not np.all(np.equal(raw_cells, np.floor(raw_cells))):
                raise ValueError("cell connectivity must contain integer indices")
        cells = raw_cells.astype(np.int64)
        if np.any(cells < 0) or np.any(cells >= len(points)):
            raise ValueError("cell connectivity contains an out-of-range vertex")
        if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
            raise ValueError("each triangle must reference three distinct vertices")
        canonical = np.sort(cells, axis=1)
        if len(np.unique(canonical, axis=0)) != len(cells):
            raise ValueError("duplicate triangles are not supported")
        if len(np.unique(cells)) != len(points):
            raise ValueError("every point must be referenced by a triangle")

        vertices = points[cells]
        edge01 = vertices[:, 1] - vertices[:, 0]
        edge02 = vertices[:, 2] - vertices[:, 0]
        signed_twice_area = edge01[:, 0] * edge02[:, 1] - edge01[:, 1] * edge02[:, 0]
        all_edges = np.stack(
            [
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 1],
                vertices[:, 0] - vertices[:, 2],
            ],
            axis=1,
        )
        max_edge_sq = np.max(np.sum(all_edges * all_edges, axis=2), axis=1)
        tolerance = 64.0 * np.finfo(float).eps * max_edge_sq
        if np.any(np.abs(signed_twice_area) <= tolerance):
            raise ValueError("mesh contains a degenerate triangle")
        clockwise = signed_twice_area < 0
        cells[clockwise, 1], cells[clockwise, 2] = (
            cells[clockwise, 2].copy(),
            cells[clockwise, 1].copy(),
        )

        self.points = points
        self.cells = cells
        self.K = len(cells)
        self.face_vertices = np.stack(
            [cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]], axis=1
        )

        self.neighbors = np.full((self.K, 3), -1, dtype=np.int64)
        self.neighbor_faces = np.full((self.K, 3), -1, dtype=np.int64)
        occurrences = {}
        for cell in range(self.K):
            for face in range(3):
                edge = self.face_vertices[cell, face]
                key = tuple(sorted((int(edge[0]), int(edge[1]))))
                occurrences.setdefault(key, []).append((cell, face))

        interior, boundary, graph_edges = [], [], []
        for key in sorted(occurrences):
            uses = occurrences[key]
            if len(uses) > 2:
                raise ValueError(f"non-manifold edge {key} belongs to {len(uses)} cells")
            if len(uses) == 1:
                boundary.append(uses[0])
                continue
            (ka, fa), (kb, fb) = uses
            edge_a = self.face_vertices[ka, fa]
            edge_b = self.face_vertices[kb, fb]
            if not np.array_equal(edge_a, edge_b[::-1]):
                raise ValueError("interior faces do not have opposite orientation")
            self.neighbors[ka, fa], self.neighbor_faces[ka, fa] = kb, fb
            self.neighbors[kb, fb], self.neighbor_faces[kb, fb] = ka, fa
            if kb < ka:
                ka, fa, kb, fb = kb, fb, ka, fa
            interior.append((ka, fa, kb, fb))
            graph_edges.append((ka, kb))

        self.interior_faces = np.asarray(interior, dtype=np.int64).reshape(-1, 4)
        self.boundary_faces = np.asarray(boundary, dtype=np.int64).reshape(-1, 2)
        self.graph_edges = np.asarray(sorted(graph_edges), dtype=np.int64).reshape(-1, 2)

        vertices = points[cells]
        self.centroids = np.mean(vertices, axis=1)
        directed_edges = np.stack(
            [
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 1],
                vertices[:, 0] - vertices[:, 2],
            ],
            axis=1,
        )
        self.edge_lengths = np.linalg.norm(directed_edges, axis=2)
        self.face_normals = np.stack(
            [directed_edges[:, :, 1], -directed_edges[:, :, 0]], axis=2
        ) / self.edge_lengths[:, :, None]
        twice_area = (
            (vertices[:, 1, 0] - vertices[:, 0, 0])
            * (vertices[:, 2, 1] - vertices[:, 0, 1])
            - (vertices[:, 1, 1] - vertices[:, 0, 1])
            * (vertices[:, 2, 0] - vertices[:, 0, 0])
        )
        self.areas = 0.5 * twice_area
        perimeter = np.sum(self.edge_lengths, axis=1)
        self.inradii = 2.0 * self.areas / perimeter
        self.circumradii = np.prod(self.edge_lengths, axis=1) / (4.0 * self.areas)
        self.radius_ratio = np.clip(2.0 * self.inradii / self.circumradii, 0.0, 1.0)
        self.skewness = 1.0 - self.radius_ratio

    @property
    def face_points(self):
        """Physical endpoints with shape ``(K, 3, 2, 2)``."""
        return self.points[self.face_vertices]

    def graph_edge_index(self):
        """Bidirectional cell-adjacency edges with shape ``(2, 2*Ni)``."""
        if not len(self.graph_edges):
            return np.empty((2, 0), dtype=np.int64)
        directed = np.concatenate([self.graph_edges, self.graph_edges[:, ::-1]])
        return directed.T

    def geometry_features(self, dimensionless=True):
        """Area, edge lengths, radii, and radius-ratio skewness per cell."""
        if dimensionless:
            root_area = np.sqrt(self.areas)[:, None]
            area = (self.areas / np.mean(self.areas))[:, None]
            return np.column_stack(
                [
                    area,
                    self.edge_lengths / root_area,
                    self.inradii[:, None] / root_area,
                    self.circumradii[:, None] / root_area,
                    self.skewness,
                ]
            )
        return np.column_stack(
            [
                self.areas,
                self.edge_lengths,
                self.inradii,
                self.circumradii,
                self.skewness,
            ]
        )

    def periodic_face_map(self, axes=(True, True), tolerance=None):
        """Pair opposite faces of an axis-aligned rectangular boundary.

        Returns ``(neighbors, neighbor_faces)`` arrays shaped ``(K, 3)``;
        entries not paired periodically remain ``-1``.
        """
        if len(axes) != 2:
            raise ValueError("axes must contain (x_periodic, y_periodic)")
        lower = np.min(self.points, axis=0)
        upper = np.max(self.points, axis=0)
        span = upper - lower
        if np.any(span <= 0):
            raise ValueError("periodic mesh must span both coordinate axes")
        tol = float(tolerance) if tolerance is not None else 1e-10 * float(np.max(span))
        if tol <= 0:
            raise ValueError("tolerance must be positive")

        periodic_neighbors = np.full((self.K, 3), -1, dtype=np.int64)
        periodic_faces = np.full((self.K, 3), -1, dtype=np.int64)
        face_points = self.face_points
        for axis, enabled in enumerate(axes):
            if not enabled:
                continue
            low, high = [], []
            for cell, face in self.boundary_faces:
                coordinates = face_points[cell, face, :, axis]
                if np.all(np.abs(coordinates - lower[axis]) <= tol):
                    low.append((int(cell), int(face)))
                elif np.all(np.abs(coordinates - upper[axis]) <= tol):
                    high.append((int(cell), int(face)))
            transverse = 1 - axis
            low.sort(key=lambda item: float(np.mean(face_points[item[0], item[1], :, transverse])))
            high.sort(key=lambda item: float(np.mean(face_points[item[0], item[1], :, transverse])))
            if not low or len(low) != len(high):
                raise ValueError(f"opposite axis-{axis} boundaries do not have matching faces")
            shift = np.zeros(2)
            shift[axis] = span[axis]
            for (ka, fa), (kb, fb) in zip(low, high):
                translated = face_points[ka, fa] + shift
                if not np.allclose(translated, face_points[kb, fb][::-1], atol=tol, rtol=0.0):
                    raise ValueError(f"opposite axis-{axis} boundary faces do not align")
                periodic_neighbors[ka, fa], periodic_faces[ka, fa] = kb, fb
                periodic_neighbors[kb, fb], periodic_faces[kb, fb] = ka, fa
        return periodic_neighbors, periodic_faces


def rectangular_mesh(xlim=(0.0, 1.0), ylim=(0.0, 1.0), nx=10, ny=10, diagonal="alternating"):
    """Split an ``nx`` by ``ny`` rectangular grid into CCW triangles."""
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    if diagonal not in ("left", "right", "alternating"):
        raise ValueError("diagonal must be 'left', 'right', or 'alternating'")
    xmin, xmax = map(float, xlim)
    ymin, ymax = map(float, ylim)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("mesh limits must be strictly increasing")

    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    cells = []
    for j in range(ny):
        for i in range(nx):
            bl = j * (nx + 1) + i
            br, tl = bl + 1, bl + nx + 1
            tr = tl + 1
            mode = diagonal
            if mode == "alternating":
                mode = "right" if (i + j) % 2 == 0 else "left"
            if mode == "right":
                cells.extend([(bl, br, tr), (bl, tr, tl)])
            else:
                cells.extend([(bl, br, tl), (br, tr, tl)])
    return TriangleMesh(points, cells)


def delaunay_mesh(points):
    """Triangulate caller-provided points with SciPy/Qhull."""
    points = np.asarray(points, dtype=float)
    try:
        cells = Delaunay(points).simplices
    except QhullError as exc:
        raise ValueError("points do not define a valid 2D Delaunay mesh") from exc
    return TriangleMesh(points, cells)


def random_delaunay_mesh(
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    n_interior=100,
    boundary_divisions=(10, 10),
    seed=None,
):
    """Seeded random Delaunay mesh whose convex hull is the given rectangle."""
    if n_interior < 0:
        raise ValueError("n_interior must be nonnegative")
    bx, by = boundary_divisions
    if bx < 1 or by < 1:
        raise ValueError("boundary divisions must be positive")
    xmin, xmax = map(float, xlim)
    ymin, ymax = map(float, ylim)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("mesh limits must be strictly increasing")

    x = np.linspace(xmin, xmax, bx + 1)
    y = np.linspace(ymin, ymax, by + 1)
    boundary = np.concatenate(
        [
            np.column_stack([x, np.full_like(x, ymin)]),
            np.column_stack([x, np.full_like(x, ymax)]),
            np.column_stack([np.full(max(by - 1, 0), xmin), y[1:-1]]),
            np.column_stack([np.full(max(by - 1, 0), xmax), y[1:-1]]),
        ]
    )
    rng = np.random.default_rng(seed)
    interior = rng.uniform((xmin, ymin), (xmax, ymax), size=(n_interior, 2))
    return delaunay_mesh(np.concatenate([boundary, interior]))


def perturbed_delaunay_mesh(
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    nx=10,
    ny=10,
    jitter=0.2,
    seed=None,
):
    """Quality-controlled unstructured mesh from jittered Cartesian points.

    Boundary points remain fixed and interior points are displaced by at most
    ``jitter`` times the local grid spacing before Delaunay triangulation.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    if not 0.0 <= jitter < 0.5:
        raise ValueError("jitter must satisfy 0 <= jitter < 0.5")
    xmin, xmax = map(float, xlim)
    ymin, ymax = map(float, ylim)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("mesh limits must be strictly increasing")
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    if nx > 1 and ny > 1 and jitter > 0:
        interior = np.array(
            [
                j * (nx + 1) + i
                for j in range(1, ny)
                for i in range(1, nx)
            ],
            dtype=np.int64,
        )
        rng = np.random.default_rng(seed)
        displacement = rng.uniform(-1.0, 1.0, size=(len(interior), 2))
        displacement *= jitter * np.array([(xmax - xmin) / nx, (ymax - ymin) / ny])
        points[interior] += displacement
    return delaunay_mesh(points)


def forward_step_mesh(nx=30, ny=10, step_x=0.6, step_y=0.2):
    """Dependency-free triangular mesh of the classic forward-step domain."""
    x = np.linspace(0.0, 3.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    ix = int(round(step_x / (3.0 / nx)))
    iy = int(round(step_y / (1.0 / ny)))
    if not np.isclose(x[ix], step_x) or not np.isclose(y[iy], step_y):
        raise ValueError("nx and ny must place vertices on the requested step corner")
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    cells = []
    for j in range(ny):
        for i in range(nx):
            if i >= ix and j < iy:
                continue
            bl = j * (nx + 1) + i
            br, tl, tr = bl + 1, bl + nx + 1, bl + nx + 2
            if (i + j) % 2 == 0:
                cells.extend([(bl, br, tr), (bl, tr, tl)])
            else:
                cells.extend([(bl, br, tl), (br, tr, tl)])
    cells = np.asarray(cells, dtype=np.int64)
    used = np.unique(cells)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return TriangleMesh(points[used], remap[cells])
