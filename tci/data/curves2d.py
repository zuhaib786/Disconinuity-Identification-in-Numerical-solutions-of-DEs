"""Signed-distance curve family for the `data-v3` generator (plan 6.2).

Every curve owns a region and reports an (at least first-order accurate) signed
distance that is negative inside it.  Sharp cut-curve fields use the Heaviside
of the distance; the steep-layer continuum replaces it by a ``tanh`` profile of
width ``delta``.  Corners, endpoints, near-tangencies, and multiple interacting
fronts all come from this family, which the single line/circle training data of
the frozen Phase 0--4 evidence never contained.
"""

import numpy as np

CURVE_TYPES = ("line", "circle", "ellipse", "strip", "polygon", "slotted")
LEGACY_CURVE_TYPES = ("line", "circle")


class Curve:
    """A region of the plane with a signed distance, negative inside."""

    kind = "curve"

    def __init__(self, **parameters):
        self.parameters = parameters

    def distance(self, x, y):
        raise NotImplementedError

    def inside_fraction(self, mesh, barycentric, area_weights):
        """Quadrature estimate of the inside-area fraction of every cell."""
        vertices = mesh.points[mesh.cells]
        fraction = np.zeros(mesh.K)
        for point, weight in zip(barycentric, area_weights):
            xy = np.einsum("i,kid->kd", point, vertices)
            fraction += weight * (self.distance(xy[:, 0], xy[:, 1]) < 0.0)
        return fraction

    def as_json(self):
        return {
            "kind": self.kind,
            **{
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in self.parameters.items()
            },
        }


class Line(Curve):
    """Half-plane behind an infinite line; the legacy training geometry."""

    kind = "line"

    def __init__(self, point, normal):
        super().__init__(point=np.asarray(point, float), normal=np.asarray(normal, float))

    def distance(self, x, y):
        point = self.parameters["point"]
        normal = self.parameters["normal"]
        return (x - point[0]) * normal[0] + (y - point[1]) * normal[1]


class Circle(Curve):
    """Disk; the other legacy training geometry."""

    kind = "circle"

    def __init__(self, center, radius):
        super().__init__(center=np.asarray(center, float), radius=float(radius))

    def distance(self, x, y):
        center = self.parameters["center"]
        return np.hypot(x - center[0], y - center[1]) - self.parameters["radius"]


class Ellipse(Curve):
    """Ellipse with aspect ratio in [1, 4]; curvature varies along the front."""

    kind = "ellipse"

    def __init__(self, center, semi_axes, angle):
        super().__init__(
            center=np.asarray(center, float),
            semi_axes=np.asarray(semi_axes, float),
            angle=float(angle),
        )

    def distance(self, x, y):
        center = self.parameters["center"]
        a, b = self.parameters["semi_axes"]
        angle = self.parameters["angle"]
        cosine, sine = np.cos(angle), np.sin(angle)
        dx, dy = x - center[0], y - center[1]
        u = cosine * dx + sine * dy
        v = -sine * dx + cosine * dy
        # First-order distance f/|grad f| of the implicit ellipse f = 0.
        f = (u / a) ** 2 + (v / b) ** 2 - 1.0
        gradient = 2.0 * np.hypot(u / a**2, v / b**2)
        return f / np.maximum(gradient, 1e-12)


class Strip(Curve):
    """Band between two parallel lines; thin bands are sub-cell features."""

    kind = "strip"

    def __init__(self, point, normal, width):
        super().__init__(
            point=np.asarray(point, float),
            normal=np.asarray(normal, float),
            width=float(width),
        )

    def distance(self, x, y):
        point = self.parameters["point"]
        normal = self.parameters["normal"]
        offset = (x - point[0]) * normal[0] + (y - point[1]) * normal[1]
        return np.abs(offset) - 0.5 * self.parameters["width"]


class Polygon(Curve):
    """Simple polygon: 3--6 straight edges meeting at visible corners."""

    kind = "polygon"

    def __init__(self, vertices):
        vertices = np.asarray(vertices, float)
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
            raise ValueError("polygon needs at least three (x, y) vertices")
        super().__init__(vertices=vertices)

    def distance(self, x, y):
        vertices = self.parameters["vertices"]
        following = np.roll(vertices, -1, axis=0)
        edges = following - vertices
        shape = np.shape(x)
        px = np.reshape(x, (-1, 1))
        py = np.reshape(y, (-1, 1))
        to_vertex = np.stack([px - vertices[:, 0], py - vertices[:, 1]], axis=2)
        projection = np.clip(
            np.sum(to_vertex * edges, axis=2) / np.sum(edges * edges, axis=1), 0.0, 1.0
        )
        closest = to_vertex - projection[:, :, None] * edges
        distance = np.min(np.linalg.norm(closest, axis=2), axis=1)
        # Even-odd crossing rule against upward/downward edges.
        straddles = (vertices[:, 1] > py) != (following[:, 1] > py)
        crossing_x = vertices[:, 0] + (py - vertices[:, 1]) * edges[:, 0] / edges[:, 1]
        with np.errstate(invalid="ignore"):
            crossings = straddles & (px < crossing_x)
        inside = np.count_nonzero(crossings, axis=1) % 2 == 1
        return np.reshape(np.where(inside, -distance, distance), shape)


class SlottedDisk(Curve):
    """Disk minus a rectangular slot: the deployment geometry's corner family."""

    kind = "slotted"

    def __init__(self, center, radius, angle, slot_width, slot_depth):
        super().__init__(
            center=np.asarray(center, float),
            radius=float(radius),
            angle=float(angle),
            slot_width=float(slot_width),
            slot_depth=float(slot_depth),
        )

    def distance(self, x, y):
        center = self.parameters["center"]
        radius = self.parameters["radius"]
        angle = self.parameters["angle"]
        width = self.parameters["slot_width"]
        depth = self.parameters["slot_depth"]
        disk = np.hypot(x - center[0], y - center[1]) - radius
        cosine, sine = np.cos(angle), np.sin(angle)
        dx, dy = x - center[0], y - center[1]
        u = cosine * dx + sine * dy
        # The slot is open at the rim, so it reaches beyond the disk boundary.
        v = -sine * dx + cosine * dy - (radius - 0.5 * depth)
        outside_u = np.abs(u) - 0.5 * width
        outside_v = np.abs(v) - 0.5 * depth - 0.5 * radius
        slot = np.minimum(np.maximum(outside_u, outside_v), 0.0) + np.hypot(
            np.maximum(outside_u, 0.0), np.maximum(outside_v, 0.0)
        )
        return np.maximum(disk, -slot)


def sample_curve(rng, kind, domain=((0.0, 1.0), (0.0, 1.0))):
    """Draw one curve of the requested kind inside ``domain``."""
    (xmin, xmax), (ymin, ymax) = domain
    width = min(xmax - xmin, ymax - ymin)
    lower = np.array([xmin, ymin])
    upper = np.array([xmax, ymax])
    interior = rng.uniform(lower + 0.2 * width, upper - 0.2 * width)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    normal = np.array([np.cos(angle), np.sin(angle)])

    if kind == "line":
        return Line(rng.uniform(lower, upper), normal)
    if kind == "circle":
        return Circle(interior, float(rng.uniform(0.1 * width, 0.3 * width)))
    if kind == "ellipse":
        semi_major = float(rng.uniform(0.12 * width, 0.3 * width))
        aspect = float(rng.uniform(1.0, 4.0))
        return Ellipse(interior, (semi_major, semi_major / aspect), angle)
    if kind == "strip":
        return Strip(
            rng.uniform(lower + 0.15 * width, upper - 0.15 * width),
            normal,
            float(np.exp(rng.uniform(np.log(0.02 * width), np.log(0.3 * width)))),
        )
    if kind == "polygon":
        count = int(rng.integers(3, 7))
        radii = rng.uniform(0.1 * width, 0.3 * width, size=count)
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=count))
        # Keep the star-shaped polygon simple by separating its vertex angles.
        angles = angles[0] + np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        vertices = interior + radii[:, None] * np.column_stack(
            [np.cos(angles), np.sin(angles)]
        )
        return Polygon(vertices)
    if kind == "slotted":
        radius = float(rng.uniform(0.12 * width, 0.3 * width))
        return SlottedDisk(
            interior,
            radius,
            angle,
            float(rng.uniform(0.15, 0.5) * radius),
            float(rng.uniform(0.8, 1.6) * radius),
        )
    raise ValueError(f"unknown curve kind {kind!r}")


def rotate_curve(curve, center, angle):
    """Rigidly rotate a curve about ``center`` by ``angle`` radians."""
    center = np.asarray(center, float)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])

    def turn(point):
        return center + rotation @ (np.asarray(point, float) - center)

    if isinstance(curve, Line):
        return Line(turn(curve.parameters["point"]), rotation @ curve.parameters["normal"])
    if isinstance(curve, Circle):
        return Circle(turn(curve.parameters["center"]), curve.parameters["radius"])
    if isinstance(curve, Ellipse):
        return Ellipse(
            turn(curve.parameters["center"]),
            curve.parameters["semi_axes"],
            curve.parameters["angle"] + angle,
        )
    if isinstance(curve, Strip):
        return Strip(
            turn(curve.parameters["point"]),
            rotation @ curve.parameters["normal"],
            curve.parameters["width"],
        )
    if isinstance(curve, Polygon):
        return Polygon(np.array([turn(vertex) for vertex in curve.parameters["vertices"]]))
    if isinstance(curve, SlottedDisk):
        return SlottedDisk(
            turn(curve.parameters["center"]),
            curve.parameters["radius"],
            curve.parameters["angle"] + angle,
            curve.parameters["slot_width"],
            curve.parameters["slot_depth"],
        )
    raise TypeError(f"cannot rotate {type(curve).__name__}")
