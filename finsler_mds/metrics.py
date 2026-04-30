"""Finsler metrics used in the embedding space.

The metrics in this module are uniform alpha-beta metrics with the beta
direction fixed to the last coordinate axis. The length of a vector ``u``
depends on ``r = ||u||`` and ``s = <u, e_z> / ||u||`` through
``F(u) = r * phi(s)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class AlphaBetaMetric:
    """Base class for alpha-beta metrics with fixed beta direction.

    Subclasses define ``phi(s)`` and ``dphi(s)``. This base class provides
    vector lengths, pairwise dissimilarities, and gradients with respect to the
    displacement vector ``u``.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Metric", "").lower()

    def phi(self, s):
        """Return the scalar profile ``phi(s)``."""
        raise NotImplementedError

    def dphi(self, s):
        """Return the derivative ``d phi / d s``."""
        raise NotImplementedError

    def split_rs(self, u):
        """Return ``r = ||u||`` and ``s = <u, e_z> / r`` for displacements."""
        u = np.asarray(u)
        r = np.linalg.norm(u, axis=-1)
        z = u[..., -1]
        s = np.divide(z, r, out=np.zeros_like(r, dtype=float), where=r > _EPS)
        return r, s

    def length(self, u):
        """Return the metric length of one or many displacement vectors."""
        r, s = self.split_rs(u)
        return r * self.phi(s)

    def pairwise(self, X):
        """Return the asymmetric pairwise dissimilarity matrix on rows of ``X``.

        Entry ``(i, j)`` is ``F(X[j] - X[i])``. For ``RandersMetric`` this
        matches the original ``canonical_randers_dissimilarity`` behavior.
        """
        X = np.asarray(X)
        diff = X[None, :, :] - X[:, None, :]
        return self.length(diff)

    def grad_u(self, u):
        """Return the gradient of ``F(u)`` with respect to ``u``.

        The gradient at ``u = 0`` is set to zero. This mostly affects diagonal
        self-distances. Distinct coincident points should usually be avoided by
        jittering the initialization or masking zero-length pairs.
        """
        u = np.asarray(u)
        r, s = self.split_rs(u)
        phi = self.phi(s)
        dphi = self.dphi(s)

        nonzero = r > _EPS
        coeff = phi - s * dphi
        direction = np.divide(
            u,
            r[..., None],
            out=np.zeros_like(u, dtype=float),
            where=nonzero[..., None],
        )
        grad = coeff[..., None] * direction
        grad[..., -1] += dphi
        grad = np.where(nonzero[..., None], grad, 0.0)
        return grad


@dataclass(frozen=True)
class RandersMetric(AlphaBetaMetric):
    """Randers metric ``F(u) = ||u|| + alpha * <u, e_z>``."""

    alpha: float = 0.0

    def __post_init__(self):
        if not 0 <= self.alpha < 1:
            raise ValueError("Randers alpha must satisfy 0 <= alpha < 1.")

    def phi(self, s):
        return 1 + self.alpha * s

    def dphi(self, s):
        return np.full_like(s, self.alpha, dtype=float)


@dataclass(frozen=True)
class MatsumotoMetric(AlphaBetaMetric):
    """Matsumoto metric ``F(u) = ||u||^2 / (||u|| - alpha * <u, e_z>)``.
    max_phi and forbidden_grad_norm handle the case alpha >= 1 where some directions 
    have infinite length and undefined gradient.
    However for optimisations based on geodesic graph distances, max_phi=None is recommended
    (makes the directions actually forbidden) and forbidden_grad_norm becomes unnecessary.
    If used, 4*alpha is a natural choice for forbidden_grad_norm, and a minimum for max_phi.
    """

    alpha: float = 0.0
    max_phi: float | None = None
    forbidden_grad_norm: float | None = None

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError("Matsumoto alpha must be non-negative.")
        if self.max_phi is not None and self.max_phi <= 0:
            raise ValueError("max_phi must be positive when provided.")
        if self.forbidden_grad_norm is not None and self.forbidden_grad_norm <= 0:
            raise ValueError("forbidden_grad_norm must be positive when provided.")

    def phi(self, s):
        denominator = 1 - self.alpha * s
        allowed = denominator > 0
        out = np.divide(
            1.0,
            denominator,
            out=np.full_like(denominator, np.inf, dtype=float),
            where=allowed,
        )
        if self.max_phi is not None:
            out = np.minimum(out, self.max_phi)
        return out

    def dphi(self, s):
        denominator = 1 - self.alpha * s
        allowed = denominator > 0
        out = np.divide(
            self.alpha,
            denominator**2,
            out=np.full_like(denominator, np.nan, dtype=float),
            where=allowed,
        )
        if self.max_phi is not None:
            clipped = self.phi(s) >= self.max_phi
            out = np.where(clipped, 0.0, out)
        return out

    def grad_u(self, u):
        """Return a gradient, including a surrogate inside forbidden directions.

        For ``alpha >= 1`` some directions satisfy ``1 - alpha*s <= 0`` and the
        Matsumoto length is infinite. Without ``forbidden_grad_norm``, the
        gradient is undefined and returned as ``nan`` in those directions. If
        ``forbidden_grad_norm`` is set, forbidden directions receive a bounded
        surrogate gradient pointing toward lower ``s``, so a gradient descent
        step moves them out of the forbidden cone.
        """
        grad = super().grad_u(u)
        if self.alpha == 0:
            return grad

        u = np.asarray(u)
        r, s = self.split_rs(u)
        forbidden = (r > _EPS) & (1 - self.alpha * s <= 0)
        if not np.any(forbidden):
            return grad
        if self.forbidden_grad_norm is None:
            grad[forbidden] = np.nan
            return grad

        direction = np.divide(
            u,
            r[..., None],
            out=np.zeros_like(u, dtype=float),
            where=(r > _EPS)[..., None],
        )
        grad_s = -s[..., None] * direction
        grad_s[..., -1] += 1
        grad_s = np.divide(
            grad_s,
            r[..., None],
            out=np.zeros_like(grad_s),
            where=(r > _EPS)[..., None],
        )
        norm = np.linalg.norm(grad_s, axis=-1)
        exit_grad = np.divide(
            grad_s,
            norm[..., None],
            out=np.zeros_like(grad_s),
            where=norm[..., None] > _EPS,
        )
        grad[forbidden] = self.forbidden_grad_norm * exit_grad[forbidden]
        return grad


@dataclass(frozen=True)
class ConvexifiedMatsumotoMetric(AlphaBetaMetric):
    """Matsumoto metric induced by the convexified ambient unit ball.

    For ``alpha > 1/2``, the Matsumoto unit ball is no longer convex. This
    metric uses the Minkowski functional of its convex hull. In axial notation:

    ``phi(s) = 1 / (1 - alpha*s)`` for ``s <= 1 / (2*alpha)``,
    ``phi(s) = 4*alpha*s`` for ``s > 1 / (2*alpha)``.

    The two branches match with the same first derivative at the boundary.
    For ``alpha <= 1/2`` the boundary is outside the attainable range
    ``s <= 1``, so this is identical to Matsumoto.
    """

    alpha: float = 0.0

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError("Convexified Matsumoto alpha must be non-negative.")

    def _linear_branch(self, s):
        if self.alpha == 0:
            return np.zeros_like(s, dtype=bool)
        return s > 1 / (2 * self.alpha)

    def phi(self, s):
        s = np.asarray(s)
        out = 1 / (1 - self.alpha * s)
        linear = self._linear_branch(s)
        return np.where(linear, 4 * self.alpha * s, out)

    def dphi(self, s):
        s = np.asarray(s)
        out = self.alpha / (1 - self.alpha * s) ** 2
        linear = self._linear_branch(s)
        return np.where(linear, 4 * self.alpha, out)


def _slope_from_s(s):
    s = np.asarray(s)
    denominator = np.sqrt(np.maximum(1 - s**2, 0.0))
    return np.divide(
        s,
        denominator,
        out=np.full_like(s, np.inf, dtype=float),
        where=denominator > _EPS,
    )


def _dslope_ds(s):
    s = np.asarray(s)
    denominator = np.maximum(1 - s**2, 0.0) ** 1.5
    return np.divide(
        1.0,
        denominator,
        out=np.full_like(s, np.inf, dtype=float),
        where=denominator > _EPS,
    )


@dataclass(frozen=True)
class ToblerMetric(AlphaBetaMetric):
    """Metric based on  Tobler's hiking function: ``v(g) = speed * exp(-a * |g + b|)``,
    where ``g`` is rise/run slope.
    """

    a: float = 3.5
    b: float = 0.05
    speed: float = 1.0

    def __post_init__(self):
        if self.a < 0:
            raise ValueError("Tobler a must be non-negative.")
        if self.speed <= 0:
            raise ValueError("Tobler speed must be positive.")

    def phi(self, s):
        g = _slope_from_s(s)
        return np.exp(self.a * np.abs(g + self.b)) / self.speed

    def dphi(self, s):
        g = _slope_from_s(s)
        signed = np.sign(g + self.b)
        return self.phi(s) * self.a * signed * _dslope_ds(s)


@dataclass(frozen=True)
class ConvexifiedToblerMetric(ToblerMetric):
    """Tobler metric induced by the convexified ambient unit ball.

    In the axial cross-section, the Tobler unit ball can have non-convex caps
    near large uphill/downhill slopes. Convexification replaces the part above
    the maximal reachable height by a horizontal segment ``z = z_max`` and the
    part below the minimal height by ``z = z_min``. In ``F(u)=r*phi(s)`` form,
    these caps become linear branches ``phi(s)=s/z_max`` and
    ``phi(s)=s/z_min``.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.a <= 0:
            raise ValueError("Convexified Tobler requires a > 0.")
        g_uphill = _positive_cubic_root(1 / self.a)
        g_downhill = -max(g_uphill, self.b)
        s_uphill = g_uphill / np.sqrt(1 + g_uphill**2)
        s_downhill = g_downhill / np.sqrt(1 + g_downhill**2)
        z_max = s_uphill / super().phi(np.asarray(s_uphill))
        z_min = s_downhill / super().phi(np.asarray(s_downhill))

        object.__setattr__(self, "_g_uphill", g_uphill)
        object.__setattr__(self, "_g_downhill", g_downhill)
        object.__setattr__(self, "_s_uphill", s_uphill)
        object.__setattr__(self, "_s_downhill", s_downhill)
        object.__setattr__(self, "_z_max", z_max)
        object.__setattr__(self, "_z_min", z_min)

    @property
    def uphill_optimal_slope(self):
        return self._g_uphill

    @property
    def downhill_optimal_slope(self):
        return self._g_downhill

    @property
    def s_uphill(self):
        return self._s_uphill

    @property
    def s_downhill(self):
        return self._s_downhill

    @property
    def z_max(self):
        return self._z_max

    @property
    def z_min(self):
        return self._z_min

    def phi(self, s):
        s = np.asarray(s)
        base = super().phi(s)
        uphill = s > self.s_uphill
        downhill = s < self.s_downhill
        out = np.where(uphill, s / self.z_max, base)
        return np.where(downhill, s / self.z_min, out)

    def dphi(self, s):
        s = np.asarray(s)
        base = super().dphi(s)
        uphill = s > self.s_uphill
        downhill = s < self.s_downhill
        out = np.where(uphill, 1 / self.z_max, base)
        return np.where(downhill, 1 / self.z_min, out)


def _positive_cubic_root(value):
    roots = np.roots([1.0, 0.0, 1.0, -value])
    real_roots = roots[np.isclose(roots.imag, 0.0)].real
    positive_roots = real_roots[real_roots > 0]
    if positive_roots.size != 1:
        raise RuntimeError("Could not solve g + g^3 = value.")
    return float(positive_roots[0])


@dataclass(frozen=True)
class MinettiMetric(AlphaBetaMetric):
    """Metric based on Minetti et al.'s metabolic cost fit as a function of slope.

    The default polynomial is the common Minetti et al. metabolic cost fit
    ``155.4*g^5 - 30.4*g^4 - 43.3*g^3 + 46.3*g^2 + 19.5*g + 3.6``.
    Dividing by ``power`` converts it to a relative travel-time factor; the
    default ``power=3.6`` normalizes flat terrain to one.
    """

    g_max: float = 0.45
    a5: float = 155.4
    a4: float = -30.4
    a3: float = -43.3
    a2: float = 46.3
    a1: float = 19.5
    a0: float = 3.6
    power: float = 3.6

    def __post_init__(self):
        if self.g_max <= 0:
            raise ValueError("Minetti g_max must be positive.")
        if self.power <= 0:
            raise ValueError("Minetti power must be positive.")

    def _cost(self, g):
        return (
            self.a5 * g**5
            + self.a4 * g**4
            + self.a3 * g**3
            + self.a2 * g**2
            + self.a1 * g
            + self.a0
        )

    def _dcost_dg(self, g):
        return (
            5 * self.a5 * g**4
            + 4 * self.a4 * g**3
            + 3 * self.a3 * g**2
            + 2 * self.a2 * g
            + self.a1
        )

    def phi(self, s):
        g = _slope_from_s(s)
        allowed = np.abs(g) <= self.g_max
        out = self._cost(g) / self.power
        return np.where(allowed, out, np.inf)

    def dphi(self, s):
        g = _slope_from_s(s)
        allowed = np.abs(g) <= self.g_max
        out = self._dcost_dg(g) * _dslope_ds(s) / self.power
        return np.where(allowed, out, np.nan)


def canonical_randers_dissimilarity(alpha):
    """Compatibility wrapper for the original Randers pairwise function."""
    metric = RandersMetric(alpha=alpha)
    return metric.pairwise


def canonical_randers_metric(alpha):
    """Compatibility wrapper for the original two-point Randers metric."""
    metric = RandersMetric(alpha=alpha)

    def randers_metric(X, Y):
        return metric.length(np.asarray(Y) - np.asarray(X))

    return randers_metric


def get_metric(metric, **params):
    """Return a metric instance from a name or pass through an existing metric."""
    if isinstance(metric, AlphaBetaMetric):
        return metric
    if metric in ("randers", RandersMetric):
        return RandersMetric(**params)
    if metric in ("matsumoto", MatsumotoMetric):
        return MatsumotoMetric(**params)
    if metric in (
        "convexified_matsumoto",
        "matsumoto_convexified",
        ConvexifiedMatsumotoMetric,
    ):
        return ConvexifiedMatsumotoMetric(**params)
    if metric in ("tobler", ToblerMetric):
        return ToblerMetric(**params)
    if metric in (
        "convexified_tobler",
        "tobler_convexified",
        ConvexifiedToblerMetric,
    ):
        return ConvexifiedToblerMetric(**params)
    if metric in ("minetti", "minetti_cost", MinettiMetric):
        return MinettiMetric(**params)
    raise ValueError(f"Unknown Finsler metric: {metric!r}")


__all__ = [
    "AlphaBetaMetric",
    "RandersMetric",
    "MatsumotoMetric",
    "ConvexifiedMatsumotoMetric",
    "ToblerMetric",
    "ConvexifiedToblerMetric",
    "MinettiMetric",
    "canonical_randers_metric",
    "canonical_randers_dissimilarity",
    "get_metric",
]
