from typing import Literal

import numpy as np
import cvxpy as cp
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
from scipy.sparse import csc_array
from qpsolvers import solve_qp

from .ellipse import Ellipse


SCALE_MODES = Literal["best", "fast"]

if cp.MOSEK in cp.installed_solvers():
    SDP_SOLVER = cp.MOSEK
else:
    SDP_SOLVER = cp.CLARABEL



class Polytope:

    def __init__(self, A: np.ndarray, b: np.ndarray):
        if A.ndim != 2:
            raise ValueError("A must be a 2D array")
        if b.ndim != 1:
            raise ValueError("b must be a 1D array")
        if A.shape[0] != b.shape[0]:
            raise ValueError("A and b must have the same number of rows")
        
        A_norm = np.linalg.norm(A, axis=1)
        valid_rows = A_norm >= 1e-16
        self._A = np.copy(A[valid_rows]) / (A_norm[valid_rows, np.newaxis])
        self._b = np.copy(b[valid_rows]) / A_norm[valid_rows]

        self._empty = None


    def __call__(self, x: np.ndarray | cp.Variable):
        if not (isinstance(x, cp.Variable) or isinstance(x, np.ndarray)):
            raise ValueError("x must be a numpy array or a cvxpy variable")
        return self.A @ x - self.b <= 0.0
    

    @property
    def A(self):
        return self._A


    @property
    def b(self):
        return self._b
    

    @property
    def m(self):
        return self._A.shape[0]


    @property
    def n(self):
        return self._A.shape[1]
    

    @classmethod
    def from_vertices(cls, vertices: np.ndarray):
        cvx_hull = ConvexHull(vertices)
        Ab = cvx_hull.equations
        return cls(Ab[:, :-1], -Ab[:, -1])
    

    @classmethod
    def from_aabb(cls, aabb: np.ndarray):
        if aabb.ndim != 2:
            raise ValueError("aabb must be a 2D array")
        if aabb.shape[-1] != 2:
            raise ValueError("aabb must have an even number of elements")
        
        if not np.all(aabb[:,0] < aabb[:,1]):
            raise ValueError("aabb must be a list of N lower bounds then N upper bounds")

        n = aabb.shape[0]
        l_bnd, u_bnd = aabb[:,0], aabb[:,1]
        A_box = np.concatenate((np.eye(n), -np.eye(n)), axis=0)
        b_box = np.concatenate((u_bnd, -l_bnd), axis=0)
        return cls(A_box, b_box)


    def contains(self, x: np.ndarray):
        initial_shape = x.shape
        if x.shape[-1] != self.n:
            raise ValueError("x must have the same dimension as the polytope")
        contains = np.all((self.A @ x.reshape(-1, self.n, 1)).squeeze(-1) <= self.b, axis=-1)
        if len(initial_shape) == 1:
            return contains[0]
        return contains.reshape(initial_shape[:-1])


    def intersect(self, other: 'Polytope'):
        A_u = np.concatenate((self.A, other.A), axis=0)
        b_u = np.concatenate((self.b, other.b), axis=0)
        return Polytope(A_u, b_u)


    def remove_redundant(self, tol: float = 1e-8):
        if self.empty():
            return Polytope(self.A, self.b)

        A_par = np.copy(self.A)
        b_par = np.copy(self.b)

        keep_inds = []

        for i in range(self.m):
            A_par[i] = 0.0
            b_par[i] = 0.0

            keep = True
            try:
                prob = linprog(-self.A[i], A_ub=A_par, b_ub=b_par, bounds=(None, None))
                if prob.status == 0:  # success
                    if self.A[i] @ prob.x < self.b[i] - tol:
                        keep = False
            except:
                pass

            if keep:
                keep_inds.append(i)
                A_par[i] = np.copy(self.A[i])
                b_par[i] = np.copy(self.b[i])

        return Polytope(A_par[keep_inds], b_par[keep_inds])


    def empty(self):
        if self._empty is not None:
            return self._empty
        try:
            prob = linprog(np.zeros((self.n,)), A_ub=self.A, b_ub=self.b, bounds=(None, None))
            self._empty = prob.status != 0
            return self._empty
        except:
            self._empty = True
            return self._empty


    def vertices(self):
        if self.empty():
            return np.zeros((0, self.n))
        center = self.cheby_center()
        try:
            Ab = np.concatenate((self.A, -self.b[:, np.newaxis]), axis=-1)
            sp_hp = HalfspaceIntersection(Ab, center)
            return sp_hp.intersections
        except:
            return None


    def volume(self):
        if self.empty():
            return 0.0
        try:
            vol = ConvexHull(self.vertices()).volume
            return vol
        except:
            return 0.0
        

    def estimate_volume(self):
        E = self.largest_inscribed_ellipse()
        if E is None:
            return 0.0
        return E.volume()


    def scale(self, scale: float, mode: SCALE_MODES = "best"):
        if scale <= 0:
            raise ValueError("scale must be positive")
        if mode == "best":
            _, d_k = self.largest_inscribed_ellipse()
        elif mode == "fast":
            d_k = self.cheby_center()
        else:
            raise ValueError(f"Invalid mode: {mode}")
        if d_k is None:
            raise ValueError("Failed to automaticallyfind center of polytope")
        b_k = np.power(scale, 1.0 / self.n) * (self.b - (self.A @ d_k)) + (self.A @ d_k)
        return Polytope(self.A, b_k)
    

    def scale_from_point(self, point: np.ndarray, scale: float):
        b_k = scale * (self.b - (self.A @ point)) + (self.A @ point)
        return Polytope(self.A, b_k)


    def cheby_center(self):
        A_hat = np.hstack([self.A, np.linalg.norm(self.A, axis=1)[:, np.newaxis]])
        c = np.hstack([np.zeros((self.n,)), -1.0])

        bounds = tuple((None if i < self.n else 0.0, None) for i in range(self.n + 1))
        prob = linprog(c=c, A_ub=A_hat, b_ub=self.b, bounds=bounds)

        if prob.success:
            return prob.x[:-1]
        return None


    def largest_inscribed_ellipse(self):
        C = cp.Variable((self.n, self.n), symmetric=True)
        d = cp.Variable((self.n,))
        constraints = [C >> 0]
        constraints += [cp.norm(self.A @ C, axis=1) + (self.A @ d) <= self.b]
        prob = cp.Problem(cp.Maximize(cp.atoms.log_det(C)), constraints)
        prob.solve(solver=SDP_SOLVER)
        if prob.status != cp.OPTIMAL:
            return None
        return Ellipse(np.linalg.inv(C.value), d.value)


    def lowner_john_ellipse(self):
        xs = self.vertices()
        if xs is None:
            return None, None
        C = cp.Variable((self.n, self.n), symmetric=True)
        d = cp.Variable((self.n,))
        constraints = [C >> 0]
        for x in xs:
            constraints += [cp.norm((C @ x) - d) <= 1]
        prob = cp.Problem(cp.Maximize(cp.atoms.log_det(C)), constraints)
        prob.solve(solver=SDP_SOLVER)
        if prob.status != cp.OPTIMAL:
            return None
        return Ellipse(C.value, np.linalg.inv(C.value) @ d.value)


    def aabb(self):
        aabb = [0.0] * self.n * 2
        for i in range(self.n):
            a = np.zeros((self.n,))
            a[i] = 1.0
            prob = linprog(a, A_ub=self.A, b_ub=self.b, bounds=(None, None))
            if prob.success:
                aabb[i] = prob.x[i]
            else:
                raise RuntimeError(f"failed to find polytope limit along axis: {i+1}")

            a[i] = -1.0
            prob = linprog(a, A_ub=self.A, b_ub=self.b, bounds=(None, None))
            if prob.success:
                aabb[i + self.n] = prob.x[i]
            else:
                raise RuntimeError(f"failed to find polytope limit along axis: {i+1}")

        return np.array(aabb).reshape(2, self.n).T


    def translate(self, d: np.ndarray):
        return Polytope(self.A, self.b + (self.A @ d))


    def rotate(self, R: np.ndarray):
        if R.shape != (self.n, self.n):
            raise ValueError("rotation matrix must be square")
        if not np.isclose(np.linalg.det(R), 1.0) or not np.all(np.isclose(R.T, np.linalg.inv(R))):
            raise ValueError("rotation matrix must be in SO(3)")
        return Polytope(self.A @ R.T, self.b)


    def sample(self, seed=None, samples: int = 1, keep_ratio: float = 0.9):
        if keep_ratio < 1e-2:
            raise ValueError("keep_ratio must be at least 0.01, otherwise performance could become extremely poor")
        if keep_ratio > 1.0:
            raise ValueError("keep_ratio must be less than 1")
        
        if seed is None:
            seed = self.cheby_center()
        if seed is None:
            return None

        total_samples = int(samples / keep_ratio)

        res = np.zeros((total_samples, self.n))
        res[0] = seed

        def lam_min_max(A, b, x, u):
            lams = (b - A @ x) / (A @ u)
            lams_pos = lams[lams > 0]
            lams_neg = lams[lams < 0]

            lam_max = np.min(lams_pos)
            lam_min = np.max(lams_neg)

            return lam_min, lam_max

        for i in range(1, total_samples):
            x_i = res[i-1]
            u = np.random.randn(self.n)
            u /= np.linalg.norm(u)

            lam_min, lam_max = lam_min_max(self.A, self.b, x_i, u)

            lam = np.random.uniform(lam_min, lam_max)

            res[i] = x_i + lam * u

        np.random.shuffle(res)
        return res[:samples]


    def distance_to(self, other: 'Polytope'):
        D = np.hstack([np.eye(self.n), -np.eye(self.n)])

        P = csc_array(D.T @ D)
        q = np.zeros((self.n * 2,))

        A_hat = np.block(
            [
                [
                    self.A,
                    np.zeros((self.m, self.n)),
                ],
                [
                    np.zeros((other.m, self.n)),
                    other.A,
                ],
            ]
        )
        A_hat_sparse = csc_array(A_hat)

        b = np.hstack([self.b, other.b])

        X = solve_qp(P=P, q=q, G=A_hat_sparse, h=b, solver="clarabel")

        if X is not None:
            return np.linalg.norm(X[:self.n] - X[self.n:])
        return None


    def max_distance_to(self, other: 'Polytope'):
        v1 = self.vertices()
        v2 = other.vertices()
        return np.amax(cdist(v1, v2))
