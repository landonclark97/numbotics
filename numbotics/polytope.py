import numpy as np

import cvxpy as cp

from qpsolvers import solve_qp

import scipy
from scipy.spatial import ConvexHull, Delaunay, HalfspaceIntersection
from scipy.spatial.distance import cdist
from scipy.special import gamma
from scipy.optimize import linprog, minimize, lsq_linear, LinearConstraint
from scipy.sparse import csc_array, csc_matrix


class Poly():
    def __init__(self, A, b):
        A_norm = np.linalg.norm(A, axis=1)
        self._A = np.copy(A) / (A_norm[:,np.newaxis] + 1e-16)
        self._b = np.copy(b) / (A_norm + 1e-16)

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def n(self):
        return self._A.shape[1]


def poly_intersect(p1, p2):
    (A1, b1, A2, b2) = (p1.A, p1.b, p2.A, p2.b)
    A_u = np.concatenate((A1, A2), axis=0)
    b_u = np.concatenate((b1, b2), axis=0)
    return Poly(A_u, b_u)


def poly_rem_redundant(p, tol=1e-8):
    (A, b) = (p.A, p.b)
    if poly_empty(p):
        return Poly(A, b)

    n = p.n
    A_par = np.copy(A)
    b_par = np.copy(b)

    keep_inds = []

    for i in range(A.shape[0]):
        A_par[i] = 0.0
        b_par[i] = 0.0

        keep = True
        try:
            prob = linprog(-A[i], A_ub=A_par, b_ub=b_par, bounds=(None, None))
            if prob.status == 0: # success
                if (A[i] @ prob.x < b[i] - 1e-7):
                    keep = False
        except:
            pass

        if keep:
            keep_inds.append(i)
            A_par[i] = np.copy(A[i])
            b_par[i] = np.copy(b[i])

    return Poly(A_par[keep_inds], b_par[keep_inds])


def poly_empty(p):
    (A, b, n) = (p.A, p.b, p.n)
    try:
        prob = linprog(np.zeros((n,)), A_ub=A, b_ub=b, bounds=(None, None))
        return prob.status != 0
    except:
        return True


def poly_vol(p):
    if poly_empty(p):
        return 0.0
    try:
        vol = ConvexHull(poly_to_vert(p)).volume
        return vol
    except:
        return 0.0


def poly_vol_est(p):
    (A, b, n) = (p.A, p.b, p.n)
    C, d = poly_max_inscribed_ellipse(p)
    fac = (np.pi**(float(n)/2.0))/gamma((float(n)/2.0)+1)
    return fac*np.linalg.det(C)


def poly_scale(p, scale):
    (A, b) = (p.A, p.b)
    _, d_k = poly_max_inscribed_ellipse(p)
    b_k = scale*(b - (A@d_k)) + (A@d_k)
    return Poly(A, b_k)


def poly_scale_quick(p, scale):
    (A, b) = (p.A, p.b)
    center = poly_cheb_center(p)
    b_k = scale*(b - (A@center)) + (A@center)
    return Poly(A, b_k)


def poly_scale_from_center(p, center, scale):
    (A, b) = (p.A, p.b)
    b_k = scale*(b - (A@center)) + (A@center)
    return Poly(A, b_k)


def poly_cheb_center(p, return_rad=False):
    (A, b, n) = (p.A, p.b, p.n)

    A_hat = np.hstack([A, np.linalg.norm(A,axis=1)[:,np.newaxis]])
    c = np.hstack([np.zeros((n,)), -1.0])

    bounds = tuple((None if i < n else 0.0, None) for i in range(n+1))
    prob = linprog(c=c, A_ub=A_hat, b_ub=b, bounds=bounds)

    if prob.success:
        if return_rad:
            return prob.x[:-1], prob.x[-1]
        return prob.x[:-1]
    if return_rad:
        return None, None
    return None


def poly_max_inscribed_ellipse(p):
    (A, b) = (p.A, p.b)
    n = p.n
    C = cp.Variable((n,n), symmetric=True)
    d = cp.Variable((n,))
    constraints = [C >> 0]
    constraints += [cp.norm(A@C, axis=1) + (A@d) <= b]
    prob = cp.Problem(cp.Maximize(cp.atoms.log_det(C)), constraints)
    prob.solve(solver=cp.MOSEK)
    return C.value, d.value


def poly_lowner_john_ellipse(p):
    xs = poly_to_vert(p)
    n = xs.shape[1]
    C = cp.Variable((n,n), symmetric=True)
    d = cp.Variable((n,))
    constraints = [C >> 0]
    for x in xs:
        constraints += [cp.norm((C@x) - d) <= 1]
    prob = cp.Problem(cp.Maximize(cp.atoms.log_det(C)), constraints)
    prob.solve(solver=cp.MOSEK)
    return C.value, np.linalg.inv(C.value)@d.value


def vert_to_poly(v):
    cvx_hull = ConvexHull(v)
    Ab = cvx_hull.equations
    return Poly(Ab[:,:-1], -Ab[:,-1])


def poly_contains(p, x):
    (A, b) = (p.A, p.b)
    return np.all(A@x <= b)


def poly_to_vert(p):
    (A, b) = (p.A, p.b)
    if poly_empty(p):
        return np.zeros((0,A.shape[1]))
    center = poly_cheb_center(p)
    try:
        Ab = np.concatenate((A,-b[:,np.newaxis]),axis=-1)
        sp_hp = HalfspaceIntersection(Ab, center)
        return sp_hp.intersections
    except:
        return np.zeros((0,A.shape[1]))


def poly_aabb(p):
    aabb = [0.0] * p.n * 2
    for i in range(p.n):
        a = np.zeros((p.n,))
        a[i] = 1.0
        prob = linprog(a, A_ub=p.A, b_ub=p.b, bounds=(None, None))
        if prob.success:
            aabb[i] = prob.x[i]
        else:
            raise RuntimeError(f'failed to find polytope limit along axis: {i+1}')

        a[i] = -1.0
        prob = linprog(a, A_ub=p.A, b_ub=p.b, bounds=(None, None))
        if prob.success:
            aabb[i + p.n] = prob.x[i]
        else:
            raise RuntimeError(f'failed to find polytope limit along axis: {i+1}')

    return tuple(aabb)


def poly_translate(p, d):
    return Poly(p.A, p.b + (p.A@d))


def poly_rotate(p, R):
    assert (R.shape[0] == R.shape[1]) and len(R.shape) == 2, 'rotation matrix must be square'
    assert np.isclose(np.linalg.det(R), 1.0) and np.all(np.isclose(R.T, np.linalg.inv(R))), 'rotation matrix must be in SO(3)'
    return Poly(p.A @ R.T, p.b)


def aabb_to_poly(aabb):
    n = len(aabb) // 2
    l_bnd, u_bnd = aabb[:n], aabb[n:]
    A_box = np.concatenate((np.eye(n), -np.eye(n)), axis=0)
    b_box = np.concatenate((np.ones((n))*u_bnd, -np.ones((n))*l_bnd), axis=0)
    return Poly(A_box, b_box)


def poly_contains_batch(p, xs):
    Ax = np.einsum('ij,Bj->Bi', p.A, xs)
    Ax_b = Ax <= p.b
    return np.all(Ax_b, axis=1)


def poly_random_sample(p, d=None, samples=100):
    n = p.A.shape[1]
    A = p.A
    b = p.b

    if d is None:
        d = poly_cheb_center(p)

    res = np.zeros((1,n))
    res[0] = d

    def lam_min_max(A, b, x, u):
        lams = (b - A@x) / (A@u)
        lams_pos = lams[lams > 0]
        lams_neg = lams[lams < 0]

        lam_max = np.min(lams_pos)
        lam_min = np.max(lams_neg)

        return lam_min, lam_max

    for _ in range(samples-1):
        x_i = res[-1]
        u = np.random.randn(n)
        u /= np.linalg.norm(u)

        lam_min, lam_max = lam_min_max(A, b, x_i, u)

        lam = np.random.uniform(lam_min, lam_max)

        x_prime = x_i + lam * u
        res = np.concatenate((res, x_prime[np.newaxis]), axis=0)

    return res


def poly_min_dist(p1, p2):
    (A1, b1, n) = (p1.A, p1.b, p1.n)
    (A2, b2) = (p2.A, p2.b)

    D = np.hstack([np.eye(n), -np.eye(n)])

    P = csc_array(D.T@D)
    q = np.zeros((n*2,))

    A_hat = np.block([
        [
            A1,
            np.zeros((A1.shape[0], n)),
        ],
        [
            np.zeros((A2.shape[0], n)),
            A2,
        ]
    ])
    A_hat_sparse = csc_array(A_hat)

    b = np.hstack([b1, b2])

    X = solve_qp(P=P, q=q, G=A_hat_sparse, h=b, solver='clarabel')

    if X is not None:
        return np.linalg.norm(X[:n] - X[n:])
    return None


def poly_max_dist(p1, p2):

    n = p1.A.shape[1]
    v1 = poly_to_vert(p1)
    v2 = poly_to_vert(p2)

    d_n = np.amax(cdist(v1, v2))

    return d_n
