from typing import Tuple, Literal
from dataclasses import dataclass, field
from itertools import repeat

import numpy as np
import cvxpy as cp
from scipy.optimize import (
    minimize,
    LinearConstraint,
    NonlinearConstraint,
)

from numbotics.math.geometry import Polytope, Ellipse, ConvexSet
from numbotics.robots import Robot
from numbotics.physics import Link, PhysicsObject
from numbotics.utils import logger, cpu_count, ResourceThreadPool



@dataclass(frozen=True)
class IrisParams:
    configuration_margin: float = 1e-1
    admissible_collisions: float = 5e-3
    max_uncertainty: float = 5e-3
    max_iters: int = 100
    num_particles: int = 1000
    num_bisections: int = 15
    termination_tolerance: float = 1e-3
    collision_tolerance: float = 1e-6
    tau: float = 0.5
    hyperplane_method: Literal['zoh', 'np2'] = 'zoh'
    threads: int = cpu_count()
    solver_options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.tau < 0.0 or self.tau > 1.0:
            raise ValueError("tau must be between 0 and 1")
        if self.configuration_margin < 0.0:
            raise ValueError("configuration_margin must be greater than 0")
        if self.admissible_collisions < 0.0 or self.admissible_collisions > 1.0:
            raise ValueError("admissible_collisions must be between 0 and 1")
        if self.max_uncertainty < 0.0 or self.max_uncertainty > 1.0:
            raise ValueError("max_uncertainty must be between 0 and 1")
        if self.max_iters < 1:
            raise ValueError("max_iters must be greater than 0")
        if self.num_particles < 1:
            raise ValueError("num_particles must be greater than 0")
        if self.num_bisections < 1:
            raise ValueError("num_bisections must be greater than 0")
        if self.termination_tolerance <= 0.0:
            raise ValueError("termination_tolerance must be greater than 0")
        if self.tau < 1e-1:
            logger.warning("Iris setting for tau is less than 0.1, this may lead to extremely large runtimes")
        if self.hyperplane_method not in ['zoh', 'np2']:
            raise ValueError(f"hyperplane_method must be either 'zoh' or 'np2', received {self.hyperplane_method}")
        if self.threads < 1:
            raise ValueError("threads must be greater than 0")
        if self.threads > cpu_count():
            logger.warning(f"threads is greater than the number of available cores...")
        if not isinstance(self.solver_options, dict):
            raise ValueError("solver_options must be a dictionary")



# IRIS-NP: https://arxiv.org/pdf/2303.14737
# IRIS-NP-fast: https://arxiv.org/pdf/2410.12649
class IrisSolver:

    def __init__(
            self, 
            subject: Robot,
            params: IrisParams = IrisParams()
        ):
        if not isinstance(subject, Robot):
            raise ValueError("subject must be a Robot")
        self._subject = subject
        self._params = params


    def new_separating_hyperplane(self, q: np.ndarray, E: Ellipse):
        a = ((E.C.T @ E.C) @ (q - E.d)) / np.linalg.norm((E.C.T @ E.C) @ (q - E.d))
        b = (a @ q) - self._params.configuration_margin
        return Polytope(np.atleast_2d(a), np.atleast_1d(b))


    def counter_ex_search_nlp(
        self,
        q_init: np.ndarray,
        body_pair: Tuple[Link | PhysicsObject, Link | PhysicsObject],
        P: Polytope,
        E: Ellipse,
    ):

        def cost(q):
            diff = q - E.d
            return diff @ ((E.C.T @ E.C) @ diff)
        
        def jac(q):
            return self._subject.jacobian_proximity(q, link=body_pair[0], obj=body_pair[1])

        dist_const = NonlinearConstraint(
            lambda x: self._subject.distance_to(x, link=body_pair[0], obj=body_pair[1])[0].distance,
            -np.inf,
            -self._params.collision_tolerance,
        )

        poly_const = LinearConstraint(P.A, -np.inf, P.b - self._params.collision_tolerance)

        default_solver_options = { 'maxiter' : 20 }
        res = minimize(
            cost, 
            x0=q_init, 
            jac=jac,
            constraints=(dist_const, poly_const),
            method='slsqp',
            options=default_solver_options.update(self._params.solver_options)
        )

        if res.success:
            return True, res.x
        return False, res.x


    def counter_ex_search_bisection(self, subject: Robot, args):
        q, E = args
        interval = [E.d, q]
        for _ in range(self._params.num_bisections):
            midpoint = (interval[0] + interval[1]) / 2.0
            if min([p.distance for p in subject.collisions(midpoint)]) < self._params.collision_tolerance:
                interval[1] = midpoint
            else:
                interval[0] = midpoint

        return interval[1]
    

    def counter_ex_search_greedy(self, S_col: np.ndarray, P: Polytope, E: Ellipse):

        metric = np.linalg.norm((E.C @ (S_col - E.d[None])[..., None]).squeeze(axis=-1), axis=1)
        idx = np.argsort(metric)
        S_col = S_col[idx]

        for q in S_col:
            if not P.contains(q):
                continue

            prox = self._subject.closest_to(q)
            _, q_cs = self.counter_ex_search_nlp(q, (prox.subject, prox.target), P, E)
            if P.contains(q_cs):
                P = P.intersect(self.new_separating_hyperplane(q_cs, E))

        return P
    

    def counter_ex_search_convex(self, E: Ellipse, P: Polytope, O: ConvexSet):
        x = cp.Variable((P.n,))
        cons = [P(x), O(x)]
        obj = cp.Minimize(cp.quad_form(x - E.d, E.C.T @ E.C))
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.MOSEK)
        return x.value


    def seperating_hyperplanes(
        self,
        P_base: Polytope,
        E: Ellipse,
        outer_iters: int,
        thread_pool: ResourceThreadPool,
    ):
        
        P = P_base
        i = outer_iters

        for k in range(self._params.max_iters):
            delta_i_k = (36.0 * self._params.max_uncertainty) / ((np.pi ** 4) * ((i+1) ** 2) * ((k+1) ** 2))
            unadaptive_samples = np.ceil(
                2.0 * np.log(1.0 / delta_i_k) / 
                (self._params.admissible_collisions * (self._params.tau ** 2))
            ).astype(np.int64)

            M = max(unadaptive_samples, self._params.num_particles)
            print(f'sampling {M} points')
            points = P.sample(samples=M, keep_ratio=0.5)

            def is_colliding_wrapper(subject, args):
                q = args
                return subject.in_collision(q, self._params.collision_tolerance)
            self.__is_colliding_wrapper = is_colliding_wrapper
            
            results = list(thread_pool.map(self.__is_colliding_wrapper, points))
            S_col = points[results]

            print(f'{float(len(S_col)) / float(M)} percent of points in collision')

            if float(len(S_col)) / float(M) < (1.0 - self._params.tau) * self._params.admissible_collisions:
                break

            if self._params.hyperplane_method == 'zoh':
                results = list(thread_pool.map(self.counter_ex_search_bisection, zip(S_col, repeat(E))))
                S_star_col = np.array(results)
                
                metric = np.linalg.norm((E.C @ (S_star_col - E.d[None])[..., None]).squeeze(axis=-1), axis=1)
                idx = np.argsort(metric)
                S_star_col = S_star_col[idx]

                for q in S_star_col:
                    if P.contains(q):
                        P = P.intersect(self.new_separating_hyperplane(q, E))
                print(f'hyperplanes: {P.m}')
            
            elif self._params.hyperplane_method == 'np2':
                P = self.counter_ex_search_greedy(S_col, P, E)
                print(f'hyperplanes: {P.m}')
            
        else:
            raise StopIteration(f"Iris solver exceeded max iterations")
        
        return P.remove_redundant()


    def solve(
        self,
        seed: np.ndarray,
        P_base: Polytope,
    ):

        if self._subject.in_collision(seed, self._params.collision_tolerance):
            raise ValueError("initial configuration in collision")
        
        if len(self._subject.collision_pairs()) == 0:
            logger.info("no collision pairs found, skipping IRIS computation")
            return P_base

        with self._subject.pool(
                poolsize=self._params.threads
            ) as subject_pool, ResourceThreadPool(
                poolsize=self._params.threads, 
                per_thread_resources=subject_pool,
            ) as thread_pool:

            E = Ellipse(np.eye(seed.shape[0]), seed)
            prev_E_vol = 0.0

            for main_iters in range(self._params.max_iters):

                if (E.volume() - prev_E_vol) / E.volume() < self._params.termination_tolerance:
                    break

                P = self.seperating_hyperplanes(P_base, E, main_iters, thread_pool)
                prev_E_vol = E.volume()
                E = P.largest_inscribed_ellipse()
                print(f'current largest inscribed ellipse volume: {E.volume()}')

        print(f"final largest inscribed ellipse volume: {E.volume()}")

        return P

