import yaml
import contextlib
from weakref import WeakSet

import numpy as np
import networkx as nx

from numbotics.physics import GraphChain, Constraint, PhysicsObject, Chain, Link
from numbotics.math import rot_diff, trans_mat
from numbotics.utils import Shape, logger
from .robot import Robot
from .helpers import nb_compute_transformation, nb_compute_jacobian


class Arm(Robot):

    def __init__(self, chain: GraphChain):
        super().__init__(chain)

        G_sorted = nx.topological_sort(self._chain._G)
        root_node = next(iter(G_sorted))

        self._link_joint_sequence = {root_node: tuple()}
        self._links_from_nodes = {root_node: self._chain._G.nodes[root_node]["link"]}
        for node in G_sorted:
            if node == root_node:
                continue
            path = nx.shortest_path(self._chain._G, root_node, node)
            fixed_joint_sequence = []

            axes = []
            offsets = []
            types = []
            idxs = []

            for (u, v) in zip(path[:-1], path[1:]):
                joint = self._chain._G.edges[(u, v)]["joint"]
                if joint.type == Constraint.FIXED:
                    fixed_joint_sequence.append(joint)
                else:
                    axes.append(joint.axis)
                    types.append(joint.type.value)
                    idxs.append(self._chain._Chain__joint_to_index.get(joint))
                    if fixed_joint_sequence:
                        T = np.eye(4)
                        for fixed_joint in fixed_joint_sequence:
                            T @= fixed_joint.offset
                        fixed_joint_sequence = []
                    else:
                        T = np.eye(4)
                    T @= joint.offset
                    offsets.append(T)

            if fixed_joint_sequence:
                T = np.eye(4)
                for fixed_joint in fixed_joint_sequence:
                    T @= fixed_joint.offset
                offsets.append(T)

            # This is nasty...
            self._link_joint_sequence[node] = (np.array(offsets), np.array(axes), np.array(types, dtype=np.int64), np.array(idxs, dtype=np.int64))
            self._links_from_nodes[node] = self._chain._G.nodes[node]["link"]

        self._additional_self_collision_pairs = set()
        self._void_self_collision_pairs = set()

        self._additional_collision_pairs = set()
        self._void_collision_pairs = set()

        # Populate self colliders cache
        self.self_collision_pairs()


    @property
    def base_pose(self):
        if self._chain._static_base:
            if hasattr(self, '_base_pose'):
                return self._base_pose
        self._base_pose = self._chain.base_pose
        return self._base_pose
    

    @property
    def base_velocity(self):
        return self._chain.base_velocity
    

    @property
    def joint_limits(self):
        return self._chain.joint_limits


    @property
    def dof(self):
        return self._chain.dof
    

    @property
    def configuration(self):
        return self._chain.configuration
    

    @configuration.setter
    def configuration(self, q: np.ndarray):
        self._chain.configuration = q


    @property
    def velocity(self):
        return self._chain.velocity
    

    @velocity.setter
    def velocity(self, qdot: np.ndarray):
        self._chain.velocity = qdot


    @property
    def effort(self):
        return self._chain.effort
    
    
    @effort.setter
    def effort(self, tau: np.ndarray):
        self._chain.effort = tau


    @contextlib.contextmanager
    def stateless(self):
        initial_configuration = self._chain.configuration
        initial_velocity = self._chain.velocity
        initial_effort = self._chain.effort

        initial_base_pose = self._chain.base_pose
        initial_base_velocity = self._chain.base_velocity

        try:
            yield

        finally:
            self._chain.configuration = initial_configuration
            self._chain.velocity = initial_velocity
            self._chain.effort = initial_effort

            self._chain.base_pose = initial_base_pose
            self._chain.base_velocity = initial_base_velocity


    @contextlib.contextmanager
    def pool(self, poolsize: int = 1):
        arm_refs = []
        arms = WeakSet()
        with self._chain.world.pool(poolsize) as worlds:
            for world in worlds:
                arm = Arm(world.get_object(f'{world.name}:{self._chain._name}'))
                arm_refs.append(arm)
                arms.add(arm_refs[-1])

                for self_collision_pair in self._additional_self_collision_pairs:
                    link_a, link_b = self_collision_pair
                    link_a_name = f'{world.name}:{":".join(link_a.name.split(":")[1:])}'
                    link_b_name = f'{world.name}:{":".join(link_b.name.split(":")[1:])}'
                    arm_refs[-1].add_collision_pair(link_a_name, link_b_name)
                
                for collision_pair in self._additional_collision_pairs:
                    link_a, link_b = collision_pair
                    link_a_name = f'{world.name}:{":".join(link_a.name.split(":")[1:])}'
                    obj_b_name = f'{world.name}:{":".join(link_b.name.split(":")[1:])}'
                    arm_refs[-1].add_collision_pair(link_a_name, obj_b_name)

                for self_collision_pair in self._void_self_collision_pairs:
                    link_a, link_b = self_collision_pair
                    link_a_name = f'{world.name}:{":".join(link_a.name.split(":")[1:])}'
                    link_b_name = f'{world.name}:{":".join(link_b.name.split(":")[1:])}'
                    arm_refs[-1].remove_collision_pair(link_a_name, link_b_name)
                    
                for collision_pair in self._void_collision_pairs:
                    link_a, link_b = collision_pair
                    link_a_name = f'{world.name}:{":".join(link_a.name.split(":")[1:])}'
                    obj_b_name = f'{world.name}:{":".join(link_b.name.split(":")[1:])}'
                    arm_refs[-1].remove_collision_pair(link_a_name, obj_b_name)

            try:
                yield arms
            finally:
                del arms
                del arm_refs


    def self_collision_pairs(self):
        if not hasattr(self, '_self_collision_pairs'):
            self._self_collision_pairs = set()

            for link_a in self._chain._links:
                if link_a._collision_shape.shape == Shape.EMPTY:
                    continue
                for link_b in self._chain._links:
                    if link_a == link_b:
                        continue
                    if link_b._collision_shape.shape == Shape.EMPTY:
                        continue
                    try:
                        G_undirected = self._chain._G.to_undirected()
                        neighbor_test = nx.shortest_path(G_undirected, link_a.name, link_b.name)
                        nonfixed_count = 0
                        for u, v in zip(neighbor_test[:-1], neighbor_test[1:]):
                            if G_undirected.edges[(u, v)]["joint"].type != Constraint.FIXED:
                                nonfixed_count += 1
                        # by default we only look at collisions between links that are not
                        # a part of either the same "weld" or neighboring "welds"
                        if nonfixed_count < 2:
                            continue
                    except nx.NetworkXException:
                        pass

                    if not G_undirected.has_edge(link_a._name, link_b._name) and (link_b, link_a) not in self._self_collision_pairs:
                        self._self_collision_pairs.add((link_a, link_b))

        return self._self_collision_pairs.union(
                self._additional_self_collision_pairs
            ).difference(
                self._void_self_collision_pairs
            )
    

    def collision_pairs(self):
        collision_pairs = set()
        objects = list(self._chain.world._static_objects.values()) + list(self._chain.world._dynamic_objects.values())
        
        for link in self._chain._links:
            if link._collision_shape.shape == Shape.EMPTY:
                continue
            for obj in objects:
                if obj == self._chain:
                    continue
                elif isinstance(obj, PhysicsObject):
                    if obj._collision_shape.shape == Shape.EMPTY:
                        continue
                    collision_pairs.add((link, obj))
                elif isinstance(obj, Chain):
                    for obj_link in obj._links:
                        collision_pairs.add((link, obj_link))

        return collision_pairs.union(
                self.self_collision_pairs()
            ).union(
                self._additional_collision_pairs
            ).difference(
                self._void_collision_pairs
            )
    
    
    def add_collision_pair(self, link_a: Link | PhysicsObject | str, link_b: Link | PhysicsObject | str):
       
        # allow string lookup. NOTE: this requires looking through all objects in the world,
        # this may or may not be a bad idea.
        if isinstance(link_a, str):
            _link_a = self._links_from_nodes.get(link_a)
            if _link_a is None:
                _link_a = self._chain.world.get_object(link_a)
                if _link_a is None:
                    raise ValueError(f"Object name: {link_a} must be a valid object in the world")
            link_a = _link_a

        if isinstance(link_b, str):
            _link_b = self._links_from_nodes.get(link_b)
            if _link_b is None:
                _link_b = self._chain.world.get_object(link_b)
                if _link_b is None:
                    raise ValueError(f"Object name: {link_b} must be a valid object in the world")
            link_b = _link_b

        link_a_in_chain = link_a in self._chain._links
        link_b_in_chain = link_b in self._chain._links

        if not link_a_in_chain and not link_b_in_chain:
            logger.warning(f"Did not add collision pair between {link_a.name} and {link_b.name} because neither is in the chain")
            return
        elif not link_a_in_chain and link_b_in_chain:
            link_a, link_b = link_b, link_a
            link_a_in_chain, link_b_in_chain = link_b_in_chain, link_a_in_chain

        if link_b_in_chain:
            if (link_a, link_b) in self.self_collision_pairs() or (link_b, link_a) in self.self_collision_pairs():
                logger.warning(f"Did not add collision pair between {link_a.name} and {link_b.name} because it is already a self collision pair")
            else:
                if (link_a, link_b) in self._additional_self_collision_pairs or (link_b, link_a) in self._additional_self_collision_pairs:
                    logger.warning(f"Did not add collision pair between {link_a.name} and {link_b.name} because it is already a self collision pair")
                else:
                    self._additional_collision_pairs.add((link_a, link_b))
            if (link_a, link_b) in self._void_self_collision_pairs:
                self._void_self_collision_pairs.remove((link_a, link_b))
            elif (link_b, link_a) in self._void_self_collision_pairs:
                self._void_self_collision_pairs.remove((link_b, link_a))
        
        else:
            if (link_a, link_b) in self.collision_pairs() or (link_b, link_a) in self.collision_pairs():
                logger.warning(f"Did not add collision pair between {link_a.name} and {link_b.name} because it is already a collision pair")
            else:
                if (link_a, link_b) in self._additional_collision_pairs or (link_b, link_a) in self._additional_collision_pairs:
                    logger.warning(f"Did not add collision pair between {link_a.name} and {link_b.name} because it is already a collision pair")
                else:
                    self._additional_collision_pairs.add((link_a, link_b))
            if (link_a, link_b) in self._void_collision_pairs:
                self._void_collision_pairs.remove((link_a, link_b))
            elif (link_b, link_a) in self._void_collision_pairs:
                self._void_collision_pairs.remove((link_b, link_a))


    def remove_collision_pair(self, link_a: Link | PhysicsObject | str, link_b: Link | PhysicsObject | str):

        # allow string lookup. NOTE: this requires looking through all objects in the world,
        # this may or may not be a bad idea.
        if isinstance(link_a, str):
            _link_a = self._links_from_nodes.get(link_a)
            if _link_a is None:
                _link_a = self._chain.world.get_object(link_a)
                if _link_a is None:
                    raise ValueError(f"Object name: {link_a} must be a valid object in the world")
            link_a = _link_a

        if isinstance(link_b, str):
            _link_b = self._links_from_nodes.get(link_b)
            if _link_b is None:
                _link_b = self._chain.world.get_object(link_b)
                if _link_b is None:
                    raise ValueError(f"Object name: {link_b} must be a valid object in the world")
            link_b = _link_b

        link_a_in_chain = link_a in self._chain._links
        link_b_in_chain = link_b in self._chain._links

        if not link_a_in_chain and not link_b_in_chain:
            logger.warning(f"Did not remove collision pair between {link_a.name} and {link_b.name} because neither is in the chain")
            return
        elif not link_a_in_chain and link_b_in_chain:
            link_a, link_b = link_b, link_a
            link_a_in_chain, link_b_in_chain = link_b_in_chain, link_a_in_chain

        if link_b_in_chain:
            if (link_a, link_b) in self._additional_self_collision_pairs:
                self._additional_self_collision_pairs.remove((link_a, link_b))
            elif (link_b, link_a) in self._additional_self_collision_pairs:
                self._additional_self_collision_pairs.remove((link_b, link_a))

            if (link_a, link_b) in self._void_self_collision_pairs or (link_b, link_a) in self._void_self_collision_pairs:
                logger.warning(f"Did not remove collision pair between {link_a.name} and {link_b.name} because it is has already been removed")
            else:
                if (link_a, link_b) in self.self_collision_pairs():
                    self._void_self_collision_pairs.add((link_a, link_b))
                elif (link_b, link_a) in self.self_collision_pairs():
                    self._void_self_collision_pairs.add((link_b, link_a))
        
        else:
            if (link_a, link_b) in self._additional_collision_pairs:
                self._additional_collision_pairs.remove((link_a, link_b))
            elif (link_b, link_a) in self._additional_collision_pairs:
                self._additional_collision_pairs.remove((link_b, link_a))

            if (link_a, link_b) in self._void_collision_pairs or (link_b, link_a) in self._void_collision_pairs:
                logger.warning(f"Did not remove collision pair between {link_a.name} and {link_b.name} because it is has already been removed")
            else:
                if (link_a, link_b) in self.collision_pairs():
                    self._void_collision_pairs.add((link_a, link_b))
                elif (link_b, link_a) in self.collision_pairs():
                    self._void_collision_pairs.add((link_b, link_a))

                
    def forward_kinematics(self, q: np.ndarray, frame: str, use_com: bool = False, local_pose: np.ndarray | None = None):

        sequence = self._link_joint_sequence.get(frame)
        if sequence is None:
            raise ValueError(f"Frame {frame} not found in chain")
        
        if q.shape[-1] != self.dof:
            raise ValueError(f"q must have {self.dof} elements")
    
        initial_q_shape = q.shape
        if q.ndim == 1:
            q = q.reshape(1, self.dof)
        else:
            q = q.reshape(-1, self.dof)
        
        if local_pose is not None:
            if local_pose.shape[-2:] != (4, 4):
                raise ValueError(f"local_pose must be a 4x4 matrix")
            if local_pose.ndim == 2:
                local_pose = np.tile(local_pose[None], (q.shape[0], 1, 1))
            else:
                if local_pose.shape[:-2] != initial_q_shape[:-1]:
                    raise ValueError(f"local_pose must have the same batch dimensions as q")
                local_pose = local_pose.reshape(-1, 4, 4)

        T = np.tile(self.base_pose, (q.shape[0], 1, 1))
        offsets, axes, joint_types, joint_idxs = sequence
        if len(offsets) > 1:
            T = nb_compute_transformation(T, offsets[:-1], axes, joint_types, joint_idxs, q)
        T @= offsets[-1]

        if use_com:
            link = self._links_from_nodes[frame]
            T_inertial = link._offset
            T @= T_inertial

        if local_pose is not None:
            T @= local_pose

        if len(initial_q_shape) == 1:
            return T[0]
        return T.reshape(*initial_q_shape[:-1], 4, 4)


    def jacobian(self, q: np.ndarray, frame: str, use_com: bool = False, local_pose: np.ndarray | None = None, global_pose: np.ndarray | None = False):

        sequence = self._link_joint_sequence.get(frame)
        if sequence is None:
            raise ValueError(f"Frame {frame} not found in chain")
        
        if q.shape[-1] != self.dof:
            raise ValueError(f"q must have {self.dof} elements")
                
        initial_q_shape = q.shape
        if q.ndim == 1:
            q = q.reshape(1, self.dof)
        else:
            q = q.reshape(-1, self.dof)

        if local_pose is not None:
            if local_pose.shape[-2:] != (4, 4):
                raise ValueError(f"local_pose must be a 4x4 matrix")
            if local_pose.ndim == 2:
                local_pose = np.tile(local_pose[None], (q.shape[0], 1, 1))
            else:
                if local_pose.shape[:-2] != initial_q_shape[:-1]:
                    raise ValueError(f"local_pose must have the same batch dimensions as q")
                local_pose = local_pose.reshape(-1, 4, 4)
        
        if global_pose is not None:
            if local_pose is not None:
                raise ValueError("local_pose and global_pose cannot both be provided")
            if global_pose.shape[-2:] != (4, 4):
                raise ValueError(f"global_pose must be a 4x4 matrix")
            if global_pose.ndim == 2:
                global_pose = np.tile(global_pose[None], (q.shape[0], 1, 1))
            else:
                if global_pose.shape[:-2] != initial_q_shape[:-1]:
                    raise ValueError(f"global_pose must have the same batch dimensions as q")
                global_pose = global_pose.reshape(-1, 4, 4)
        
        offsets, axes, joint_types, joint_idxs = sequence
        T_mats = np.zeros((q.shape[0], offsets.shape[0], 4, 4))
        T = np.tile(self.base_pose, (q.shape[0], 1, 1))
        com_offset = self._links_from_nodes[frame]._offset

        J = np.zeros((q.shape[0], 6, self.dof))
        if len(offsets) > 1:
            J = nb_compute_jacobian(T, T_mats, com_offset, offsets, axes, joint_types, joint_idxs, q, local_pose, global_pose, use_com)

        if len(initial_q_shape) == 1:
            return J[0]
        return J.reshape(*initial_q_shape[:-1], 6, self.dof)


    def inverse_kinematics(
        self,
        pose: np.ndarray,
        q0: np.ndarray,
        frame: str,
        use_com: bool = False,
        use_limits: bool = False,
        tol: float = 1e-6,
        max_iter: int = 100,
        max_failures: int = 15,
    ):
        
        sequence = self._link_joint_sequence.get(frame)
        if sequence is None:
            raise ValueError(f"Frame {frame} not found in chain")
        
        if q0.shape[-1] != self.dof:
            raise ValueError(f"q0 must have {self.dof} elements")
        
        if max_iter < 1:
            raise ValueError("max_iter must be greater than 0")
        
        lower_limits = self.joint_limits[:, 0]
        upper_limits = self.joint_limits[:, 1]
        
        batch_dims = None
        if pose.ndim > 2:
            batch_dims = pose.shape[:-2]
            pose = pose.reshape(-1, 4, 4)

        if q0.ndim > 1:
            if batch_dims:
                if q0.shape[:-1] != batch_dims:
                    raise ValueError("pose and q0 must have the same batch dimensions")
            else:
                batch_dims = q0.shape[:-1]
                pose = np.tile(pose[None], (q0.shape[0], 1, 1))

        if batch_dims is None:
            pose = pose.reshape(-1, 4, 4)
            q0 = q0.reshape(-1, self.dof)

        q = q0.copy()
        ee_pose = self.forward_kinematics(q, frame, use_com)

        diff = np.zeros((q.shape[0], 6))
        diff[:, :3] = pose[:, :3, 3] - ee_pose[:, :3, 3]
        diff[:, 3:] = rot_diff(ee_pose[:, :3, :3], pose[:, :3, :3])
        diff_norm = np.linalg.norm(diff, axis=-1)
        B_I = np.tile(np.eye(6)[None], (q.shape[0], 1, 1))

        lambdas = np.ones((q.shape[0],)) * 1e-1
        failures = np.zeros((q.shape[0],), dtype=np.int64)
        for _ in range(max_iter):
            running = np.where((diff_norm > tol) & (failures < max_failures))
            if len(running[0]) == 0:
                break
            
            J = self.jacobian(q[running], frame, use_com)
            J_T = np.swapaxes(J, -2, -1)
            
            q[running] = q[running] + (
                J_T @ (
                    np.linalg.solve((J @ J_T) + (lambdas[running][...,None,None] * B_I[running]), diff[running][...,None])
                )
            ).squeeze(-1)
            if use_limits:
                q[running] = np.clip(q[running], lower_limits, upper_limits)
            
            ee_pose = self.forward_kinematics(q[running], frame, use_com)
            diff[running, :3] = pose[running, :3, 3] - ee_pose[:, :3, 3]
            diff[running, 3:] = rot_diff(ee_pose[:, :3, :3], pose[running, :3, :3])

            prev_diff_norm = np.copy(diff_norm[running])
            diff_norm = np.linalg.norm(diff, axis=-1)

            relative_diffs = diff_norm[running] > prev_diff_norm
            delta_lambda = np.where(relative_diffs, 1.2, 0.5)
            lambdas[running] *= delta_lambda

            failures[running[0][relative_diffs]] += 1
            failures[running[0][~relative_diffs]] = 0

            if np.all(diff_norm < tol):
                break

        if batch_dims is None:
            return diff_norm < tol, q[0]
        return diff_norm < tol, q.reshape(*batch_dims, self.dof)


    def collisions(self, q: np.ndarray):
        if q.shape != (self.dof,):
            raise ValueError(f"q must be a 1D array with {self.dof} elements")
        
        with self.stateless():
            self._chain.configuration = q
            proximities = []
            self_collision_pairs = self.self_collision_pairs()
            collision_pairs = self.collision_pairs()
            for obj in list(self._chain.world._static_objects.values()) + list(self._chain.world._dynamic_objects.values()):
                if not isinstance(obj, Chain):
                    if obj._collision_shape.shape == Shape.EMPTY:
                        continue
                obj_proximities = self._chain.distance_to(obj)
                if obj == self._chain:
                    obj_proximities = [
                        p for p in obj_proximities
                        if (p.subject, p.target) in self_collision_pairs or (p.target, p.subject) in self_collision_pairs
                    ]
                else:
                    obj_proximities = [
                        p for p in obj_proximities
                        if (p.subject, p.target) in collision_pairs or (p.target, p.subject) in collision_pairs
                    ]
                proximities.extend(obj_proximities)
            return proximities

            
    def self_collisions(self, q: np.ndarray):
        if q.shape != (self.dof,):
            raise ValueError(f"q must be a 1D array with {self.dof} elements")
        
        self_collision_pairs = self.self_collision_pairs()
        with self.stateless():
            self._chain.configuration = q

            proximities = self._chain.distance_to(self._chain)
            proximities = [
                p for p in proximities
                if (p.subject, p.target) in self_collision_pairs or (p.target, p.subject) in self_collision_pairs
            ]
            return proximities
        

    def closest_to(self, q: np.ndarray):
        return min([p for p in self.collisions(q)], key=lambda x: x.distance)


    def in_collision(self, q: np.ndarray, threshold: float = 0.0):
        return self.closest_to(q).distance < threshold


    def distance_to(self, q: np.ndarray, obj: PhysicsObject | Chain | Link, link : Link | None = None):
        with self.stateless():
            self._chain.configuration = q
            collision_pairs = self.collision_pairs()
            if link is None:
                return [p for p in self._chain.distance_to(obj) if (p.subject, p.target) in collision_pairs or (p.target, p.subject) in collision_pairs]
            else:
                if (link, obj) in collision_pairs or (obj, link) in collision_pairs:
                    return self._links_from_nodes[link._name].distance_to(obj)
                else:
                    raise ValueError(f"Collision pair ({link.name}, {obj.name}) not valid")
                

    def jacobian_proximity(self, q: np.ndarray, obj: PhysicsObject | Chain | Link, link : Link | None = None):
        
        proximities = self.distance_to(q, obj, link)

        J = np.zeros((len(proximities), self.dof))
        for i, p in enumerate(proximities):
            J[i] = p.normal_target_to_subject @ self.jacobian(q, p.subject._name, global_pose=trans_mat(pos=p.position_on_subject))[:3]
            if p.target._name in self._links_from_nodes:
                J[i] -= p.normal_target_to_subject @ self.jacobian(q, p.target._name, global_pose=trans_mat(pos=p.position_on_target))[:3]
        
        if J.shape[0] == 1:
            return J[0]
        return J

