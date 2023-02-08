# Numbotics

Numbotics is a library made to facilitate numerical analysis of robotic arms. The primary design feature of this library is its mathematical consistency. Many robotics software projects make dealing with underlying data tedious and confusing by wrapping every possible thought you might have in a class. The goal here is to use only vectors, matrices, and tensors to represent what you care about (the numbers), and keep the tedious robotics calculations encapsulated - like what classes were meant to do! 


## Robots

The Robot class holds all of the important functions and what not to get started. All robots are defined in this library by their DH parameters. A robot can be created using the following script:

```
import numbotics.robot as rob

l1 = rob.Link([1, 1.0, 0.0, 0.0, 0.0])
l2 = rob.Link([1, 1.0, 0.0, 0.0, 0.0])
l3 = rob.Link([1, 1.0, 0.0, 0.0, 0.0])

params = {'use_dyn': False,
          'use_gfx': False,
          'pos_dof': 2,
          'orn_dof': 0}

arm = rob.Robot(links=[l1,l2,l3], params=params)

```
This script creates a simple planar 3R robot with equal link lengths and two positional constraints (x and y positions). Because writing these scripts is rather tedious, an approach for robots whose structure does not need to be changed is using a `.rob` file. These files are laid out as follows:

```
# solver parameters
# use dynamics, use graphics, joints, pos dof, orn dof
  0             1             n       3        3


# kinematic parameters
# type  |  a  |  alpha  |  d  |  theta
  1       0.5     0.2     1.5     0.0
  0       1.2     0.9     0.0     1.4
  .        .       .       .       .
  .        .       .       .       .
  .        .       .       .       .
 t_n      a_n    alp_n    d_n    th_n

```
Of course, the table needs to be filled out with scalar values, and joints needs to be set to an actual number as well. The type field corresponds to whether or not the given joint is prismatic (0) or revolute (1). Also, the `pos_dof` and `orn_dof` fields represent the number of positioning and orienting degrees of freedom in the task space. The following table lists which task space variables are considered for each value of `pos_dof` and `orn_dof`

```
pos_dof  
1: (x)
2: (x,y)
3: (x,y,z)

orn_dof  
1: z axis
3: SO(3)

```

## Basics

Numpy arrays are the expected input for the majority of functions. If a textbook were to use a vector to represent something, say a joint configuration, then a vector is used here. Likewise, it's worth noting that coordinate frames are represented using homogenous transformation matrices in this library. So if a function, e.g. inverse kinematics, requires some position, orientation, or frame to be specified - that's right, just pass in a 4 by 4 matrix. This means that even if a robot's task space is not the full 6D task space, a 4 by 4 matrix must still be used - although the underlying algorithms will of course drop the orientations and positions variables that do not belong to the task space. It may seem tedious to always require 4 by 4 matrices even when only positions or orientations are necessary, but it reduces the ambiguity, plus there is a function in `spatial` called `trans_mat` which can be used with only a position and/or orientation argument to construct such a matrix. Plus, here's the kicker, even if you want to make your own 4 by 4 matrix to input into a function, there's nothing stopping you. No type errors, no missing attributes, no missing inheritance from `Spatial3dEulRot_External_Cache__` - just a 2D Numpy array doing the Lord's work.


## GPU

Several batch operations are supported at the moment, such as forward kinematics, forward kinematics error w.r.t. a batch of end-effector locations, and the Jacobian. These can be performed on the GPU by using what the Lord has blessed us with, PyTorch. There is no need to pass parameters in or type cast to try to finagle PyTorch to be used - these operations automagically check to see if the input variable was a `torch.Tensor` object, and if so they push the computations to the GPU. To set the default device to CPU, simply import `numbotics.config` and set `TORCH_DEV = torch.device('cpu')`.


## Future Plans

Ideally, the library would be expanded upon and cleaned up. Expansion included:
- [ ] Support for dynamics simulation
- [ ] Path planners
- [ ] Control algorithms

Right now the code is only in decent shape. Cleaning includes:
- [ ] Replacing asserts with proper exception handling
- [ ] Effective and concise comments - not the page long, undescriptive balogne Python is famous for
- [ ] Proper documentation - i.e. an actual webpage
