# Numbotics

Numbotics is a library made to facilitate numerical analysis of robots. The primary design feature of this library is its mathematical consistency. Many robotics software projects make dealing with underlying data tedious and confusing by wrapping every possible thought you might have in a class. The goal here is to use only vectors, matrices, and tensors to represent what you care about (the numbers), and keep the tedious robotics calculations encapsulated - like what classes were meant to do! 


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
Of course, the table needs to be filled out with scalar values, and joints needs to be set to an actual number as well. The type field corresponds to whether or not the given joint is prismatic (0) or revolute (1).


## Basics

Numpy arrays are the expected input for the majority of functions. If the function you are using has some mathematical interpretation, let's say it uses a joint configuration, then it can be expected that the input's most likely mathematical form will work here too, i.e. an n by 1 vector. Likewise, just as DH parameters are the de facto representation of kinematic structure, so are homogenous transformation matrices for cooridnate frames. Thus, Numbotics expects the input to anything expecting a homogenous transformation to be a 4 by 4 Numpy array. Even when performing the inverse kinematics of a robot that does not have any orientation constraints, a 4 by 4 Numpy array is still expected. It may seem tedious to do so, but it reduces the ambiguity, plus there is a function in `spatial` called `trans_mat` which can be used with only a position and/or orientation argument to construct such a matrix. But, here's the kicker, even if you want to make your own 4 by 4 matrix to input into a function, there's nothing stopping you. No type errors, no missing attributes, no missing inheritance from `Spatial3dEulRot_External_Cache__` - just a 2D Numpy array doing the Lord's work.


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
