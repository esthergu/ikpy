# coding= utf8
import scipy.optimize
import numpy as np
from . import logs


def inverse_kinematic_optimization(chain, target_frame, starting_nodes_angles, regularization_parameter=None, max_iter=None):
    """
    Computes the inverse kinematic on the specified target with an optimization method

    Parameters
    ----------
    chain: ikpy.chain.Chain
        The chain used for the Inverse kinematics.
    target_frame: numpy.array
        The desired target.
    starting_nodes_angles: numpy.array
        The initial pose of your chain.
    regularization_parameter: float
        The coefficient of the regularization.
    max_iter: int
        Maximum number of iterations for the optimisation algorithm.
    """
    target_pos = target_frame[:3, 3]
    target_orient = target_frame[:-1, 0:3]

    if starting_nodes_angles is None:
        raise ValueError("starting_nodes_angles must be specified")

    def optimize_target(x):
        y = chain.active_to_full(x, starting_nodes_angles)
        forward_matrix = chain.forward_kinematics(y)
        quaternion_distance = np.linalg.norm(forward_matrix[:-1, 0:3] - target_orient)
        return quaternion_distance

    # Compute squared distance to target
    def constraint(x):
        y = chain.active_to_full(x, starting_nodes_angles)
        forward_matrix = chain.forward_kinematics(y)
        squared_distance = np.linalg.norm(forward_matrix[:3, -1] - target_pos)
        return 0.0001 - squared_distance

    # If a regularization is selected
    if regularization_parameter is not None:
        def optimize_total(x):
            regularization = np.linalg.norm(x - starting_nodes_angles[chain.first_active_joint:])
            return optimize_target(x) + regularization_parameter * regularization
    else:
        def optimize_total(x):
            return optimize_target(x)

    # Compute bounds
    real_bounds = [link.bounds for link in chain.links]
    # real_bounds = real_bounds[chain.first_active_joint:]
    real_bounds = chain.active_from_full(real_bounds)

    options = {}
    # Manage iterations maximum
    # if max_iter is not None:
    options["maxiter"] = 10
    options["ftol"] = 0.01
    
    # Utilisation d'une optimisation L-BFGS-B
    # quaternion_distance = float("inf")
    # while quaternion_distance > 0.01:
    #     print("starting_nodes_angles")
    #     print(starting_nodes_angles)
    #     res = scipy.optimize.minimize(optimize_total, chain.active_from_full(starting_nodes_angles), method='L-BFGS-B', bounds=real_bounds, options=options)
    #     y = chain.active_to_full(res.x, starting_nodes_angles)
    #     forward_matrix = chain.forward_kinematics(y)
    #     quaternion_distance = np.linalg.norm(forward_matrix[:-1, 0:3] - target_orient)
    #     print(quaternion_distance)
    #     starting_nodes_angles = y

    # Utilisation d'une optimisation SLSQP_**
    res = scipy.optimize.minimize(optimize_total, chain.active_from_full(starting_nodes_angles), method='SLSQP', bounds=real_bounds, options=options, constraints={"fun": constraint, "type": "ineq"})
    print("iteration")
    print(res.nit)
    # res = scipy.optimize.minimize(optimize_total, chain.active_from_full(starting_nodes_angles), method='L-BFGS-B', bounds=real_bounds, options=options)

    logs.logger.info("Inverse kinematic optimisation OK, done in {} iterations".format(res.nit))

    return chain.active_to_full(res.x, starting_nodes_angles)
    # return(np.append(starting_nodes_angles[:chain.first_active_joint], res.x))
