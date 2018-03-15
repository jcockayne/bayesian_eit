from scipy import stats
import bayesian_pdes as bpdes
import numpy as np
import contextlib
from .shared import theta_to_a


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)


def phi(grid, op_system, theta, likelihood_variance, pattern, data, collocate_args, proposal_dot_mat, use_c=False, debug=False):
    # first solve forward
    design_int = grid.interior_plus_boundary
    # now determine voltage at the sensor locations
    # we have seven observations so take one for each sensor other than sensor 1, the reference sensor
    augmented_locations = np.column_stack([grid.sensors, np.nan * np.zeros((8, 3))])

    if use_c:
        mu_mult, Sigma = construct_c_posterior(augmented_locations, grid, theta, collocate_args, proposal_dot_mat, debug=debug)
    else:
        posterior = construct_posterior(grid, op_system, theta, collocate_args, proposal_dot_mat, debug=debug)
        mu_mult, Sigma = posterior.no_obs_posterior(augmented_locations)

    

    # now need to iterate the stim patterns and compute the residual
    rhs_int = np.zeros((len(design_int), 1))

    Sigma_obs = np.dot(pattern.meas_pattern, np.dot(Sigma, pattern.meas_pattern.T))
    likelihood_cov = Sigma_obs + likelihood_variance * np.eye(Sigma_obs.shape[0])
    # likelihood_cov = likelihood_variance*np.eye(Sigma_obs.shape[0])
    likelihood_dist = stats.multivariate_normal(np.zeros(Sigma_obs.shape[0]), likelihood_cov)

    if debug:
        print(likelihood_cov)

    likelihood = 0
    for voltage, current in zip(data, pattern.stim_pattern):
        rhs_bdy = current[:, None]
        rhs = np.row_stack([rhs_int, rhs_bdy])

        model_voltage = np.dot(pattern.meas_pattern, np.dot(mu_mult, rhs))

        residual = voltage.ravel() - model_voltage.ravel()
        this_likelihood = likelihood_dist.logpdf(residual)
        if debug:
            with printoptions(precision=4, suppress=True):
                print("Model | True| Residual | Diag(cov)\n {}".format(np.c_[model_voltage, voltage, residual, np.diag(likelihood_cov)]))
            print("Likelihood: {}   |   Residual: {}".format(this_likelihood, np.abs(residual).sum()))
        likelihood += this_likelihood
    return -likelihood


def construct_posterior(grid, op_system, theta, collocate_args, proposal_dot_mat, debug=False):
    design_int = grid.interior
    a_int, a_bdy, a_sensor, a_x, a_y = theta_to_a(theta,
                                                  grid,
                                                  proposal_dot_mat
                                                  )

    augmented_int = np.column_stack([design_int, a_int, a_x, a_y])
    augmented_bdy = np.column_stack([grid.bdy, a_bdy, np.nan * np.zeros((a_bdy.shape[0], 2))])
    augmented_sens = np.column_stack([grid.sensors, a_sensor, np.nan * np.zeros((a_sensor.shape[0], 2))])
    obs = [
        (augmented_int, None),
        (augmented_bdy, None),
        (augmented_sensor, None)
    ]
    posterior = bpdes.collocate(
        op_system.operators,
        op_system.operators_bar,
        obs,
        op_system,
        collocate_args,
        inverter='np'
    )
    return posterior