import numpy as np
from .. import collocate, simulate
from .shared import theta_to_a


def construct_posterior(locations, grid, theta, collocate_args, proposal_dot_mat, debug=False):
    a_int, a_bdy, a_sensor, a_x, a_y = theta_to_a(theta,
                                                  grid,
                                                  proposal_dot_mat
                                                  )
    assert a_int.shape[0] == grid.interior.shape[0]
    assert a_x.shape[0] == grid.interior.shape[0]
    assert a_y.shape[0] == grid.interior.shape[0]
    
    augmented_int = np.column_stack([grid.interior, a_int, a_x, a_y])
    augmented_bdy = np.column_stack([grid.boundary, a_bdy, np.nan * np.zeros((a_bdy.shape[0], 2))])
    augmented_sens = np.column_stack([grid.sensors, a_sensor, np.nan * np.zeros((a_sensor.shape[0], 2))])
    mu_mult, Sigma = collocate.collocate_no_obs(
        np.asfortranarray(locations),
        np.asfortranarray(augmented_int),
        np.asfortranarray(augmented_bdy),
        np.asfortranarray(augmented_sens),
        np.asfortranarray(collocate_args)
    )
    return mu_mult, Sigma


def phi(grid, theta, likelihood_variance, pattern, data, collocate_args, proposal_dot_mat, bayesian=True, debug=False):
    return -collocate.log_likelihood(
        np.asfortranarray(grid.interior),
        np.asfortranarray(grid.boundary),
        np.asfortranarray(grid.sensors),
        np.asfortranarray(theta),
        np.asfortranarray(proposal_dot_mat),
        np.asfortranarray(collocate_args),
        np.asfortranarray(pattern.stim_pattern),
        np.asfortranarray(pattern.meas_pattern),
        np.asfortranarray(data),
        likelihood_variance,
        bayesian=bayesian,
        debug=debug
    )


def phi_tempered(grid, theta, likelihood_variance, pattern, data_1, data_2, temp, collocate_args, proposal_dot_mat, bayesian=True, debug=False):
    return -collocate.log_likelihood_tempered(
        np.asfortranarray(grid.interior),
        np.asfortranarray(grid.boundary),
        np.asfortranarray(grid.sensors),
        np.asfortranarray(theta),
        np.asfortranarray(proposal_dot_mat),
        np.asfortranarray(collocate_args),
        np.asfortranarray(pattern.stim_pattern),
        np.asfortranarray(pattern.meas_pattern),
        np.asfortranarray(data_1),
        np.asfortranarray(data_2),
        temp,
        likelihood_variance,
        bayesian=bayesian,
        debug=debug
    )


class PCNKernel_C(object):
    def __init__(self, beta, prior_mean, sqrt_prior_cov, grid, likelihood_variance, pattern, data, collocate_args, proposal_dot_mat):
        self.__beta__ = beta
        self.__prior_mean__ = prior_mean
        self.__sqrt_prior_cov__ = sqrt_prior_cov   
        self.__grid__ = grid
        self.__likelihood_variance__ = likelihood_variance
        self.__pattern__ = pattern
        self.__data__ = data
        self.collocate_args = collocate_args
        self.__proposal_dot_mat__ = proposal_dot_mat



    def phi(self, theta, collocate_args=None, bayesian=True, debug=False):
        return phi(
            self.__grid__,
            theta,
            self.__likelihood_variance__,
            self.__pattern__,
            self.__data__,
            self.collocate_args if collocate_args is None else collocate_args,
            self.__proposal_dot_mat__,
            bayesian=bayesian,
            debug=debug
        )

    def get_posterior(self, theta, locations, stim=None):
        mu_mult, cov = construct_posterior(
            locations, 
            self.__grid__, 
            theta, 
            self.collocate_args, 
            self.__proposal_dot_mat__
        )
        if stim is None:
            return mu_mult, cov
        mu = np.dot(mu_mult, np.r_[
            np.zeros(len(self.__grid__.interior_plus_boundary)),
            stim
        ])
        return mu, cov

    def apply(self, kappa_0, n_iter, n_threads=1, beta=None, bayesian=True):
        if len(kappa_0.shape) == 1:
            kappa_0 = np.copy(kappa_0[None, :])
        return simulate.run_pcn_parallel(
            n_iter,
            self.__beta__ if beta is None else beta,
            np.asfortranarray(kappa_0),
            np.asfortranarray(self.__prior_mean__),
            np.asfortranarray(self.__sqrt_prior_cov__),
            np.asfortranarray(self.__grid__.interior),
            np.asfortranarray(self.__grid__.boundary),
            np.asfortranarray(self.__grid__.sensors),
            np.asfortranarray(self.__proposal_dot_mat__),
            np.asfortranarray(self.collocate_args),
            np.asfortranarray(self.__pattern__.stim_pattern),
            np.asfortranarray(self.__pattern__.meas_pattern),
            np.asfortranarray(self.__data__),
            self.__likelihood_variance__,
            n_threads,
            bayesian=bayesian
        )


class PCNTemperingKernel_C(object):
    def __init__(self, beta, prior_mean, sqrt_prior_cov, grid, likelihood_variance, pattern, data_1, data_2, temp, collocate_args, proposal_dot_mat):
        self.__beta__ = beta
        self.__prior_mean__ = prior_mean
        self.__sqrt_prior_cov__ = sqrt_prior_cov   
        self.__grid__ = grid
        self.__likelihood_variance__ = likelihood_variance
        self.__pattern__ = pattern
        self.__data_1__ = data_1
        self.__data_2__ = data_2
        self.__temp__ = temp
        self.__collocate_args__ = collocate_args
        self.__proposal_dot_mat__ = proposal_dot_mat

    def phi(self, theta, bayesian=True, debug=False):
        return phi_tempered(
            self.__grid__,
            theta,
            self.__likelihood_variance__,
            self.__pattern__,
            self.__data_1__,
            self.__data_2__,
            self.__temp__,
            self.__collocate_args__,
            self.__proposal_dot_mat__,
            bayesian=bayesian,
            debug=debug
        )

    def get_posterior(self, theta, locations):
        return construct_posterior(
            locations, 
            self.__grid__, 
            theta, 
            self.__collocate_args__, 
            self.__proposal_dot_mat__
        )

    def apply(self, kappa_0, n_iter, n_threads=1, beta=None, bayesian=True):
        if len(kappa_0.shape) == 1:
            kappa_0 = np.copy(kappa_0[None, :])
        return simulate.run_pcn_parallel_tempered(
            n_iter,
            self.__beta__ if beta is None else beta,
            np.asfortranarray(kappa_0),
            np.asfortranarray(self.__prior_mean__),
            np.asfortranarray(self.__sqrt_prior_cov__),
            np.asfortranarray(self.__grid__.interior),
            np.asfortranarray(self.__grid__.boundary),
            np.asfortranarray(self.__grid__.sensors),
            np.asfortranarray(self.__proposal_dot_mat__),
            np.asfortranarray(self.__collocate_args__),
            np.asfortranarray(self.__pattern__.stim_pattern),
            np.asfortranarray(self.__pattern__.meas_pattern),
            np.asfortranarray(self.__data_1__),
            np.asfortranarray(self.__data_2__),
            self.__temp__,
            self.__likelihood_variance__,
            n_threads,
            bayesian=bayesian
        )
