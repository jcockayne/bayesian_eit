from eigency.core cimport *
from libcpp.memory cimport unique_ptr
cimport numpy as np
from cython.operator cimport dereference as deref
from collocation_solvers cimport *


cdef extern from "collocate.hpp":
	cdef cppclass CollocationResult:
		MatrixXd mu_mult
		MatrixXd cov
		CollocationResult(MatrixXd mu_mult, MatrixXd cov)
	cdef cppclass CollocationMatrices:
		MatrixXd kern;
		MatrixXd left;
		MatrixXd central;
		CollocationMatrices(MatrixXd kern, MatrixXd left, MatrixXd central)
	cdef unique_ptr[CollocationResult] _collocate_no_obs "collocate_no_obs"(
		Map[MatrixXd] x,
		Map[MatrixXd] interior,
		Map[MatrixXd] boundary,
		Map[MatrixXd] sensors,
		Map[VectorXd] kernel_args,
		CollocationSolver solver
	)
	cdef unique_ptr[CollocationMatrices] _collocate_matrices_no_obs "collocate_matrices_no_obs"(
		Map[MatrixXd] x,
		Map[MatrixXd] interior,
		Map[MatrixXd] boundary,
		Map[MatrixXd] sensors,
		Map[VectorXd] kernel_args
	)

cdef extern from "likelihood.hpp":
	cdef double _log_likelihood "log_likelihood"(
		Map[MatrixXd] interior,
		Map[MatrixXd] boundary,
		Map[MatrixXd] sensors,
		Map[VectorXd] theta,
		Map[MatrixXd] theta_projection_mat,
		Map[VectorXd] kernel_args,
		Map[MatrixXd] stim_pattern,
		Map[MatrixXd] meas_pattern,
		Map[MatrixXd] data,
		double likelihood_variance,
		CollocationSolver solver,
		bint bayesian,
		bint debug
	)
	cdef double _log_likelihood_tempered "log_likelihood_tempered"(
		Map[MatrixXd] interior,
		Map[MatrixXd] boundary,
		Map[MatrixXd] sensors,
		Map[VectorXd] theta,
		Map[MatrixXd] theta_projection_mat,
		Map[VectorXd] kernel_args,
		Map[MatrixXd] stim_pattern,
		Map[MatrixXd] meas_pattern,
		Map[MatrixXd] data_1,
		Map[MatrixXd] data_2,
		double temp,
		double likelihood_variance,
		CollocationSolver solver,
		bint bayesian,
		bint debug
	)

def collocate_no_obs(
	np.ndarray[dtype=np.float_t, ndim=2] x,
	np.ndarray[dtype=np.float_t, ndim=2] interior,
	np.ndarray[dtype=np.float_t, ndim=2] boundary,
	np.ndarray[dtype=np.float_t, ndim=2] sensors,
	np.ndarray[dtype=np.float_t, ndim=1] kernel_args,
	solver="LDLT"
):
	cdef unique_ptr[CollocationResult] ret = _collocate_no_obs(
		Map[MatrixXd](x),
		Map[MatrixXd](interior),
		Map[MatrixXd](boundary),
		Map[MatrixXd](sensors),
		Map[VectorXd](kernel_args),
		solver_to_enum(solver)
	)
	mu_mult = ndarray_copy(deref(ret).mu_mult)
	cov = ndarray_copy(deref(ret).cov)
	return mu_mult, cov

def collocate_matrices_no_obs(
	np.ndarray[dtype=np.float_t, ndim=2] x,
	np.ndarray[dtype=np.float_t, ndim=2] interior,
	np.ndarray[dtype=np.float_t, ndim=2] boundary,
	np.ndarray[dtype=np.float_t, ndim=2] sensors,
	np.ndarray[dtype=np.float_t, ndim=1] kernel_args
):
	cdef unique_ptr[CollocationMatrices] ret = _collocate_matrices_no_obs(
		Map[MatrixXd](x),
		Map[MatrixXd](interior),
		Map[MatrixXd](boundary),
		Map[MatrixXd](sensors),
		Map[VectorXd](kernel_args)
	)
	kern = ndarray_copy(deref(ret).kern)
	left = ndarray_copy(deref(ret).left)
	central = ndarray_copy(deref(ret).central)
	return kern, left, central

def log_likelihood(
	np.ndarray[dtype=np.float_t, ndim=2] interior,
	np.ndarray[dtype=np.float_t, ndim=2] boundary,
	np.ndarray[dtype=np.float_t, ndim=2] sensors,
	np.ndarray[dtype=np.float_t, ndim=1] theta,
	np.ndarray[dtype=np.float_t, ndim=2] theta_projection_mat,
	np.ndarray[dtype=np.float_t, ndim=1] kernel_args,
	np.ndarray[dtype=np.float_t, ndim=2] stim_pattern,
	np.ndarray[dtype=np.float_t, ndim=2] meas_pattern,
	np.ndarray[dtype=np.float_t, ndim=2] data,
	double likelihood_variance,
	solver="LDLT",
	bint bayesian=True,
	bint debug=False
):
	return _log_likelihood(
		Map[MatrixXd](interior),
		Map[MatrixXd](boundary),
		Map[MatrixXd](sensors),
		Map[VectorXd](theta),
		Map[MatrixXd](theta_projection_mat),
		Map[VectorXd](kernel_args),
		Map[MatrixXd](stim_pattern),
		Map[MatrixXd](meas_pattern),
		Map[MatrixXd](data),
		likelihood_variance,
		solver_to_enum(solver),
		bayesian,
		debug
	)


def log_likelihood_tempered(
	np.ndarray[dtype=np.float_t, ndim=2] interior,
	np.ndarray[dtype=np.float_t, ndim=2] boundary,
	np.ndarray[dtype=np.float_t, ndim=2] sensors,
	np.ndarray[dtype=np.float_t, ndim=1] theta,
	np.ndarray[dtype=np.float_t, ndim=2] theta_projection_mat,
	np.ndarray[dtype=np.float_t, ndim=1] kernel_args,
	np.ndarray[dtype=np.float_t, ndim=2] stim_pattern,
	np.ndarray[dtype=np.float_t, ndim=2] meas_pattern,
	np.ndarray[dtype=np.float_t, ndim=2] data_1,
	np.ndarray[dtype=np.float_t, ndim=2] data_2,
	double temp,
	double likelihood_variance,
	solver="LDLT",
	bint bayesian=True,
	bint debug=False
):
	return _log_likelihood_tempered(
		Map[MatrixXd](interior),
		Map[MatrixXd](boundary),
		Map[MatrixXd](sensors),
		Map[VectorXd](theta),
		Map[MatrixXd](theta_projection_mat),
		Map[VectorXd](kernel_args),
		Map[MatrixXd](stim_pattern),
		Map[MatrixXd](meas_pattern),
		Map[MatrixXd](data_1),
		Map[MatrixXd](data_2),
		temp,
		likelihood_variance,
		solver_to_enum(solver),
		bayesian,
		debug
	)