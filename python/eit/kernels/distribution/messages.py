import pickle

def deserialize(message):
    return pickle.loads(message)


def serialize(message):
    return pickle.dumps(message)


class ApplyKernel(object):
    def __init__(self, kernel_id, kappa_0, n_iter, beta=None, bayesian=True):
        self.kernel_id = kernel_id
        self.kappa_0 = kappa_0
        self.n_iter = n_iter
        self.beta = beta
        self.bayesian = bayesian


class ClearKernels(object):
    def __init__(self, run_id=None):
        self.run_id = run_id


class CreateKernels(object):
    def __init__(self, 
        run_id, 
        init_beta,
        prior_mean,
        sqrt_prior_cov,
        grid,
        likelihood_variance,
        stim_pattern,
        all_data,
        temperatures,
        collocate_args,
        proposal_dot_mat
    ):
        self.run_id = run_id
        self.init_beta = init_beta
        self.prior_mean = prior_mean
        self.sqrt_prior_cov = sqrt_prior_cov
        self.likelihood_variance = likelihood_variance
        self.stim_pattern = stim_pattern
        self.all_data = all_data
        self.temperatures = temperatures
        self.collocate_args = collocate_args
        self.proposal_dot_mat = proposal_dot_mat