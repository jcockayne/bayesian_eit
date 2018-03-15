import numpy as np


def theta_to_a(theta, grid, proposal_dot_mat):
    theta = np.real_if_close(theta)
    theta_mod = np.dot(proposal_dot_mat, theta[:,None])
    sz_int = len(grid.interior)
    sz_bdy = len(grid.boundary)
    sz_sensor = len(grid.sensors)
    kappa_int = theta_mod[:sz_int]
    kappa_bdy = theta_mod[sz_int:sz_int+sz_bdy]
    kappa_sensor = theta_mod[sz_int+sz_bdy:sz_int+sz_bdy+sz_sensor]
    grad_kappa_x = theta_mod[sz_int+sz_bdy+sz_sensor:2*sz_int+sz_bdy+sz_sensor]
    grad_kappa_y = theta_mod[2*sz_int+sz_bdy+sz_sensor:]
    return kappa_int, kappa_bdy, kappa_sensor, grad_kappa_x, grad_kappa_y