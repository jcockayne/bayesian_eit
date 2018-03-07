#include <Eigen/Dense>
#include <math.h>
#include "collocate.hpp"
#include "likelihood.hpp"
#include <iostream>
#include <fstream>
#include "logging/logging.hpp"
#include "debug.hpp"
#include "memory_utils.hpp"

double log_likelihood(
    const Eigen::Ref<const Eigen::MatrixXd> &interior,
    const Eigen::Ref<const Eigen::MatrixXd> &boundary,
    const Eigen::Ref<const Eigen::MatrixXd> &sensors,
    const Eigen::Ref<const Eigen::VectorXd> &theta,
    const Eigen::Ref<const Eigen::MatrixXd> &theta_projection_mat,
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args,
    const Eigen::Ref<const Eigen::MatrixXd> &stim_pattern,
    const Eigen::Ref<const Eigen::MatrixXd> &meas_pattern,
    const Eigen::Ref<const Eigen::MatrixXd> &data,
    double likelihood_variance,
    CollocationSolver solver,
    bool bayesian,
    bool debug
)
{
    std::unique_ptr<Collocator> collocator = make_unique<Collocator>(sensors, interior.rows() + boundary.rows() + sensors.rows(), kernel_args, solver);
    return log_likelihood(
        interior,
        boundary,
        sensors,
        theta,
        theta_projection_mat,
        kernel_args,
        stim_pattern,
        meas_pattern,
        data,
        likelihood_variance,
        collocator.get(),
        bayesian,
        debug
    );
}

double log_likelihood(
    const Eigen::Ref<const Eigen::MatrixXd> &interior,
    const Eigen::Ref<const Eigen::MatrixXd> &boundary,
    const Eigen::Ref<const Eigen::MatrixXd> &sensors,
    const Eigen::Ref<const Eigen::VectorXd> &theta,
    const Eigen::Ref<const Eigen::MatrixXd> &theta_projection_mat,
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args,
    const Eigen::Ref<const Eigen::MatrixXd> &stim_pattern,
    const Eigen::Ref<const Eigen::MatrixXd> &meas_pattern,
    const Eigen::Ref<const Eigen::MatrixXd> &data,
    double likelihood_variance,
    Collocator *collocator,
    bool bayesian,
    bool debug
)
{
    return log_likelihood_tempered(
        interior,
        boundary,
        sensors,
        theta,
        theta_projection_mat,
        kernel_args,
        stim_pattern,
        meas_pattern,
        data,
        Eigen::MatrixXd(0, 0),
        0.,
        likelihood_variance,
        collocator,
        bayesian,
        debug
    );
}

double log_likelihood_tempered(
    const Eigen::Ref<const Eigen::MatrixXd> &interior,
    const Eigen::Ref<const Eigen::MatrixXd> &boundary,
    const Eigen::Ref<const Eigen::MatrixXd> &sensors,
    const Eigen::Ref<const Eigen::VectorXd> &theta,
    const Eigen::Ref<const Eigen::MatrixXd> &theta_projection_mat,
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args,
    const Eigen::Ref<const Eigen::MatrixXd> &stim_pattern,
    const Eigen::Ref<const Eigen::MatrixXd> &meas_pattern,
    const Eigen::Ref<const Eigen::MatrixXd> &data_1,
    const Eigen::Ref<const Eigen::MatrixXd> &data_2,
    double temperature,
    double likelihood_variance,
    CollocationSolver solver,
    bool bayesian,
    bool debug
)
{
    std::unique_ptr<Collocator> collocator = make_unique<Collocator>(sensors, interior.rows() + boundary.rows() + sensors.rows(), kernel_args, solver);
    return log_likelihood_tempered(
        interior,
        boundary,
        sensors,
        theta,
        theta_projection_mat,
        kernel_args,
        stim_pattern,
        meas_pattern,
        data_1,
        data_2,
        temperature,
        likelihood_variance,
        collocator.get(),
        bayesian,
        debug
    );
}

double log_likelihood_tempered(
    const Eigen::Ref<const Eigen::MatrixXd> &interior,
    const Eigen::Ref<const Eigen::MatrixXd> &boundary,
    const Eigen::Ref<const Eigen::MatrixXd> &sensors,
    const Eigen::Ref<const Eigen::VectorXd> &theta,
    const Eigen::Ref<const Eigen::MatrixXd> &theta_projection_mat,
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args,
    const Eigen::Ref<const Eigen::MatrixXd> &stim_pattern,
    const Eigen::Ref<const Eigen::MatrixXd> &meas_pattern,
    const Eigen::Ref<const Eigen::MatrixXd> &data_1,
    const Eigen::Ref<const Eigen::MatrixXd> &data_2,
    double temperature,
    double likelihood_variance,
    Collocator *collocator,
    bool bayesian,
    bool debug
)
{
    DEBUG_FUNCTION_ENTER;
    /*
    std::cout << "Bayesian = " << bayesian << std::endl;
    std::cout << "Temperature = " << temperature << std::endl;
    std::cout << "Data_1: " << data_1.rows() << std::endl;
    std::cout << "Data_2: " << data_2.rows() << std::endl;  
    */
    int counter = 0;
    Eigen::VectorXd projected_theta = theta_projection_mat*theta;
    Eigen::VectorXd theta_int = projected_theta.topRows(interior.rows()); counter += theta_int.rows();
    Eigen::VectorXd theta_bdy = projected_theta.middleRows(counter, boundary.rows()); counter += theta_bdy.rows();
    Eigen::VectorXd theta_sens = projected_theta.middleRows(counter, sensors.rows()); counter += theta_sens.rows();
    Eigen::VectorXd theta_x = projected_theta.segment(counter, interior.rows());
    Eigen::VectorXd theta_y = projected_theta.bottomRows(interior.rows());

    LOG_DEBUG("Calculated theta.");
    // augment the interior, sensors with theta
    Eigen::MatrixXd augmented_int(interior.rows(), 5);
    augmented_int << interior, theta_int, theta_x, theta_y;
    Eigen::MatrixXd augmented_bdy(boundary.rows(), 5);
    augmented_bdy.leftCols(3) << boundary, theta_bdy;
    Eigen::MatrixXd augmented_sens(sensors.rows(), 5);
    augmented_sens.leftCols(3) << sensors, theta_sens;
    

    LOG_DEBUG("Augmented input points.");

    std::unique_ptr<CollocationResult> posterior;
    posterior = collocator->collocate_no_obs(augmented_sens, augmented_int, augmented_bdy, augmented_sens, kernel_args);

    LOG_DEBUG("Called collocate.");

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(posterior->mu_mult.cols());
    int n_meas = stim_pattern.rows();
    Eigen::MatrixXd likelihood_cov = likelihood_variance*Eigen::MatrixXd::Identity(n_meas, n_meas);
    if(bayesian) {
        likelihood_cov += meas_pattern*posterior->cov*meas_pattern.transpose();
    }
    
    /*
    std::ofstream file1("likelihood_cov.txt");
    file1 << likelihood_cov;
    */

    auto likelihood_cov_decomp = likelihood_cov.llt();
    Eigen::MatrixXd L = likelihood_cov_decomp.matrixL();
    double halflogdet = 0;
    for(int i = 0; i < L.rows(); i++)
        halflogdet += log(L(i,i));
    double halflog2pi = 0.5*log(2*M_PI);
    double log_norm_const = -halflog2pi*n_meas - halflogdet;

    LOG_DEBUG("Log normalising constant = " << log_norm_const);

    Eigen::MatrixXd left_model_mult = meas_pattern * posterior->mu_mult;
    double likelihood_1 = 0;
    double likelihood_2 = 0;
    if(temperature < 1 && data_1.rows() > 0) {
        for(int i = 0; i < data_1.rows(); i++) {
            rhs.bottomRows(stim_pattern.cols()) = stim_pattern.row(i).transpose();
            Eigen::VectorXd residual = left_model_mult*rhs - data_1.row(i).transpose();

            double this_likelihood = -0.5*residual.dot(likelihood_cov_decomp.solve(residual)) + log_norm_const;
            likelihood_1 += this_likelihood;
        }
        //std::cout << "Likelihood 1 " << likelihood_1 << std::endl;
    }


    if(temperature > 0 && data_2.rows() > 0) {
        for(int i = 0; i < data_2.rows(); i++) {
            rhs.bottomRows(stim_pattern.cols()) = stim_pattern.row(i).transpose();
            Eigen::VectorXd residual = left_model_mult*rhs - data_2.row(i).transpose();

            double this_likelihood = -0.5*residual.dot(likelihood_cov_decomp.solve(residual)) + log_norm_const;
            likelihood_2 += this_likelihood;
        }
        //std::cout << "Likelihood 2 " << likelihood_2 << std::endl;
    }
    DEBUG_FUNCTION_EXIT;
    return likelihood_1*(1-temperature) + likelihood_2*temperature;
}