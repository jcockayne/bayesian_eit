#include <Eigen/Core>
#include <memory>

#ifndef COLLOCATE_H

class CollocationResult {
public:
    const Eigen::MatrixXd mu_mult;
    const Eigen::MatrixXd cov;
    CollocationResult(Eigen::MatrixXd mu_mult, Eigen::MatrixXd cov) : mu_mult(mu_mult), cov(cov) { };
};

class CollocationMatrices {
public:
    CollocationMatrices(const Eigen::MatrixXd &kern, const Eigen::MatrixXd &left, const Eigen::MatrixXd &central) 
        : kern(kern), left(left), central(central) 
        { };
    const Eigen::MatrixXd kern;
    const Eigen::MatrixXd left;
    const Eigen::MatrixXd central;
};

class Collocator {
public:
    Collocator(const Eigen::Ref<const Eigen::MatrixXd> &x, int N_collocate, const Eigen::Ref<const Eigen::VectorXd> &kernel_args);
    std::unique_ptr<CollocationResult> collocate_no_obs(
        const Eigen::Ref<const Eigen::MatrixXd> &x,
        const Eigen::Ref<const Eigen::MatrixXd> &interior, 
        const Eigen::Ref<const Eigen::MatrixXd> &boundary, 
        const Eigen::Ref<const Eigen::MatrixXd> &sensors,
        const Eigen::Ref<const Eigen::VectorXd> &kernel_args
    );
    std::unique_ptr<CollocationMatrices> get_matrices(
        const Eigen::Ref<const Eigen::MatrixXd> &x,
        const Eigen::Ref<const Eigen::MatrixXd> &interior,
        const Eigen::Ref<const Eigen::MatrixXd> &boundary,
        const Eigen::Ref<const Eigen::MatrixXd> &sensors,
        const Eigen::Ref<const Eigen::VectorXd> &kernel_arg
    );
private:
    Eigen::MatrixXd _left;
    Eigen::MatrixXd _central;
    Eigen::MatrixXd _kern;
    void build_matrices(
        const Eigen::Ref<const Eigen::MatrixXd> &x,
        const Eigen::Ref<const Eigen::MatrixXd> &interior,
        const Eigen::Ref<const Eigen::MatrixXd> &boundary,
        const Eigen::Ref<const Eigen::MatrixXd> &sensors,
        const Eigen::Ref<const Eigen::VectorXd> &kernel_args
    );
};

std::unique_ptr<CollocationResult> collocate_no_obs(
    const Eigen::Ref<const Eigen::MatrixXd> &x,
    const Eigen::Ref<const Eigen::MatrixXd> &interior, 
    const Eigen::Ref<const Eigen::MatrixXd> &boundary, 
    const Eigen::Ref<const Eigen::MatrixXd> &sensors,
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args
);

std::unique_ptr<CollocationMatrices> collocate_matrices_no_obs(
    const Eigen::Ref<const Eigen::MatrixXd> &x,
    const Eigen::Ref<const Eigen::MatrixXd> &interior, 
    const Eigen::Ref<const Eigen::MatrixXd> &boundary, 
    const Eigen::Ref<const Eigen::MatrixXd> &sensors,
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args
);

#define COLLOCATE_H
#endif