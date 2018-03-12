#include <Eigen/Core>
#include <memory>

#ifndef COLLOCATE_H

enum CollocationSolver {
    LU = 0,
    LDLT = 1,
    QR = 2,
    SVD = 3
};

#define COLLOCATION_SOLVER_DEFAULT LU

class CollocationResult {
public:
    const Eigen::MatrixXd mu_mult;
    const Eigen::MatrixXd cov;
    const double solve_error;
    CollocationResult(Eigen::MatrixXd mu_mult, Eigen::MatrixXd cov, double solve_error) : mu_mult(mu_mult), cov(cov), solve_error(solve_error) { };
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
    Collocator(const Eigen::Ref<const Eigen::MatrixXd> &x, 
        int N_collocate, 
        const Eigen::Ref<const Eigen::VectorXd> &kernel_args, 
        CollocationSolver solver=COLLOCATION_SOLVER_DEFAULT
    );
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
    CollocationSolver _solver;
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
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args,
    CollocationSolver solver = COLLOCATION_SOLVER_DEFAULT
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