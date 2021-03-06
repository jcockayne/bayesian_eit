#include "operators.hpp"
#include "collocate.hpp"
#include <Eigen/Dense>
#include "memory_utils.hpp"
#include <iostream>
#include <fstream>
#include "logging/logging.hpp"
#include "debug.hpp"


Collocator::Collocator(
    const Eigen::Ref<const Eigen::MatrixXd> &x, 
    int N_collocate, 
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args, 
    CollocationSolver solver
) 
{
    _kern = Id_Id(x, x, kernel_args);
    _left = Eigen::MatrixXd(x.rows(), N_collocate);
    _central = Eigen::MatrixXd(N_collocate, N_collocate);
    _solver = solver;
}

std::unique_ptr<CollocationMatrices> Collocator::get_matrices(
    const Eigen::Ref<const Eigen::MatrixXd> &x,
    const Eigen::Ref<const Eigen::MatrixXd> &interior,
    const Eigen::Ref<const Eigen::MatrixXd> &boundary,
    const Eigen::Ref<const Eigen::MatrixXd> &sensors,
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args
)
{
    build_matrices(x, interior, boundary, sensors, kernel_args);
    return make_unique<CollocationMatrices>(_kern, _left, _central);
}


void Collocator::build_matrices(
    const Eigen::Ref<const Eigen::MatrixXd> &x,
    const Eigen::Ref<const Eigen::MatrixXd> &interior,
    const Eigen::Ref<const Eigen::MatrixXd> &boundary,
    const Eigen::Ref<const Eigen::MatrixXd> &sensors,
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args
)
{
    DEBUG_FUNCTION_ENTER;
    LOG_DEBUG("x shape is (" << x.rows() << "," << x.cols() << ").");
    LOG_DEBUG("Interior shape is (" << interior.rows() << "," << interior.cols() << ").");
    LOG_DEBUG("Boundary shape is (" << boundary.rows() << "," << boundary.cols() << ").");
    LOG_DEBUG("Sensors shape is (" << sensors.rows() << "," << sensors.cols() << ").");

    Eigen::MatrixXd all_boundary(boundary.rows() + sensors.rows(), boundary.cols());
    all_boundary << boundary, sensors;
    LOG_DEBUG("Generated the boundary points, shape is (" << all_boundary.rows() << "," << all_boundary.cols() << ").");
    
    // populate the _left matrix
    LOG_DEBUG("Starting population of left, which has shape (" << _left.rows() << "," << _left.cols() << ").");
    LOG_DEBUG("Populating _left Id_A.");
    Id_A(x, interior, kernel_args, _left.leftCols(interior.rows()));
    LOG_DEBUG("Populating _left Id_B.");
    Id_B(x, all_boundary, kernel_args, _left.rightCols(all_boundary.rows()));
    LOG_DEBUG("Done with _left.");

    // populate the _central matrix
    LOG_DEBUG("Populating _central A_A.");
    A_A(interior, interior, kernel_args, _central.topLeftCorner(interior.rows(), interior.rows()));
    LOG_DEBUG("Populating _central A_B.");
    A_B(interior, all_boundary, kernel_args, _central.topRightCorner(interior.rows(), all_boundary.rows()));
    LOG_DEBUG("Populating _central B_B.");
    B_B(all_boundary, all_boundary, kernel_args, _central.bottomRightCorner(all_boundary.rows(), all_boundary.rows()));
    LOG_DEBUG("Making _central symmetric.");
    _central.bottomLeftCorner(all_boundary.rows(), interior.rows()) 
        = _central.topRightCorner(interior.rows(), all_boundary.rows()).transpose();
    LOG_DEBUG("Populated matrices.");


    #ifdef ENABLE_LOG_DEBUG
    auto format = Eigen::IOFormat(Eigen::FullPrecision);
    std::ofstream file1("central.txt");
    file1 << _central.format(format);
    std::ofstream file2("left.txt");
    file2 << _left.format(format);
    std::ofstream file3("kern.txt");
    file3 << _kern.format(format);
    #endif

    DEBUG_FUNCTION_EXIT;
}



std::unique_ptr<CollocationResult> Collocator::collocate_no_obs(
    const Eigen::Ref<const Eigen::MatrixXd> &x,
    const Eigen::Ref<const Eigen::MatrixXd> &interior,
    const Eigen::Ref<const Eigen::MatrixXd> &boundary,
    const Eigen::Ref<const Eigen::MatrixXd> &sensors,
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args
)
{
    build_matrices(x, interior, boundary, sensors, kernel_args);

    //Id_Id(x, x, kernel_args, kern);
    
    // and lastly build the posterior
    // first invert the central matrix...
    Eigen::MatrixXd solution;
    switch(_solver) {
        case LU:
        {
            solution = _central.fullPivLu().solve(_left.transpose());
            break;
        }
        case LDLT: 
        {
            solution = _central.ldlt().solve(_left.transpose());
            break;
        }
        case QR:
        {
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(_central);
            solution = _central.colPivHouseholderQr().solve(_left.transpose());
            break;
        }
        case SVD:
        {
            solution = _central.jacobiSvd().solve(_left.transpose());
            break;
        }
        default:
            throw "Solver type not understood!";
    }
    double error_norm = 0;
    #ifdef ENABLE_LOG_DEBUG
    {
        error_norm = (_central*solution - _left.transpose()).norm();
        LOG_DEBUG("Solution error norm is " << error_norm);
    }
    #endif
    Eigen::MatrixXd mu_mult = solution.transpose();
    Eigen::MatrixXd cov = _kern - mu_mult * _left.transpose();
    LOG_DEBUG("Built posterior.");

    DEBUG_FUNCTION_EXIT;
    return make_unique<CollocationResult>(mu_mult, cov, error_norm);
}

std::unique_ptr<CollocationResult> collocate_no_obs(
    const Eigen::Ref<const Eigen::MatrixXd> &x,
    const Eigen::Ref<const Eigen::MatrixXd> &interior, 
    const Eigen::Ref<const Eigen::MatrixXd> &boundary,
    const Eigen::Ref<const Eigen::MatrixXd> &sensors,
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args,
    CollocationSolver solver
)
{
    Collocator collocator(x, interior.rows() + boundary.rows() + sensors.rows(), kernel_args, solver);
    return collocator.collocate_no_obs(x, interior, boundary, sensors, kernel_args);
}

std::unique_ptr<CollocationMatrices> collocate_matrices_no_obs(
    const Eigen::Ref<const Eigen::MatrixXd> &x,
    const Eigen::Ref<const Eigen::MatrixXd> &interior, 
    const Eigen::Ref<const Eigen::MatrixXd> &boundary, 
    const Eigen::Ref<const Eigen::MatrixXd> &sensors,
    const Eigen::Ref<const Eigen::VectorXd> &kernel_args
)
{
    Collocator collocator(x, interior.rows() + boundary.rows() + sensors.rows(), kernel_args);
    return collocator.get_matrices(x, interior, boundary, sensors, kernel_args);
}