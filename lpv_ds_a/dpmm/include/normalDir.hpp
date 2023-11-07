#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>

#define PI 3.141592653589793

using namespace Eigen;

template<typename T>
class NormalDir
{
    public:
        NormalDir(const Matrix<T,Dynamic,1> &meanPos, const Matrix<T,Dynamic, Dynamic> &covPos,
        const Matrix<T,Dynamic,1>& meanDir, T covDir, boost::mt19937 &rndGen);   
        ~NormalDir(){};
        T logProb(const Matrix<T,Dynamic,1> &x_i);


    public:
        boost::mt19937 rndGen_;

        // parameters
        Matrix<T,Dynamic,1> meanPos_;
        Matrix<T,Dynamic,1> meanDir_;
        Matrix<T,Dynamic,1> mean_;
        Matrix<T,Dynamic,Dynamic> covPos_;
        T covDir_;
        Matrix<T,Dynamic,Dynamic> cov_;
        uint32_t dim_;
};
