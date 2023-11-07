#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>

#define PI 3.141592653589793

using namespace Eigen;

template<typename T>
class Normal
{
    public:
        Normal(const Matrix<T,Dynamic,1> &mu, const Matrix<T,Dynamic,Dynamic> &Sigma, boost::mt19937 &rndGen);
        ~Normal();
        T logProb(const Matrix<T,Dynamic,1> &x_i);

        // T logPosteriorProb(const Vector<T,Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k);
        // T logProb(const Matrix<T,Dynamic,1>& x_i);
        // NIW<T> posterior(const Matrix<T,Dynamic, Dynamic>& x_k);
        // void getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k);
        // Parameter sampleParameter();

    public:
        boost::mt19937 rndGen_;

        // parameters
        Matrix<T,Dynamic,1> mu_;
        Matrix<T,Dynamic,Dynamic> Sigma_;
        uint32_t dim_;
};
