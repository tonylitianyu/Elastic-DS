#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include "normalDir.hpp"
#include "niw.hpp"


using namespace Eigen;

template<typename T>
class NIWDIR
{
    public:
        NIWDIR(const Matrix<T,Dynamic,Dynamic>& sigma, const Matrix<T,Dynamic,Dynamic>& mu, T nu, T kappa,
        boost::mt19937 &rndGen);
        NIWDIR(const Matrix<T,Dynamic,1>& muPos, const Matrix<T,Dynamic,Dynamic>& SigmaPos, 
        const Matrix<T,Dynamic,1>& muDir, T SigmaDir, 
        T nu, T kappa, T count, boost::mt19937 &rndGen);        
        ~NIWDIR();

        NIW<T> getNIW();

        void getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k);
        NIWDIR<T> posterior(const Matrix<T,Dynamic, Dynamic>& x_k);
        NormalDir<T> samplePosteriorParameter(const Matrix<T,Dynamic, Dynamic> &x_k);
        NormalDir<T> sampleParameter();

        T logPosteriorProb(const Vector<T,Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k);
        T logProb(const Matrix<T,Dynamic,1>& x_i);
        T prob(const Matrix<T,Dynamic,1>& x_i);

    public:
        boost::mt19937 rndGen_;


        // Hyperparameters remain fixed once initialized
        Matrix<T,Dynamic,Dynamic> SigmaPos_;
        T SigmaDir_;
        Matrix<T,Dynamic,Dynamic> Sigma_;
        Matrix<T,Dynamic,1> muPos_;
        Matrix<T,Dynamic,1> muDir_;
        Matrix<T,Dynamic,1> mu_;
        T nu_,kappa_;
        uint32_t dim_;


        // Sufficient statistics changes everytime when posterior method is called
        Matrix<T,Dynamic,Dynamic> ScatterPos_;
        T ScatterDir_;
        Matrix<T,Dynamic,1> meanPos_;
        Matrix<T,Dynamic,1> meanDir_;
        uint16_t count_;
};

