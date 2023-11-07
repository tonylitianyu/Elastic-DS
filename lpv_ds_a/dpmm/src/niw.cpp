#include "niw.hpp"
#include <cmath>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>


#define PI 3.141592653589793


template<class T>
NIW<T>::NIW(const Matrix<T,Dynamic,Dynamic> &Sigma, 
  const Matrix<T,Dynamic,Dynamic> &mu, T nu, T kappa, boost::mt19937 &rndGen)
: Sigma_(Sigma), mu_(mu), nu_(nu), kappa_(kappa), dim_(mu.size()), rndGen_(rndGen) 
{
  assert(Sigma_.rows()==mu_.size()); 
  assert(Sigma_.cols()==mu_.size());
};


template<class T>
NIW<T>::~NIW()
{};


template<class T>
T NIW<T>::logPosteriorProb(const Vector<T,Dynamic> &x_i, const Matrix<T,Dynamic, Dynamic> &x_k)
{
  NIW<T> posterior = this->posterior(x_k);
  return posterior.logProb(x_i);
};


// template<class T>
// T NIW<T>::logPosteriorProb(const Matrix<T,Dynamic, Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_j, )
// {
//   NIW<T> posterior = this ->posterior(x_k);

//   return posterior.logProb(x_i);
// };


template<class T>
NIW<T> NIW<T>::posterior(const Matrix<T,Dynamic, Dynamic> &x_k)
{
  getSufficientStatistics(x_k);
  return NIW<T>(
    Sigma_+Scatter_ + ((kappa_*count_)/(kappa_+count_))
      *(mean_-mu_)*(mean_-mu_).transpose(), 
    (kappa_*mu_+ count_*mean_)/(kappa_+count_),
    nu_+count_,
    kappa_+count_, rndGen_);
};


template<class T>
void NIW<T>::getSufficientStatistics(const Matrix<T,Dynamic, Dynamic> &x_k)
{
	mean_ = x_k.colwise().mean();
  Matrix<T,Dynamic, Dynamic> x_k_mean;
  x_k_mean = x_k.rowwise() - mean_.transpose();
  Scatter_ = x_k_mean.adjoint() * x_k_mean;
	count_ = x_k.rows();
};


template<class T>
T NIW<T>::logProb(const Matrix<T,Dynamic,1>& x_i)
{
  // std::cout << x_i << std::endl;
  // using multivariate student-t distribution; missing terms?
  // https://en.wikipedia.org/wiki/Multivariate_t-distribution
  // https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf pg.21
  T doF = nu_ - dim_ + 1.;
  Matrix<T,Dynamic,Dynamic> scaledSigma = Sigma_*(kappa_+1.)/(kappa_*(nu_-dim_+1));   
  // scaledSigma(dim_-1, dim_-1) = Sigma_(dim_-1, dim_-1); //testing the z-value effects if all zeros
                    
  T logProb = boost::math::lgamma(0.5*(doF + dim_));
  logProb -= boost::math::lgamma(0.5*(doF));
  logProb -= 0.5*dim_*log(doF);
  logProb -= 0.5*dim_*log(PI);
  logProb -= 0.5*log(scaledSigma.determinant());
  // logProb -= 0.5*((scaledSigma.eigenvalues()).array().log().sum()).real();
  logProb -= (0.5*(doF + dim_))
    *log(1.+ 1/doF*((x_i-mu_).transpose()*scaledSigma.inverse()*(x_i-mu_)).sum());
  // approximate using moment-matched Gaussian; Erik Sudderth PhD essay
  return logProb;
};


template<class T>
T NIW<T>::prob(const Matrix<T,Dynamic,1>& x_i)
{ 
  T logProb = this ->logProb(x_i);
  return exp(logProb);
};


template<class T>
Normal<T> NIW<T>::samplePosteriorParameter(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  NIW<T> posterior = this ->posterior(x_k);
  return posterior.sampleParameter();
}



template<class T>
Normal<T> NIW<T>::sampleParameter()
{
  Matrix<T,Dynamic,Dynamic> sampledCov(dim_,dim_);
  Matrix<T,Dynamic,1> sampledMean(dim_);

  LLT<Matrix<T,Dynamic,Dynamic> > lltObj(Sigma_);
  Matrix<T,Dynamic,Dynamic> cholFacotor = lltObj.matrixL();

  Matrix<T,Dynamic,Dynamic> matrixA(dim_,dim_);
  matrixA.setZero();
  boost::random::normal_distribution<> gauss_;
  for (uint32_t i=0; i<dim_; ++i)
  {
    boost::random::chi_squared_distribution<> chiSq_(nu_-i);
    matrixA(i,i) = sqrt(chiSq_(rndGen_)); 
    for (uint32_t j=i+1; j<dim_; ++j)
    {
      matrixA(j, i) = gauss_(rndGen_);
    }
  }
  sampledCov = matrixA.inverse()*cholFacotor;
  sampledCov = sampledCov.transpose()*sampledCov;


  lltObj.compute(sampledCov);
  cholFacotor = lltObj.matrixL();

  for (uint32_t i=0; i<dim_; ++i)
    sampledMean[i] = gauss_(rndGen_);
  sampledMean = cholFacotor * sampledMean / sqrt(kappa_) + mu_;

  
  return Normal<T>(sampledMean, sampledCov, rndGen_);
};




// template<class T>
// T NIW<T>::logPosteriorProb(const Matrix<T,Dynamic,Dynamic>& x, VectorXu& z, uint32_t k, uint32_t i)
// {
//   uint32_t z_i = z[i];
//   z[i] = k+1; // so that we definitely not use x_i in posterior computation 
//   // (since the posterior is only computed form x_{z==k})
//   NIW posterior = this->posterior(x,z,k);
//   z[i] = z_i; // reset to old value
//   return posterior.logProb(x.col(i));
// };



template class NIW<double>;