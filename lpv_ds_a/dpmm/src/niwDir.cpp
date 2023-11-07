#include "niwDir.hpp"
#include "karcher.hpp"
#include <cmath>
#include <limits>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>


template<typename T>
NIWDIR<T>::NIWDIR(const Matrix<T,Dynamic,Dynamic>& sigma, 
  const Matrix<T,Dynamic,Dynamic>& mu, T nu,  T kappa, boost::mt19937 &rndGen):
  Sigma_(sigma), mu_(mu), nu_(nu), kappa_(kappa), dim_(mu.size()), rndGen_(rndGen) //dim_ is dimParam defined in main.cpp
{
  if (mu.rows()==4)
  {
    muPos_ = mu_(seq(0, 1));
    SigmaPos_ = Sigma_(seq(0, 1), seq(0, 1));
    muDir_ = mu_(seq(2, 3));
    SigmaDir_ = Sigma_(2, 2);
  }
  else if (mu.rows()==6)
  {
    muPos_ = mu_(seq(0, 2));
    SigmaPos_ = Sigma_(seq(0, 2), seq(0, 2));
    muDir_ = mu_(seq(3, 5));
    SigmaDir_ = Sigma_(3, 3);
  }
};



template<typename T>
NIWDIR<T>::NIWDIR(const Matrix<T,Dynamic,1>& muPos, const Matrix<T,Dynamic,Dynamic>& SigmaPos,  
  const Matrix<T,Dynamic,1>& muDir, T SigmaDir,
  T nu, T kappa, T count, boost::mt19937 &rndGen):
  SigmaPos_(SigmaPos), SigmaDir_(SigmaDir), muPos_(muPos), muDir_(muDir), nu_(nu), kappa_(kappa), count_(count), rndGen_(rndGen) //dim_ is dimParam defined in main.cpp
{
  if (SigmaPos.cols()==2)
  {
    Sigma_.setZero(3, 3);
    Sigma_(seq(0,1), seq(0,1)) = SigmaPos_;
    Sigma_(2, 2) = SigmaDir_;
    mu_.setZero(3);
    mu_(seq(0,1)) = muPos_;
  }
  else if (SigmaPos.cols()==3)
  {
    Sigma_.setZero(4, 4);
    Sigma_(seq(0,2), seq(0,2)) = SigmaPos_;
    Sigma_(3, 3) = SigmaDir_;
    mu_.setZero(4);
    mu_(seq(0,2)) = muPos_;
  }
};



template<typename T>
NIWDIR<T>::~NIWDIR()
{};


template<typename T>
NIW<T> NIWDIR<T>::getNIW()
{
  return NIW<T>(SigmaPos_, muPos_, nu_, kappa_, rndGen_);
};



template<typename T>
void NIWDIR<T>::getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  int pos_dim;
  if (x_k.cols()==4) pos_dim=1;
  else if (x_k.cols()==6) pos_dim=2;

  meanDir_ = karcherMean(x_k);
  meanPos_ = x_k(all, (seq(0, pos_dim))).colwise().mean().transpose();  //!!!!later change the number 1 to accomodate for 3D data

  Matrix<T,Dynamic, Dynamic> x_k_mean;  //this is the value of each data point subtracted from the mean value calculated from the previous procedure
  x_k_mean = x_k.rowwise() - x_k.colwise().mean(); 
  ScatterPos_ = (x_k_mean.adjoint() * x_k_mean)(seq(0, pos_dim), seq(0, pos_dim)); //!!!!later change the number 1 to accomodate for 3D data
  ScatterDir_ = karcherScatter(x_k, meanDir_);  //!!! This is karcher scatter
  count_ = x_k.rows();
};


template<typename T>
NIWDIR<T> NIWDIR<T>::posterior(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  getSufficientStatistics(x_k);
  return NIWDIR<T>(
    (kappa_*muPos_+ count_*meanPos_)/(kappa_+count_),
    SigmaPos_+ScatterPos_ + ((kappa_*count_)/(kappa_+count_))*(meanPos_-muPos_)*(meanPos_-muPos_).transpose(),
    meanDir_,
    SigmaDir_+ScatterDir_ + ((kappa_*count_)/(kappa_+count_))*pow(rie_log(meanDir_, muDir_).norm(), 2),
    nu_+count_,
    kappa_+count_,
    count_, 
    rndGen_);
};


template<class T>
NormalDir<T> NIWDIR<T>::samplePosteriorParameter(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  NIWDIR<T> posterior = this ->posterior(x_k);
  return posterior.sampleParameter();
}


template<class T>
NormalDir<T> NIWDIR<T>::sampleParameter()
{

  int dim;
  if (SigmaPos_.cols()==2) dim =2;
  else if (SigmaPos_.cols()==3) dim =3;


  // std::cout << dim << std::endl;


  Matrix<T,Dynamic,1> meanPos(dim);
  Matrix<T,Dynamic,Dynamic> covPos(dim, dim);
  Matrix<T,Dynamic,1> meanDir(dim);
  T covDir;


  LLT<Matrix<T,Dynamic,Dynamic> > lltObj(SigmaPos_);
  Matrix<T,Dynamic,Dynamic> cholFacotor = lltObj.matrixL();

  Matrix<T,Dynamic,Dynamic> matrixA(dim, dim);
  matrixA.setZero();
  boost::random::normal_distribution<> gauss_;
  for (uint32_t i=0; i<dim; ++i)
  {
    boost::random::chi_squared_distribution<> chiSq_(nu_-i);
    matrixA(i,i) = sqrt(chiSq_(rndGen_)); 
    for (uint32_t j=i+1; j<dim; ++j)
    {
      matrixA(j, i) = gauss_(rndGen_);
    }
  }
  covPos = matrixA.inverse()*cholFacotor;
  covPos = covPos.transpose()*covPos;


  lltObj.compute(covPos);
  cholFacotor = lltObj.matrixL();

  for (uint32_t i=0; i<dim; ++i)
    meanPos[i] = gauss_(rndGen_);
  meanPos = cholFacotor * meanPos / sqrt(kappa_) + muPos_;


  // covDir = SigmaDir_;
  boost::random::chi_squared_distribution<> chiSq_(nu_);
  T inv_chi_sqrd = 1 / chiSq_(rndGen_);
  covDir = inv_chi_sqrd * SigmaDir_ / count_ * nu_;
  if (covDir > 0.2) covDir = 0.1;

  // std::cout << covDir << std::endl;

  // meanDir = muDir_; 
  if (dim==2)
  {
    boost::random::normal_distribution<> normal_(0, covDir/kappa_);
    T angDiff = normal_(rndGen_);
    Matrix<T,Dynamic,Dynamic> rotationMatrix(2, 2); // change the rotation matrix dimension later on to accomodate for 3D data
    rotationMatrix << cos(angDiff), -sin(angDiff), sin(angDiff), cos(angDiff);
    meanDir = (muDir_.transpose() * rotationMatrix).transpose();
  }
  else meanDir = muDir_;

  // std::cout << meanDir << std::endl;

  return NormalDir<T>(meanPos, covPos, meanDir, covDir, rndGen_);
};



template<class T>
T NIWDIR<T>::logProb(const Matrix<T,Dynamic,1>& x_i)
{
  // This is the log posterior predictive probability of x_i
  if (x_i.rows()==2)
  {
    int dim = 2;
    Matrix<T,Dynamic,Dynamic> Sigma(2, 2);
    Sigma = Sigma_(seq(0, 1), seq(0, 1));
    Matrix<T,Dynamic,1> mu(2);
    mu = mu_(seq(0, 1));
  
    T doF = nu_ - dim + 1.;
    Matrix<T,Dynamic,Dynamic> scaledSigma = Sigma*(kappa_+1.)/(kappa_*(nu_-dim+1));   
    LLT<Matrix<T,Dynamic,Dynamic>> lltObj(scaledSigma);

    T logProb = boost::math::lgamma(0.5*(doF + dim));
    logProb -= boost::math::lgamma(0.5*(doF));
    logProb -= 0.5*dim*log(doF);
    logProb -= 0.5*dim*log(PI);
    logProb -= 0.5*log(lltObj.matrixL().determinant());
    logProb -= (0.5*(doF + dim))
      *log(1.+ 1/doF*(lltObj.matrixL().solve(x_i-mu)).squaredNorm());
    return logProb;
  }
  else
  {
    int dim = 3;

    Matrix<T,Dynamic,1> x_i_new(dim);
    x_i_new.setZero();
    x_i_new(seq(0, dim-2)) = x_i(seq(0, dim-2));
    Matrix<T,Dynamic,1> x_i_dir(2);
    x_i_dir << x_i[dim-1] , x_i[dim];
    x_i_new(dim-1) = (rie_log(muDir_, x_i_dir)).norm();


    T doF = nu_ - dim + 1.;
    Matrix<T,Dynamic,Dynamic> scaledSigma = Sigma_*(kappa_+1.)/(kappa_*(nu_-dim+1));   
    LLT<Matrix<T,Dynamic,Dynamic>> lltObj(scaledSigma);

    T logProb = boost::math::lgamma(0.5*(doF + dim));
    logProb -= boost::math::lgamma(0.5*(doF));
    logProb -= 0.5*dim*log(doF);
    logProb -= 0.5*dim*log(PI);
    logProb -= 0.5*log(lltObj.matrixL().determinant());
    logProb -= (0.5*(doF + dim))
      *log(1.+ 1/doF*(lltObj.matrixL().solve(x_i_new-mu_)).squaredNorm());
    return logProb;
  }
};


template<class T>
T NIWDIR<T>::prob(const Matrix<T,Dynamic,1>& x_i)
{ 
  T logProb = this ->logProb(x_i);
  return exp(logProb);
};


template<typename T>
T NIWDIR<T>::logPosteriorProb(const Vector<T,Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k)
{
  NIWDIR<T> posterior = this ->posterior(x_k);
  return posterior.logProb(x_i);
};

template class NIWDIR<double>;
