#include "normalDir.hpp"
#include "karcher.hpp"


#define PI 3.141592653589793


template<class T>
NormalDir<T>::NormalDir(const Matrix<T,Dynamic,1> &meanPos, const Matrix<T,Dynamic, Dynamic> &covPos,
const Matrix<T,Dynamic,1>& meanDir, T covDir, boost::mt19937 &rndGen) 
:meanPos_(meanPos), covPos_(covPos), meanDir_(meanDir), covDir_(covDir), rndGen_(rndGen)
{
  if (meanPos.rows()==2)
  {
    cov_.setZero(3, 3);
    cov_(seq(0,1), seq(0,1)) = covPos_;
    cov_(2, 2) = covDir_;
    mean_.setZero(3);
    mean_(seq(0,1)) = meanPos_;
  }
  else if (meanPos.rows()==3)
  {
    cov_.setZero(4, 4);
    cov_(seq(0,2), seq(0,2)) = covPos_;
    cov_(3, 3) = covDir_;
    mean_.setZero(4);
    mean_(seq(0,2)) = meanPos_;
  }
};


template<class T>
T NormalDir<T>::logProb(const Matrix<T,Dynamic,1> &x_i)
{ 
  if (x_i.rows()==2) //3D data only pos 
  {
    int dim = 2;
    Matrix<T,Dynamic,Dynamic> cov(2, 2);
    cov = cov_(seq(0, 1), seq(0, 1));
    Matrix<T,Dynamic,1> mean(2);
    mean = mean_(seq(0, 1));
  
    LLT<Matrix<T,Dynamic,Dynamic>> lltObj(cov);
    T logProb =  dim * log(2*PI);
    logProb += 2 * log(lltObj.matrixL().determinant());
    logProb += (lltObj.matrixL().solve(x_i-mean)).squaredNorm();
    return -0.5 * logProb;
  }
  else if (x_i.rows()==4) //2D data full pos and dir
  {
    int dim = 3;
    Matrix<T,Dynamic,1> x_i_new(dim);
    x_i_new.setZero();
    x_i_new(seq(0, dim-2)) = x_i(seq(0, dim-2));
    Matrix<T,Dynamic,1> x_i_dir(2);
    x_i_dir << x_i[dim-1] , x_i[dim];
    x_i_new(dim-1) = (rie_log(meanDir_, x_i_dir)).norm();
  
    LLT<Matrix<T,Dynamic,Dynamic>> lltObj(cov_);
    T logProb =  dim * log(2*PI);
    logProb += 2 * log(lltObj.matrixL().determinant());
    logProb += (lltObj.matrixL().solve(x_i_new-mean_)).squaredNorm();
    return -0.5 * logProb;
  }
  else if (x_i.rows()==6)  //3D data full pos and dir
  {
    int dim = 4;
    Matrix<T,Dynamic,1> x_i_new(4);
    x_i_new.setZero();
    x_i_new(seq(0, 2)) = x_i(seq(0, 2));
    Matrix<T,Dynamic,1> x_i_dir(3);
    x_i_dir << x_i[3] , x_i[4], x_i[5];
    x_i_new(3) = (rie_log(meanDir_, x_i_dir)).norm();
  
    LLT<Matrix<T,Dynamic,Dynamic>> lltObj(cov_);
    T logProb =  dim * log(2*PI);
    logProb += 2 * log(lltObj.matrixL().determinant());
    logProb += (lltObj.matrixL().solve(x_i_new-mean_)).squaredNorm();
    return -0.5 * logProb;
  }
  else return 0;
  // return -0.5 * logProb;
};


template class NormalDir<double>;