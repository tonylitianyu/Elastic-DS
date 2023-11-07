#include <iostream>
// #include <cmath>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#define PI 3.141592653589793


template <typename T>
T unsigned_angle(const Matrix<T,Dynamic, 1>&u, const Matrix<T,Dynamic, 1>&v)
{
    T theta;
    if (u.dot(v) > 1) return 0;
    if (u.dot(v) < -1) return PI;
    theta = acos(u.dot(v));
    // std::cout << theta;
    return theta; 
}


template <typename T>
Matrix<T,Dynamic, 1> rie_log(const Matrix<T,Dynamic, 1>&pp, const Matrix<T,Dynamic, 1>&xx)
{   // Given the coordinate of the reference direction pp and target direction xx w.r.t world orgin
    // Return the coordinate of x_tp w.r.t the tip of pp
    Matrix<T,Dynamic, 1> x_tp;
    T theta = unsigned_angle(pp, xx);
    if (theta < 0.01)
    {   
        x_tp.setZero(pp.rows());
        return x_tp; //p and x are same
    }
    return (xx - pp * cos(theta)) * theta / sin(theta);
}

template <typename T>
Matrix<T,Dynamic, 1> rie_exp(Matrix<T,Dynamic, 1>&pp, const Matrix<T,Dynamic, 1>&xx_tp)
{
    // Given the coordinate of xx_tp w.r.t the tip of pp, and coordinate of pp w.r.t world origin
    // Return the coordinate of x w.r.t world origin
    // T theta;
    T theta = xx_tp.norm();
    if (theta < 0.01)
        return pp;   // p and x are same
    pp =  pp * cos(theta) + xx_tp / theta * sin(theta);
    return pp;
}


template<typename T>
Matrix<T, Dynamic, 1> karcherMean(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  float tolerance = 0.01;
  T angle;
  Matrix<T, Dynamic, 1> angle_sum(x_k.cols()/2);
  Matrix<T, Dynamic, 1> x_tp(x_k.cols()/2);   // x in tangent plane
  Matrix<T, Dynamic, 1> x(x_k.cols()/2);
  Matrix<T, Dynamic, 1> p(x_k.cols()/2);
  
  p = x_k(0, seq(x_k.cols()/2, last)).transpose();

  // std::cout << p << std::endl; 
  if (x_k.rows() == 1) return p;

  while (1)
  { 
    angle_sum.setZero();
    for (int i=0; i<x_k.rows(); ++i)
    {
      x = x_k(i, seq(x_k.cols()/2, last)).transpose();
      angle_sum = angle_sum + rie_log(p, x);
    }
    x_tp = 1. / x_k.rows() * angle_sum;
    // cout << x_tp.norm() << endl;
    if (x_tp.norm() < tolerance) 
    {
      return p;
    }
    p = rie_exp(p, x_tp);
  }
};



template<typename T>
T karcherScatter(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  return karcherScatter(x_k, karcherMean(x_k));
}


template<typename T>
T karcherScatter(const Matrix<T,Dynamic, Dynamic>& x_k, Matrix<T, Dynamic, 1> mean)
{
  T scatter = 0;
  Matrix<T, Dynamic, 1> x_i_dir(x_k.cols()/2);

  for (int i = 0; i < x_k.rows(); ++i)
  {
    x_i_dir = x_k(i, seq(x_k.cols()/2, last)).transpose();
    scatter = scatter + pow(rie_log(mean, x_i_dir).norm(), 2); //squared
  }
  return scatter;
}