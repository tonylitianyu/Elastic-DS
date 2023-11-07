#pragma once

#include <boost/random/mersenne_twister.hpp>
#include <Eigen/Dense>
#include "normal.hpp"

using namespace Eigen;
using namespace std;


template <class dist_t>
class DPMM
{
public:
  DPMM(const MatrixXd& x, int init_cluster, double alpha, const dist_t& H, const boost::mt19937& rndGen);
  DPMM(const MatrixXd& x, const VectorXi& z, const vector<int> indexList, const double alpha, const dist_t& H, boost::mt19937& rndGen);
  ~DPMM(){};

  void splitProposal();
  // void mergeProposal();
  void sampleCoefficients();
  void sampleCoefficients(const uint32_t index_i, const uint32_t index_j);
  void sampleParameters();
  void sampleParameters(const uint32_t index_i, const uint32_t index_j);
  Normal<double> sampleParameters(vector<int> indexList);
  void sampleCoefficientsParameters();
  void sampleCoefficientsParameters(const uint32_t index_i, const uint32_t index_j);
  void sampleLabels();
  void sampleLabels(const uint32_t index_i, const uint32_t index_j);

  double transitionProb(const uint32_t index_i, const uint32_t index_j);
  double transitionProb(const uint32_t index_i, const uint32_t index_j, VectorXi z_original);
  double posteriorRatio(vector<int> indexList_i, vector<int> indexList_j, vector<int> indexList_ij);
  void reorderAssignments();
  void updateIndexLists();
  vector<vector<int>> getIndexLists();
  const VectorXi & getLabels(){return z_;};

  int splitProposal(vector<int> indexList);
  void sampleSplit(uint32_t z_i, uint32_t z_j);
  int mergeProposal(vector<int> indexList_i, vector<int> indexList_j);

public:
  //class constructor(indepedent of data)
  double alpha_; 
  dist_t H_; 
  boost::mt19937 rndGen_;

  //class initializer(dependent on data)
  MatrixXd x_;
  VectorXi z_;  //membership vector
  VectorXd Pi_; //coefficient vector
  VectorXi index_; //index vector
  uint16_t N_;
  uint16_t K_;

  //sampled parameters
  vector<dist_t> components_; // NIW
  vector<Normal<double>> parameters_; //Normal

  //spilt/merge proposal
  vector<int> indexList_;
  vector<vector<int>> indexLists_;
};