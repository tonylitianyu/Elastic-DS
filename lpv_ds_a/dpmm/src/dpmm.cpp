#include <iostream>
#include <limits>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include "dpmm.hpp"
#include "niw.hpp"
#include "niwDir.hpp"


template <class dist_t> 
DPMM<dist_t>::DPMM(const MatrixXd& x, int init_cluster, double alpha, const dist_t& H, const boost::mt19937 &rndGen)
: alpha_(alpha), H_(H), rndGen_(rndGen), x_(x), N_(x.rows())
{
  VectorXi z(x.rows());
  if (init_cluster == 1) 
  {
    z.setZero();
  }
  else if (init_cluster >= 1)
  {
    boost::random::uniform_int_distribution<> uni_(0, init_cluster-1);
    // #pragma omp parallel for num_threads(8) schedule(static)
    for (int i=0; i<N_; ++i)z[i] = uni_(rndGen_); 
  }
  else
  { 
    cout<< "Number of initial clusters not supported yet" << endl;
    exit(1);
  }
  z_ = z;

  K_ = z_.maxCoeff() + 1; // equivalent to the number of initial clusters
  // indexList_ = 
  // this -> sampleCoefficients(); //Pi_
  // this -> sampleParameters();  //parameters_; components_
};


template <class dist_t> 
DPMM<dist_t>::DPMM(const MatrixXd& x, const VectorXi& z, const vector<int> indexList, const double alpha, const dist_t& H, boost::mt19937 &rndGen)
: alpha_(alpha), H_(H), rndGen_(rndGen), x_(x), N_(x.rows()), z_(z), K_(z.maxCoeff() + 1), indexList_(indexList)
{};


template <class dist_t> 
int DPMM<dist_t>::mergeProposal(vector<int> indexList_i, vector<int> indexList_j)
{ 
  VectorXi z_split = z_; //original split state
  
  boost::random::uniform_int_distribution<> uni_i(0, indexList_i.size()-1);
  boost::random::uniform_int_distribution<> uni_j(0, indexList_j.size()-1);
  uint32_t index_i = indexList_i[uni_i(rndGen_)];
  uint32_t index_j = indexList_j[uni_j(rndGen_)];
  assert(index_i!=index_j);

  VectorXi z_launch = z_; 
  uint32_t z_split_i = z_[index_i];
  uint32_t z_split_j = z_[index_j];

  vector<int> indexList;
  indexList.reserve( indexList_i.size() + indexList_j.size() ); // preallocate memory
  indexList.insert( indexList.end(), indexList_i.begin(), indexList_i.end() );
  indexList.insert( indexList.end(), indexList_j.begin(), indexList_j.end() );
  assert(indexList.size() == indexList_i.size() + indexList_j.size());

  boost::random::uniform_int_distribution<> uni_01(0, 1);
  for (uint32_t ii = 0; ii<indexList.size(); ++ii)
  {
    if (uni_01(rndGen_) == 0) z_launch[indexList[ii]] = z_split_i;
    else z_launch[indexList[ii]] = z_split_j;
  }
  z_launch[index_i] = z_split_i;
  z_launch[index_j] = z_split_j;


  DPMM<dist_t> dpmm_merge(x_, z_launch, indexList, alpha_, H_, rndGen_);
  for (uint32_t t=0; t<50; ++t)
  {
    // std::cout << t << std::endl;
    dpmm_merge.sampleCoefficients(index_i, index_j);
    dpmm_merge.sampleParameters(index_i, index_j); 
    // dpmm_split.sampleCoefficientsParameters(index_i, index_j);
    dpmm_merge.sampleLabels(index_i, index_j);  
  }

  indexList_i = dpmm_merge.getIndexLists()[z_split_i];
  indexList_j = dpmm_merge.getIndexLists()[z_split_j];

  double transitionRatio = dpmm_merge.transitionProb(index_i, index_j, z_);
  double posteriorRatio = 1.0 / dpmm_merge.posteriorRatio(indexList_i, indexList_j, indexList);
  double acceptanceRatio = transitionRatio * posteriorRatio;

  if (acceptanceRatio >= 1) 
  {
    std::cout << "Component " << z_split_i << " and " << z_split_j << ": Merge proposal Aceepted" << std::endl; 
    z_ = dpmm_merge.z_;
    K_ -= 1;
    this ->updateIndexLists();
    this -> sampleCoefficients();
    this -> sampleParameters();
    std::cout << "Component " << z_split_i << " and " << z_split_j << ": Merge proposal Aceepted" << std::endl;
    // std::cout << Pi_ << std::endl;
    // std::cout << parameters_.size() << std::endl;
    return 0;
  }
  else
  {
    std::cout  << "Component " << z_split_i << " and " << z_split_j << ": Merge proposal Rejected" << std::endl;
    return 1;
  }
}


template <class dist_t> 
int DPMM<dist_t>::splitProposal(vector<int> indexList)
{ 
  // /*
  boost::random::uniform_int_distribution<> uni_(0, indexList.size()-1);
  uint32_t index_i = indexList[uni_(rndGen_)];
  uint32_t index_j = indexList[uni_(rndGen_)];
  while (index_i == index_j)
  {
    index_i = indexList[uni_(rndGen_)];
    index_j = indexList[uni_(rndGen_)];
  }
  assert(index_i!=index_j);
  

  VectorXi z_launch = z_; //original assignment vector
  uint32_t z_split_i = z_launch.maxCoeff() + 1;
  uint32_t z_split_j = z_launch[index_j];


  boost::random::uniform_int_distribution<> uni_01(0, 1);
  for (uint32_t ii = 0; ii<indexList.size(); ++ii)
  {
    if (uni_01(rndGen_) == 0) z_launch[indexList[ii]] = z_split_i;
    else z_launch[indexList[ii]] = z_split_j;
  }


  z_launch[index_i] = z_split_i;
  z_launch[index_j] = z_split_j;
  // */
  

  /*
  VectorXi z_launch = z_; //original assignment vector including all xs
  uint32_t z_split_i = z_launch.maxCoeff() + 1;
  uint32_t z_split_j = z_launch[indexList[0]];

  boost::random::uniform_int_distribution<> uni_01(0, 1);
  for (uint32_t ii = 0; ii<indexList.size(); ++ii)
  {
    if (uni_01(rndGen_) == 0) z_launch[indexList[ii]] = z_split_i;
    else z_launch[indexList[ii]] = z_split_j;
  }
  */
  
  DPMM<dist_t> dpmm_split(x_, z_launch, indexList, alpha_, H_, rndGen_);

  for (uint32_t t=0; t<500; ++t)
  {
    // std::cout << t << std::endl;
    // dpmm_split.sampleCoefficients(index_i, index_j);
    // dpmm_split.sampleParameters(index_i, index_j); 
    dpmm_split.sampleCoefficientsParameters(index_i, index_j);
    dpmm_split.sampleLabels(index_i, index_j);  
    // dpmm_split.sampleSplit(z_split_i, z_split_j);  
  }
  
  vector<int> indexList_i = dpmm_split.getIndexLists()[z_split_i];
  vector<int> indexList_j = dpmm_split.getIndexLists()[z_split_j];

  double transitionRatio = 1.0 / dpmm_split.transitionProb(index_i, index_j);
  double posteriorRatio = dpmm_split.posteriorRatio(indexList_i, indexList_j, indexList);
  double acceptanceRatio = transitionRatio * posteriorRatio;
  

  if (acceptanceRatio >= 1) 
  {
    z_ = dpmm_split.z_;
    K_ += 1;
    this -> updateIndexLists();
    this -> sampleCoefficients();
    this -> sampleParameters();
    std::cout << "Component " << z_split_j <<": Split proposal Aceepted" << std::endl;
    // std::cout << Pi_ << std::endl;
    // std::cout << parameters_.size() << std::endl;
    return 0;
  }
  else
  {
    std::cout << "Component " << z_split_j <<": Split proposal Rejected" << std::endl;
    return 1;
  }
}


template <class dist_t> 
double DPMM<dist_t>::transitionProb(const uint32_t index_i, const uint32_t index_j)
{
  assert(!components_.empty());
  double transitionProb = 1;
  for (uint32_t ii=0; ii < indexList_.size(); ++ii)
  {
    if (z_[indexList_[ii]] == z_[index_i])
    transitionProb *= Pi_(0) * components_[0].prob(x_(indexList_[ii], all))/
    (Pi_(0) * components_[0].prob(x_(indexList_[ii], all)) + Pi_(1) *  components_[1].prob(x_(indexList_[ii], all)));
    else
    transitionProb *= Pi_(1) * components_[1].prob(x_(indexList_[ii], all))/
    (Pi_(0) * components_[0].prob(x_(indexList_[ii], all)) + Pi_(1) *  components_[1].prob(x_(indexList_[ii], all)));
  }
  return transitionProb;
}

template <class dist_t> 
double DPMM<dist_t>::transitionProb(const uint32_t index_i, const uint32_t index_j,VectorXi z_original)
{
  z_ = z_original;
  return this->transitionProb(index_i, index_j);
}


template <class dist_t>
double DPMM<dist_t>::posteriorRatio(vector<int> indexList_i, vector<int> indexList_j, vector<int> indexList_ij)
{
  Normal<double> parameter_ij = H_.posterior(x_(indexList_ij, all)).sampleParameter();
  Normal<double> parameter_i  = H_.posterior(x_(indexList_i, all)).sampleParameter();
  Normal<double> parameter_j  = H_.posterior(x_(indexList_j, all)).sampleParameter();

  double logPosteriorRatio = 0;
  for (uint32_t ii=0; ii < indexList_i.size(); ++ii)
  {
    logPosteriorRatio += log(indexList_i.size()) + parameter_i.logProb(x_(indexList_i[ii], all)) ;
    logPosteriorRatio -= parameter_ij.logProb(x_(indexList_i[ii], all));
  }
  for (uint32_t jj=0; jj < indexList_j.size(); ++jj)
  {
    logPosteriorRatio += log(indexList_j.size()) + parameter_j.logProb(x_(indexList_j[jj], all)) ;
    logPosteriorRatio -= parameter_ij.logProb(x_(indexList_j[jj], all));
  }

  return exp(logPosteriorRatio);
}


template <class dist_t> 
void DPMM<dist_t>::sampleCoefficients()
{
  VectorXi Nk(K_);
  Nk.setZero();
  for(uint32_t ii=0; ii<N_; ++ii)
  {
    Nk(z_(ii))++;
  }
  // Nk(K_) = alpha_;
  // std::cout << Nk << std::endl;

  VectorXd Pi(K_);
  for (uint32_t k=0; k<K_; ++k)
  {
    assert(Nk(k)!=0);
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();
  // std::cout << Nk << std::endl;
}

// template <class dist_t> 
// void DPMM<dist_t>::sampleSplit()
// {






// }

template <class dist_t> 
void DPMM<dist_t>::sampleCoefficients(const uint32_t index_i, const uint32_t index_j)
{
  VectorXi Nk(2);
  Nk.setZero();

  for(uint32_t ii=0; ii<indexList_.size(); ++ii)
  {
    if (z_[indexList_[ii]]==z_[index_i]) Nk(0)++;
    else Nk(1)++;
  }

  //`````testing``````````````
  // std::cout << Nk <<std::endl;
  //`````testing``````````````


  VectorXd Pi(2);
  for (uint32_t k=0; k<2; ++k)
  {
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();
  // std::cout << Pi_ <<std::endl;
}


template <class dist_t> 
void DPMM<dist_t>::sampleCoefficientsParameters()
{ 
  components_.clear();
  parameters_.clear();
  VectorXd Pi(K_);

  vector<vector<int>> indexLists(K_);
  for (uint32_t ii = 0; ii<N_; ++ii)
  {
    indexLists[z_[ii]].push_back(ii); 
  }
  
  for (uint32_t k=0; k<K_; ++k)
  {
    boost::random::gamma_distribution<> gamma_(indexLists[k].size(), 1);
    Pi(k) = gamma_(rndGen_);
    components_.push_back(H_.posterior(x_(indexLists[k], all)));
    parameters_.push_back(components_[k].sampleParameter());
  }
  Pi_ = Pi / Pi.sum();
}


template <class dist_t> 
void DPMM<dist_t>::sampleParameters()
{ 
  components_.clear();
  parameters_.clear();

  for (uint32_t k=0; k<K_; ++k)
  {
    vector<int> indexList_k;
    for (uint32_t ii = 0; ii<N_; ++ii)
    {
      if (z_[ii] == k) indexList_k.push_back(ii); 
    }
    MatrixXd x_k(indexList_k.size(), x_.cols()); 
    x_k = x_(indexList_k, all);

    components_.push_back(H_.posterior(x_k));
    parameters_.push_back(components_[k].sampleParameter());
  }
}


template <class dist_t> 
Normal<double> DPMM<dist_t>::sampleParameters(vector<int> indexList)
{ 
  return H_.posterior(x_(indexList, all)).sampleParameter();
}


template <class dist_t> 
void DPMM<dist_t>::sampleParameters(const uint32_t index_i, const uint32_t index_j)
{
  components_.clear();
  parameters_.clear();

  vector<int> indexList_i;
  vector<int> indexList_j;

  for (uint32_t ii = 0; ii<indexList_.size(); ++ii)  //To-do: This can be combined with sampleCoefficients
  {
    if (z_[indexList_[ii]]==z_[index_i]) 
    indexList_i.push_back(indexList_[ii]); 
    else if (z_[indexList_[ii]]==z_[index_j])
    indexList_j.push_back(indexList_[ii]);
  }
  assert(indexList_i.size() + indexList_j.size() == indexList_.size());
  MatrixXd x_i(indexList_i.size(), x_.cols()); 
  MatrixXd x_j(indexList_j.size(), x_.cols()); 


  //`````testing``````````````
  // std::cout << indexList_i.size() << std::endl << indexList_j.size() << std::endl;
  //`````testing``````````````


  x_i = x_(indexList_i, all);
  x_j = x_(indexList_j, all);

  components_.push_back(H_.posterior(x_i));
  components_.push_back(H_.posterior(x_j));
  parameters_.push_back(components_[0].sampleParameter());
  parameters_.push_back(components_[1].sampleParameter());
}




template <class dist_t> 
void DPMM<dist_t>::sampleSplit(uint32_t z_i, uint32_t z_j)
{
  vector<int> indexList_i;
  vector<int> indexList_j;

  // #pragma omp parallel for num_threads(8) schedule(dynamic,100)
  for (uint32_t ii = 0; ii<indexList_.size(); ++ii) 
  {
    if (z_[indexList_[ii]]==z_i) 
    indexList_i.push_back(indexList_[ii]); 
    else if (z_[indexList_[ii]]==z_j)
    indexList_j.push_back(indexList_[ii]);
  }
  assert(indexList_i.size() + indexList_j.size() == indexList_.size());

  MatrixXd x_i(indexList_i.size(), x_.cols()); 
  MatrixXd x_j(indexList_j.size(), x_.cols()); 
  x_i = x_(indexList_i, all);
  x_j = x_(indexList_j, all);
  
  components_.clear();
  parameters_.clear();
  components_.push_back(H_.posterior(x_i));
  components_.push_back(H_.posterior(x_j));
  parameters_.push_back(components_[0].sampleParameter());
  parameters_.push_back(components_[1].sampleParameter());
  

  VectorXi Nk(2);
  Nk(0) = indexList_i.size();
  Nk(1) = indexList_j.size();


  // //`````testing``````````````
  // std::cout << Nk <<std::endl;
  // //`````testing``````````````


  VectorXd Pi(2);
  for (uint32_t k=0; k<2; ++k)
  {
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();


  // //`````testing``````````````
  // std::cout << Pi_ <<std::endl;  
  // //`````testing``````````````

  boost::random::uniform_01<> uni_;    //maybe put in constructor?
  #pragma omp parallel for num_threads(4) schedule(static) private(rndGen_)
  for(uint32_t i=0; i<indexList_.size(); ++i)
  {
    VectorXd x_i;
    x_i = x_(indexList_[i], all); //current data point x_i from the index_list
    VectorXd prob(2);
    for (uint32_t k=0; k<2; ++k)
    {
      prob[k] = log(Pi_[k]) + parameters_[k].logProb(x_i); //first component is always the set of x_i (different notion from x_i here)
    }

    double prob_max = prob.maxCoeff();
    prob = (prob.array()-(prob_max + log((prob.array() - prob_max).exp().sum()))).exp().matrix();
    prob = prob / prob.sum();
    for (uint32_t ii = 1; ii < prob.size(); ++ii){
      prob[ii] = prob[ii-1]+ prob[ii];
    }
    double uni_draw = uni_(rndGen_);
    if (uni_draw < prob[0]) z_[indexList_[i]] = z_i;
    else z_[indexList_[i]] = z_j;
  }
}




template <class dist_t> 
void DPMM<dist_t>::sampleCoefficientsParameters(uint32_t index_i, uint32_t index_j)
{
  vector<int> indexList_i;
  vector<int> indexList_j;

  // std::cout << index_i <<std::endl << index_j << std::endl;
  assert(z_[index_i] !=  z_[index_j]);
  // #pragma omp parallel for num_threads(8) schedule(dynamic,100)
  for (uint32_t ii = 0; ii<indexList_.size(); ++ii) 
  {
    if (z_[indexList_[ii]]==z_[index_i]) 
    indexList_i.push_back(indexList_[ii]); 
    else if (z_[indexList_[ii]]==z_[index_j])
    indexList_j.push_back(indexList_[ii]);
  }
  assert(indexList_i.size() + indexList_j.size() == indexList_.size());

  MatrixXd x_i(indexList_i.size(), x_.cols()); 
  MatrixXd x_j(indexList_j.size(), x_.cols()); 
  x_i = x_(indexList_i, all);
  x_j = x_(indexList_j, all);
  
  components_.clear();
  parameters_.clear();
  components_.push_back(H_.posterior(x_i));
  components_.push_back(H_.posterior(x_j));
  parameters_.push_back(components_[0].sampleParameter());
  parameters_.push_back(components_[1].sampleParameter());
  

  VectorXi Nk(2);
  Nk(0) = indexList_i.size();
  Nk(1) = indexList_j.size();


  // //`````testing``````````````
  // std::cout << Nk <<std::endl;
  // //`````testing``````````````


  VectorXd Pi(2);
  for (uint32_t k=0; k<2; ++k)
  {
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();


  // //`````testing``````````````
  // std::cout << Pi_ <<std::endl;  
  // //`````testing``````````````
}



template <class dist_t> 
void DPMM<dist_t>::sampleLabels(const uint32_t index_i, const uint32_t index_j)
{
  uint32_t z_i = z_[index_i];
  uint32_t z_j = z_[index_j];
  assert(z_i!=z_j);
  boost::random::uniform_01<> uni_;    //maybe put in constructor?
  #pragma omp parallel for num_threads(4) schedule(static) private(rndGen_)
  for(uint32_t i=0; i<indexList_.size(); ++i)
  {
    VectorXd x_i;
    x_i = x_(indexList_[i], all); //current data point x_i from the index_list
    VectorXd prob(2);
    for (uint32_t k=0; k<2; ++k)
    {
      prob[k] = log(Pi_[k]) + parameters_[k].logProb(x_i); //first component is always the set of x_i (different notion from x_i here)
    }

    double prob_max = prob.maxCoeff();
    prob = (prob.array()-(prob_max + log((prob.array() - prob_max).exp().sum()))).exp().matrix();
    prob = prob / prob.sum();
    for (uint32_t ii = 1; ii < prob.size(); ++ii){
      prob[ii] = prob[ii-1]+ prob[ii];
    }
    double uni_draw = uni_(rndGen_);
    if (uni_draw < prob[0]) z_[indexList_[i]] = z_i;
    else z_[indexList_[i]] = z_j;
  }
  z_[index_i] = z_i;
  z_[index_j] = z_j;
  // //`````testing``````````````
  // std::cout << z_i << std::endl << z_j <<std::endl;
  // //`````testing``````````````
}


template <class dist_t> 
void DPMM<dist_t>::sampleLabels()
{
  #pragma omp parallel for num_threads(4) schedule(static) private(rndGen_)
  for(uint32_t i=0; i<N_; ++i)
  {
    VectorXd x_i;
    x_i = x_(i, all); //current data point x_i
    VectorXd prob(K_);
    for (uint32_t k=0; k<K_; ++k)
    {
      prob[k] = log(Pi_[k]) + parameters_[k].logProb(x_i);
    }

    double prob_max = prob.maxCoeff();
    prob = (prob.array()-(prob_max + log((prob.array() - prob_max).exp().sum()))).exp().matrix();
    prob = prob / prob.sum();
    for (uint32_t ii = 1; ii < prob.size(); ++ii){
      prob[ii] = prob[ii-1]+ prob[ii];
    }
    boost::random::uniform_01<> uni_;   
    double uni_draw = uni_(rndGen_);
    uint32_t k = 0;
    while (prob[k] < uni_draw) k++;
    z_[i] = k;
  }
}



//   vector<int> i_array = generateRandom(N_);
//   for(uint32_t j=0; j<N_; ++j)
//   // for(uint32_t i=0; i<N_; ++i)
//   {
//     int i = i_array[j];
//     // cout << "number of data point: " << i << endl;
//     VectorXi Nk(K_);
//     Nk.setZero();
//     for(uint32_t ii=0; ii<N_; ++ii)
//     {
//       if (ii != i) Nk(z_(ii))++;
//     }
//     // cout<< Nk << endl;
//     VectorXd pi(K_+1); 
//     // VectorXd pi(K_); 

//     VectorXd x_i;
//     x_i = x_(i, all); //current data point x_i

//     // #pragma omp parallel for
//     for (uint32_t k=0; k<K_; ++k)
//     { 
//       vector<int> indexList_k;
//       for (uint32_t ii = 0; ii<N_; ++ii)
//       {
//         if (ii!= i && z_[ii] == k) indexList_k.push_back(ii); 
//       }
//       // if (indexList_k.empty()) 
//       // cout << "no index" << endl;
//       // cout << "im here" <<endl;


//       MatrixXd x_k(indexList_k.size(), x_.cols()); 
//       x_k = x_(indexList_k, all);
//       // cout << "x_i" << x_i << endl;
//       // cout << "x_k" << x_k << endl;
//       // cout << "component:" <<k  <<endl;
//       // cout << x_k << endl;
//       // cout << Nk(k) << endl;
//       if (Nk(k)!=0)
//       pi(k) = log(Nk(k))-log(N_+alpha_) + parameters_[k].logPosteriorProb(x_i, x_k);
//       else
//       pi(k) = - std::numeric_limits<float>::infinity();
//     }
//     pi(K_) = log(alpha_)-log(N_+alpha_) + H_.logProb(x_i);


//     // cout << pi <<endl;
//     // exit(1);


    
    
//     double pi_max = pi.maxCoeff();
//     pi = (pi.array()-(pi_max + log((pi.array() - pi_max).exp().sum()))).exp().matrix();
//     pi = pi / pi.sum();


//     for (uint32_t ii = 1; ii < pi.size(); ++ii){
//       pi[ii] = pi[ii-1]+ pi[ii];
//     }
   
//     boost::random::uniform_01<> uni_;   
//     boost::random::variate_generator<boost::random::mt19937&, 
//                            boost::random::uniform_01<> > var_nor(*H_.rndGen_, uni_);
//     double uni_draw = var_nor();
//     uint32_t k = 0;
//     while (pi[k] < uni_draw) k++;
//     z_[i] = k;
//     this -> reorderAssignments();
//   }

// };


template <class dist_t>
void DPMM<dist_t>::reorderAssignments()
{ 
  // cout << z_ << endl;
  vector<uint8_t> rearrange_list;
  for (uint32_t i=0; i<N_; ++i)
  {
    if (rearrange_list.empty()) rearrange_list.push_back(z_[i]);
    // cout << *rearrange_list.begin() << endl;
    // cout << *rearrange_list.end()  << endl;
    vector<uint8_t>::iterator it;
    it = find (rearrange_list.begin(), rearrange_list.end(), z_[i]);
    if (it == rearrange_list.end())
    {
      rearrange_list.push_back(z_[i]);
      // z_[i] = rearrange_list.end() - rearrange_list.begin();
      z_[i] = rearrange_list.size() - 1;
    }
    else if (it != rearrange_list.end())
    {
      int index = it - rearrange_list.begin();
      z_[i] = index;
    }
  }
  K_ = z_.maxCoeff() + 1;
}


template <class dist_t>
vector<vector<int>> DPMM<dist_t>::getIndexLists()
{
  this ->updateIndexLists();
  return indexLists_;
}

template <class dist_t>
void DPMM<dist_t>::updateIndexLists()
{
  vector<vector<int>> indexLists(K_);
  for (uint32_t ii = 0; ii<N_; ++ii)
  {
    indexLists[z_[ii]].push_back(ii); 
  }
  indexLists_ = indexLists;
}



// template <class dist_t>
// void DPMM<dist_t>::removeEmptyClusters()
// {
//   for(uint32_t k=parameters_.size()-1; k>=0; --k)
//   {
//     bool haveCluster_k = false;
//     for(uint32_t i=0; i<z_.size(); ++i)
//       if(z_(i)==k)
//       {
//         haveCluster_k = true;
//         break;
//       }
//     if (!haveCluster_k)
//     {
//       for (uint32_t i=0; i<z_.size(); ++i)
//         if(z_(i) >= k) z_(i) --;
//       parameters_.erase(parameters_.begin()+k);
//     }
//   }
// }

template class DPMM<NIW<double>>;
// template class DPMM<NIWDIR<double>>;

