#include <iostream>
#include <limits>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include "dpmmDir.hpp"
#include "niwDir.hpp"
#include "niw.hpp"


template <class dist_t> 
DPMMDIR<dist_t>::DPMMDIR(const MatrixXd& x, int init_cluster, double alpha, const dist_t& H, const boost::mt19937 &rndGen)
: alpha_(alpha), H_(H), rndGen_(rndGen), x_(x), N_(x.rows())
{
  VectorXi z(x.rows());
  if (init_cluster == 1) 
  {
    z.setZero();
  }
  else if (init_cluster > 1)
  {
    boost::random::uniform_int_distribution<> uni_(0, init_cluster-1);
    for (int i=0; i<N_; ++i)z[i] = uni_(rndGen_); 
  }
  else
  { 
    cout<< "Number of initial clusters not supported yet" << endl;
    exit(1);
  }
  z_ = z;
  K_ = z_.maxCoeff() + 1; // equivalent to the number of initial clusters
};



template <class dist_t> 
DPMMDIR<dist_t>::DPMMDIR(const MatrixXd& x, const VectorXi& z, const vector<int> indexList, const double alpha, const dist_t& H, boost::mt19937 &rndGen)
: alpha_(alpha), H_(H), rndGen_(rndGen), x_(x), N_(x.rows()), z_(z), K_(z.maxCoeff() + 1), indexList_(indexList)
{
  vector<int> indexList_i;
  vector<int> indexList_j;
  int z_split_i = z_.maxCoeff() + 1;
  int z_split_j = z_[indexList[0]];

  boost::random::uniform_int_distribution<> uni_01(0, 1);
  for (int i = 0; i<indexList_.size(); ++i)
  {
    if (uni_01(rndGen_) == 0)
      {
        indexList_i.push_back(indexList_[i]);
        z_[indexList_[i]] = z_split_i;
      }
    else 
      {
        indexList_j.push_back(indexList_[i]);
        z_[indexList_[i]] = z_split_j;
      }
  }
  indexLists_.push_back(indexList_i);
  indexLists_.push_back(indexList_j);
};


template <class dist_t> 
void DPMMDIR<dist_t>::sampleCoefficients()
{
  VectorXi Nk(K_);
  Nk.setZero();
  for(uint32_t ii=0; ii<N_; ++ii)
  {
    Nk(z_(ii))++;
  }

  VectorXd Pi(K_);
  for (uint32_t k=0; k<K_; ++k)
  {
    assert(Nk(k)!=0);
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();
}


template <class dist_t> 
void DPMMDIR<dist_t>::sampleParameters()
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

    components_.push_back(H_.posterior(x_k));  //components are NIW
    parameters_.push_back(components_[k].sampleParameter()); //parameters are Normal
  }
}


template <class dist_t> 
void DPMMDIR<dist_t>::sampleCoefficientsParameters()
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
void DPMMDIR<dist_t>::sampleLabels()
{
  double logLik = 0;
  #pragma omp parallel for num_threads(4) schedule(static) private(rndGen_)
  for(uint32_t i=0; i<N_; ++i)
  {
    VectorXd x_i;
    x_i = x_(i, all); //current data point x_i
    VectorXd prob(K_);
    double logLik_i = 0;
    for (uint32_t k=0; k<K_; ++k)
    { 
      double logProb =  parameters_[k].logProb(x_i);
      prob[k] = log(Pi_[k]) + logProb;
      logLik_i += Pi_[k] * exp(logProb);
    }
    logLik += log(logLik_i);
    // std::cout << logLik << std::endl;

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
  // std::cout << logLik << std::endl;
  logLogLik_.push_back(logLik);
}


template <class dist_t>
void DPMMDIR<dist_t>::reorderAssignments()
{ 
  vector<uint8_t> rearrange_list;
  for (uint32_t i=0; i<N_; ++i)
  {
    if (rearrange_list.empty()) rearrange_list.push_back(z_[i]);
    vector<uint8_t>::iterator it;
    it = find (rearrange_list.begin(), rearrange_list.end(), z_[i]);
    if (it == rearrange_list.end())
    {
      rearrange_list.push_back(z_[i]);
      z_[i] = rearrange_list.size() - 1;
    }
    else if (it != rearrange_list.end())
    {
      int index = it - rearrange_list.begin();
      z_[i] = index;
    }
  }
  K_ = z_.maxCoeff() + 1;
  logNum_.push_back(K_);
}


template <class dist_t>
vector<vector<int>> DPMMDIR<dist_t>::getIndexLists()
{
  this ->updateIndexLists();
  return indexLists_;
}

template <class dist_t>
void DPMMDIR<dist_t>::updateIndexLists()
{
  assert(z_.maxCoeff()+1 == K_);
  vector<vector<int>> indexLists(K_);
  for (uint32_t ii = 0; ii<N_; ++ii)
  {
    indexLists[z_[ii]].push_back(ii); 
  }
  indexLists_ = indexLists;
}


template <class dist_t> 
int DPMMDIR<dist_t>::splitProposal(vector<int> indexList)
{
  VectorXi z_launch = z_;
  VectorXi z_split = z_;
  uint32_t z_split_i = z_split.maxCoeff() + 1;
  uint32_t z_split_j = z_split[indexList[0]];


  NIW<double> dist = H_.getNIW();
  // NIWDIR<double> dist = H_;
  
  DPMMDIR<NIW<double>> dpmm_split(x_, z_launch, indexList, alpha_, dist, rndGen_);
  for (int tt=0; tt<50; ++tt)
  {
    if (dpmm_split.indexLists_[0].size()==1 || dpmm_split.indexLists_[1].size() ==1 || dpmm_split.indexLists_[0].empty()==true || dpmm_split.indexLists_[1].empty()==true)
    {
      // std::cout << "Component " << z_split_j <<": Split proposal Rejected" << std::endl;
      return 1;
    }
    // dpmm_split.sampleCoefficientsParameters(indexList);
    // dpmm_split.sampleLabels(indexList);
    // std::cout << "H" << std::endl;
    dpmm_split.sampleLabelsCollapsed(indexList);
  }

  vector<int> indexList_i = dpmm_split.indexLists_[0];
  vector<int> indexList_j = dpmm_split.indexLists_[1];


  
  double logAcceptanceRatio = 0;
  // logAcceptanceRatio -= dpmm_split.logTransitionProb(indexList_i, indexList_j);
  // logAcceptanceRatio += dpmm_split.logPosteriorProb(indexList_i, indexList_j);;
  // if (logAcceptanceRatio < 0) 
  // {
  //   std::cout << "Component " << z_split_j <<": Split proposal Rejected with Log Acceptance Ratio " << logAcceptanceRatio << std::endl;
  //   return 1;
  // }
  
  for (int i = 0; i < indexList_i.size(); ++i)
  {
    z_split[indexList_i[i]] = z_split_i;
  }
  for (int i = 0; i < indexList_j.size(); ++i)
  {
    z_split[indexList_j[i]] = z_split_j;
  }

  z_ = z_split;
  // z_ = dpmm_split.z_;
  K_ += 1;
  logNum_.push_back(K_);
  // this -> updateIndexLists();
  std::cout << "Component " << z_split_j <<": Split proposal Aceepted with Log Acceptance Ratio " << logAcceptanceRatio << std::endl;
  
  return 0;
}


template <class dist_t> 
int DPMMDIR<dist_t>::mergeProposal(vector<int> indexList_i, vector<int> indexList_j)
{
  VectorXi z_launch = z_;
  VectorXi z_merge = z_;
  uint32_t z_merge_i = z_merge[indexList_i[0]];
  uint32_t z_merge_j = z_merge[indexList_j[0]];

  vector<int> indexList;
  indexList.reserve(indexList_i.size() + indexList_j.size() ); // preallocate memory
  indexList.insert( indexList.end(), indexList_i.begin(), indexList_i.end() );
  indexList.insert( indexList.end(), indexList_j.begin(), indexList_j.end() );


  DPMMDIR<dist_t> dpmm_merge(x_, z_launch, indexList, alpha_, H_, rndGen_);
  for (int tt=0; tt<100; ++tt)
  {    
    if (dpmm_merge.indexLists_[0].size()==0 || dpmm_merge.indexLists_[1].size() ==0)
    {
      double logAcceptanceRatio = 0;
      logAcceptanceRatio += log(dpmm_merge.transitionProb(indexList_i, indexList_j));
      logAcceptanceRatio -= dpmm_merge.logPosteriorProb(indexList_i, indexList_j);;

      std::cout << logAcceptanceRatio << std::endl;
      for (int i = 0; i < indexList_i.size(); ++i) z_merge[indexList_i[i]] = z_merge_j;
      z_ = z_merge;
      this -> reorderAssignments();
      std::cout << "Component " << z_merge_j << "and" << z_merge_i <<": Merge proposal Aceepted" << std::endl;
      return 0;
    };
    dpmm_merge.sampleCoefficientsParameters(indexList);
    dpmm_merge.sampleLabels(indexList);
  }

  double logAcceptanceRatio = 0;
  logAcceptanceRatio += log(dpmm_merge.transitionProb(indexList_i, indexList_j));
  logAcceptanceRatio -= dpmm_merge.logPosteriorProb(indexList_i, indexList_j);;

  std::cout << logAcceptanceRatio << std::endl;
  if (logAcceptanceRatio > 0)
  {
    for (int i = 0; i < indexList_i.size(); ++i) z_merge[indexList_i[i]] = z_merge_j;
    z_ = z_merge;
    this -> reorderAssignments();
    std::cout << "Component " << z_merge_j << "and" << z_merge_i <<": Merge proposal Aceepted" << std::endl;
    return 0;
  }
  std::cout << "Component " << z_merge_j << "and" << z_merge_i <<": Merge proposal Rejected" << std::endl;
  return 1;
}



template <class dist_t> 
void DPMMDIR<dist_t>::sampleCoefficientsParameters(vector<int> indexList)
{
  vector<int> indexList_i = indexLists_[0];
  vector<int> indexList_j = indexLists_[1];

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


  VectorXd Pi(2);
  for (uint32_t k=0; k<2; ++k)
  {
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();
}


template <class dist_t> 
void DPMMDIR<dist_t>::sampleLabels(vector<int> indexList)
{
  vector<int> indexList_i;
  vector<int> indexList_j;

  boost::random::uniform_01<> uni_;    
  // #pragma omp parallel for num_threads(4) schedule(static) private(rndGen_)
  for(uint32_t i=0; i<indexList.size(); ++i)
  {
    VectorXd x_i;
    x_i = x_(indexList[i], seq(0,1)); //current data point x_i from the index_list
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
    if (uni_draw < prob[0]) indexList_i.push_back(indexList_[i]);
    else indexList_j.push_back(indexList_[i]);
  }


  indexLists_.clear();
  indexLists_.push_back(indexList_i);
  indexLists_.push_back(indexList_j);
}


template <class dist_t> 
void DPMMDIR<dist_t>::sampleLabelsCollapsed(vector<int> indexList)
{
  int dimPos;
  if (x_.cols()==4) dimPos=1;
  else if (x_.cols()==6) dimPos=2;
  int index_i = z_[indexLists_[0][0]];
  int index_j = z_[indexLists_[1][0]];


  boost::random::uniform_01<> uni_;
  vector<int> indexList_i;
  vector<int> indexList_j;

  // #pragma omp parallel for num_threads(4) schedule(static) private(rndGen_)
  for(int i=0; i<indexList.size(); ++i)
  {
    VectorXd x_i;
    x_i = x_(indexList[i], seq(0,dimPos)); //current data point x_i from the index_list
    VectorXd prob(2);

    for (int ii=0; ii < indexList.size(); ++ii)
    {
      if (z_[indexList[ii]] == index_i && ii!=i) indexList_i.push_back(indexList[ii]);
      else if (z_[indexList[ii]] == index_j && ii!=i) indexList_j.push_back(indexList[ii]);
    }

    if (indexList_i.empty()==true || indexList_j.empty()==true)
    {
      indexLists_.clear();
      indexLists_.push_back(indexList_i);
      indexLists_.push_back(indexList_j);
      return;
    } 

    prob[0] = log(indexList_i.size()) + H_.logPosteriorProb(x_i, x_(indexList_i, seq(0,dimPos))); 
    prob[1] = log(indexList_j.size()) + H_.logPosteriorProb(x_i, x_(indexList_j, seq(0,dimPos))); 

    double prob_max = prob.maxCoeff();
    prob = (prob.array()-(prob_max + log((prob.array() - prob_max).exp().sum()))).exp().matrix();
    prob = prob / prob.sum();
    for (uint32_t ii = 1; ii < prob.size(); ++ii)
    {
      prob[ii] = prob[ii-1]+ prob[ii];
    }
    double uni_draw = uni_(rndGen_);
    if (uni_draw < prob[0]) z_[indexList[i]] = index_i;
    else z_[indexList[i]] = index_j;
    
    indexList_i.clear();
    indexList_j.clear();
  }


  for (int i=0; i < indexList.size(); ++i)
  {
    if (z_[indexList[i]] == index_i) indexList_i.push_back(indexList[i]);
    else if (z_[indexList[i]] == index_j)indexList_j.push_back(indexList[i]);
  }
  indexLists_.clear();
  indexLists_.push_back(indexList_i);
  indexLists_.push_back(indexList_j);
}


template <class dist_t> 
double DPMMDIR<dist_t>::transitionProb(vector<int> indexList_i, vector<int> indexList_j)
{
  double transitionProb = 1;

  for (uint32_t ii=0; ii < indexList_i.size(); ++ii)
  {
    transitionProb *= Pi_(0) * components_[0].prob(x_(indexList_i[ii], all))/
    (Pi_(0) * components_[0].prob(x_(indexList_i[ii], all)) + Pi_(1) *  components_[1].prob(x_(indexList_i[ii], all)));
  }

  for (uint32_t ii=0; ii < indexList_j.size(); ++ii)
  {
    transitionProb *= Pi_(0) * components_[0].prob(x_(indexList_j[ii], all))/
    (Pi_(0) * components_[0].prob(x_(indexList_j[ii], all)) + Pi_(1) *  components_[1].prob(x_(indexList_j[ii], all)));
  }
  
  // std::cout << transitionProb << std::endl;

  return transitionProb;
}


template <class dist_t> 
double DPMMDIR<dist_t>::logTransitionProb(vector<int> indexList_i, vector<int> indexList_j)
{
  double logTransitionProb = 0;

  for (uint32_t ii=0; ii < indexList_i.size(); ++ii)
  {
    logTransitionProb += log(Pi_(0) * components_[0].prob(x_(indexList_i[ii], all))) -
    log(Pi_(0) * components_[0].prob(x_(indexList_i[ii], all)) + Pi_(1) *  components_[1].prob(x_(indexList_i[ii], all)));
  }

  for (uint32_t ii=0; ii < indexList_j.size(); ++ii)
  {
    logTransitionProb += log(Pi_(0) * components_[0].prob(x_(indexList_j[ii], all))) -
    log(Pi_(0) * components_[0].prob(x_(indexList_j[ii], all)) + Pi_(1) *  components_[1].prob(x_(indexList_j[ii], all)));
  }
  
  // std::cout << transitionProb << std::endl;

  return logTransitionProb;
}


template <class dist_t>
double DPMMDIR<dist_t>::logPosteriorProb(vector<int> indexList_i, vector<int> indexList_j)
{
  vector<int> indexList;
  indexList.reserve(indexList_i.size() + indexList_j.size() ); // preallocate memory
  indexList.insert( indexList.end(), indexList_i.begin(), indexList_i.end() );
  indexList.insert( indexList.end(), indexList_j.begin(), indexList_j.end() );

  NormalDir<double> parameter_ij = H_.posterior(x_(indexList, all)).sampleParameter();
  NormalDir<double> parameter_i  = H_.posterior(x_(indexList_i, all)).sampleParameter();
  NormalDir<double> parameter_j  = H_.posterior(x_(indexList_j, all)).sampleParameter();

  double logPosteriorRatio = 0;
  for (uint32_t ii=0; ii < indexList_i.size(); ++ii)
  {
    logPosteriorRatio += log(indexList_i.size()) + parameter_i.logProb(x_(indexList_i[ii], all)) ;
    logPosteriorRatio -= log(indexList.size()) - parameter_ij.logProb(x_(indexList_i[ii], all));
  }
  for (uint32_t jj=0; jj < indexList_j.size(); ++jj)
  {
    logPosteriorRatio += log(indexList_j.size()) + parameter_j.logProb(x_(indexList_j[jj], all)) ;
    logPosteriorRatio -= log(indexList.size()) - parameter_ij.logProb(x_(indexList_j[jj], all));
  }

  return logPosteriorRatio;
}



template class DPMMDIR<NIWDIR<double>>;

