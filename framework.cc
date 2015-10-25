#include <cmath>
#include <iostream>
#include "framework.h"

static double Gaussian(double x, double half_inv_sigma2) {
  return exp(-x * x * half_inv_sigma2);
}

static double NormUniformDraw() {
  return double(rand()) / double(RAND_MAX); 
}

Node::Node(const std::string& debug_name) :
  debug_name_(debug_name)
{
  ClearEvidence();
  ClearInitialized();
}

const std::string& Node::GetName() const {
  return debug_name_;
}

double Node::GetValue() const {
  return value_;
}

void Node::SetEvidence() {
  is_evidence_ = true;
}

void Node::ClearEvidence() {
  is_evidence_ = false;
}

bool Node::IsEvidence() const {
  return is_evidence_;
}

void Node::SetValue(double value) {
  value_ = value;
}

bool Node::AllParentsInitialized() const {
  bool all_parents_initialized = true;
  for (Node* parent : GetParents()) {
    if (!parent->IsInitialized()) {
      all_parents_initialized = false;
    }
  }

  return all_parents_initialized;
}


void Node::SetInitialized() {
  is_initialized_ = true;
}

void Node::ClearInitialized() {
  is_initialized_ = false;
}

bool Node::IsInitialized() const {
  return is_initialized_;
}

const std::vector<Node*>& Node::GetChildren() const {
  return children_;
}

const std::vector<Node*>& Node::GetParents() const {
  return parents_;
}

int Node::GetNumParents() const {
  return parents_.size();
}

void Node::AddChild(Node* node) {
  children_.push_back(node);
}

void Node::EdgeFrom(Node* from) {
  parents_.push_back(from);
  from->AddChild(this);
}

ContinuousNode::ContinuousNode(const std::string& debug_name) :
  Node(debug_name)
{}

GaussianNode::GaussianNode(const std::vector<double>& beta, double sigma2, const std::string& debug_name) :
    ContinuousNode(debug_name),
    beta_(beta),
    sigma2_(sigma2),
    half_inv_sigma2_(0.5 / sigma2),
    gaussian_source_(sigma2)
{}

double GaussianNode::GetConditional() const {
  return Gaussian(GetMean() - GetValue(), half_inv_sigma2_);
}

double GaussianNode::GetMean() const {
  double mean = beta_[0]; 
  const std::vector<Node*> parents = GetParents();
  for (int i = 0; i < parents.size(); ++i) {
    mean += beta_[i+1] * parents[i]->GetValue();
  }
  return mean;
}

double GaussianNode::GetSample() {
  return GetMean() + gaussian_source_.Draw();
}

GaussianEvidenceNode::GaussianEvidenceNode(const std::vector<double>& beta, double sigma2, double value, const std::string& debug_name) :
  GaussianNode(beta, sigma2, debug_name)
{
  SetValue(value);
  SetEvidence();
}

double GaussianEvidenceNode::GetSample() {
  return GetValue();
}

EvidenceNode::EvidenceNode(double value) {
  SetValue(value);
  SetEvidence();
}

double EvidenceNode::GetConditional() const {
  return 1.0;
}

double EvidenceNode::GetSample() {
  return GetValue();
}

UniformNode::UniformNode(double from, double to, const std::string& debug_name) :
  ContinuousNode(debug_name),
  from_(from),
  to_(to)
{}

double UniformNode::GetConditional() const {
  return 1.0;
  //return 1.0/(to_ - from_);
}

double UniformNode::GetSample() {
  return NormUniformDraw() * (to_ - from_) + from_;
}

const std::vector<Node*>& Sampler::NonEvidenceNodes() const {
  return non_evidence_nodes_;
}

int Sampler::Register(Node* node) {
  if (!node->IsEvidence()) {
    non_evidence_nodes_.push_back(node);
  }
  all_nodes_.emplace_back(node);
  return all_nodes_.size() - 1;
}

Node* Sampler::GetNode(int registration_idx) {
  return all_nodes_[registration_idx].get();
}


void Sampler::Register(Worker* worker) {
  worker_.reset(worker);
}

Worker* Sampler::GetWorker() {
  return worker_.get();
}

GaussianSource::GaussianSource(double sigma2) :
  normal_distribution_(0.0, sigma2),
  generator_()
{}

double GaussianSource::Draw() {
  return normal_distribution_(generator_);
}

GaussianProposalDensity1D::GaussianProposalDensity1D(double sigma2) :
  norm_(sqrt(0.5 / sigma2 / M_PI)),
  half_inv_sigma2_(0.5 / sigma2),
  gaussian_source_(sigma2)
{}

double GaussianProposalDensity1D::Draw(double current_value) {
  //return current_value * (1.0  + gaussian_source_.Draw());
  return current_value + gaussian_source_.Draw();
}

MetroSampler::MetroSampler(ProposalDensity1D* proposal) :
    proposal_density_(proposal)
{}

void MetroSampler::Infer(int num_iterations) {
  std::cerr << "MetroSampler::Infer going for " << num_iterations << " iterations" << std::endl;
  for (int i = 0; i < num_iterations; ++i) {
    for (Node* node : NonEvidenceNodes()) {
      MetroStep(node);
    }
    GetWorker()->Sample(this);
  }
  std::cerr << "MetroSampler::Infer done" << std::endl;
}

void MetroSampler::MetroStep(Node* node) {
  double original = node->GetValue();
  double proposal = proposal_density_->Draw(original);  
  double likelihood_ratio = GetLikelihoodRatio(node, proposal, original);
  double transition_odds = GetTransitionProbabilityRatio(node, proposal, original);
  double transition_probability = likelihood_ratio * transition_odds; 
  /*
  std::cerr << "MetroSampler::MetroStep node [" << node->GetName()
            << "] original: " << original
            << " proposal: " << proposal
            << " r_likelihood: " << likelihood_ratio
            << std::endl;
            */
  if (transition_probability >= 1.0 ||
      NormUniformDraw() < transition_probability) {
    node->SetValue(proposal);
  } else {
    node->SetValue(original);
  }
}

double MetroSampler::GetLikelihoodRatio(Node* node,
                                        double proposal,
                                        double original) {
  return GetUnnormalizedLikelihood(node, proposal) /
          GetUnnormalizedLikelihood(node, original);
}

double MetroSampler::GetTransitionProbabilityRatio(
    Node* node, double proposal, double original) {
  return proposal_density_->GetUnnormalizedTransitionProbability(proposal, original) /
    proposal_density_->GetUnnormalizedTransitionProbability(original, proposal);
}

double MetroSampler::GetUnnormalizedLikelihood(
    Node* node, double value) {
  double original = node->GetValue();
  node->SetValue(value);
  double likelihood = node->GetConditional();
  for (Node* child : node->GetChildren()) {
    // FIXME - change to log domain.
    likelihood *= child->GetConditional(); 
  }
  node->SetValue(original);
  return likelihood;
}

double GaussianProposalDensity1D::GetUnnormalizedTransitionProbability(
    double from, double to) const {
  return Gaussian(from - to, half_inv_sigma2_); 
}

void Sampler::Reset() {
  std::unique_ptr<std::vector<Node*>> q1(new std::vector<Node*>);
  std::unique_ptr<std::vector<Node*>> q2(new std::vector<Node*>);
  for (int i = 0; i < all_nodes_.size(); ++i) {
    q1->push_back(all_nodes_[i].get());
  }

  while (!q1->empty()) {
    for (Node* node : *q1) {
      if (node->IsEvidence()) {
        node->SetInitialized();
      } else {
        if (node->AllParentsInitialized()) {
          node->SetValue(node->GetSample());
          node->SetInitialized();
        } else {
          q2->push_back(node);
        }
      }
    }
    q1->clear();
    q1.swap(q2);
  }

  worker_->Reset();
}

HistogramWorker::HistogramWorker(double range_start, double range_end, int num_bins, int node_idx) :
  histogram_(range_start, range_end, num_bins),
  node_idx_(node_idx)
{}

std::string HistogramWorker::ToJsonString() const {
  return histogram_.ToJsonString();
}

void HistogramWorker::Reset() {
  histogram_.Reset();
  std::cerr << "HistogramWorker::Reset:" << std::endl;
  std::cerr << histogram_.ToJsonString() << std::endl;
}

void HistogramWorker::Sample(Sampler* sampler) {
  double sample = sampler->GetNode(node_idx_)->GetValue();
  //std::cerr << "HistogramWorker sample " << sample << std::endl; 
  histogram_.Accumulate(sample);
}

