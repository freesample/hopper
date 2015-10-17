#include <cmath>
#include <iostream>
#include "framework.h"

static double Gaussian(double x, double half_inv_sigma2) {
  return exp(-x * x * half_inv_sigma2);
}

static double NormUniformDraw() {
  return double(rand()) / double(RAND_MAX); 
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

const int Node::GetNumParents() const {
  return parents_.size();
}

void Node::AddChild(Node* node) {
  children_.push_back(node);
}

void Node::AddParent(Node* node) {
  parents_.push_back(node);
}

GaussianNode::GaussianNode(const std::vector<double>& beta, double sigma2) :
    beta_(beta),
    sigma2_(sigma2),
    half_inv_sigma2_(0.5 / sigma2),
    gaussian_source_(sigma2)
{}

double GaussianNode::GetConditional() const {
  double mean = GetMean();
  return Gaussian(mean - GetValue(), half_inv_sigma2_);
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
  double mean = GetMean();
  return mean + gaussian_source_.Draw();
}

EvidenceNode::EvidenceNode(double value) {
  SetValue(value);
  SetEvidence();
}

double EvidenceNode::GetSample() {
  return GetValue();
}

double EvidenceNode::GetConditional() const {
  return 1.0;
}

const std::vector<Node*>& Sampler::NonEvidenceNodes() const {
  return non_evidence_nodes_;
}

void Sampler::Register(Node* node) {
  if (!node->IsEvidence()) {
    non_evidence_nodes_.push_back(node);
  }
  all_nodes_.push_back(node);
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

const std::vector<Node*>& Sampler::GetAllNodes() const {
  return all_nodes_;
}

MetroSampler::MetroSampler(ProposalDensity1D* proposal) :
    proposal_density_(proposal)
{}

void MetroSampler::Infer(Worker* worker, int num_iterations) {
  Initialize();
  for (int i = 0; i < num_iterations; ++i) {
    for (Node* node : NonEvidenceNodes()) {
      MetroStep(node);
    }
    worker->Sample(GetAllNodes());
  }
}

void MetroSampler::MetroStep(Node* node) {
  double original = node->GetValue();
  double proposal = proposal_density_->Draw(original);  
  double likelihood_ratio = GetLikelihoodRatio(node, proposal, original);
  double transition_odds = GetTransitionProbabilityRatio(node, proposal, original);
  double transition_probability = likelihood_ratio * transition_odds; 
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
    likelihood *= child->GetConditional(); 
  }
  node->SetValue(original);
  return likelihood;
}

double GaussianProposalDensity1D::GetUnnormalizedTransitionProbability(
    double from, double to) const {
  return Gaussian(from - to, half_inv_sigma2_); 
}

void HistogramWorker::Sample(const std::vector<Node*>& nodes) {
}

void Sampler::Initialize() {
  if (IsInitialized()) {
    return;
  }
  std::unique_ptr<std::vector<Node*>> q1(new std::vector<Node*>);
  std::unique_ptr<std::vector<Node*>> q2(new std::vector<Node*>);
  for (Node* node : GetAllNodes()) {
    q1->push_back(node);
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

  SetInitialized();
}

void Sampler::SetInitialized() {
  is_initialized_ = true;
}

void Sampler::ClearInitialized() {
  is_initialized_ = false;
}

bool Sampler::IsInitialized() const {
  return is_initialized_;
}

