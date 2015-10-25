#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include <vector>
#include <random>

#include "histogram.h"

class GaussianSource {
  private:
  std::normal_distribution<double> normal_distribution_;
  std::default_random_engine generator_;

  public:
  GaussianSource(double sigma2);
  double Draw();
};

class Node {
  private:
  std::vector<Node*> parents_;
  std::vector<Node*> children_;
  bool is_evidence_;
  double value_;
  bool is_initialized_;
  std::string debug_name_;

  void AddChild(Node* node);

  public:
  virtual double GetConditional() const = 0;
  virtual double GetSample() = 0;

  double GetValue() const;
  void SetValue(double value);
  bool IsEvidence() const;
  void SetEvidence();
  void ClearEvidence();
  bool IsInitialized() const;
  void SetInitialized();
  void ClearInitialized();
  bool AllParentsInitialized() const;

  const std::vector<Node*>& GetChildren() const;
  const std::vector<Node*>& GetParents() const;
  int GetNumParents() const;

  void EdgeFrom(Node* node);

  const std::string& GetName() const;

  protected:
  Node(const std::string& debug_name = "hieronymous");
};

class DiscreteNode : public Node {
  public:
  double GetConditional() const override;
};

class ContinuousNode : public Node {
  public:
  virtual double GetConditional() const = 0;

  protected:
  ContinuousNode(const std::string& debug_name = "anon_continuous");
};

class GaussianNode : public ContinuousNode {
  private:
  std::vector<double> beta_;
  double sigma2_;
  double half_inv_sigma2_;
  GaussianSource gaussian_source_;

  public:
  GaussianNode(const std::vector<double>& beta, double sigma2, const std::string& debug_name = "anon_gaussian");
  double GetConditional() const override;
  virtual double GetSample() override;
  double GetMean() const;
};

class GaussianEvidenceNode : public GaussianNode {
  public:
  GaussianEvidenceNode(const std::vector<double>& beta, double sigma2, double value, const std::string& debug_name = "anon_gaussian_evidence");
  double GetSample() override;
};

class EvidenceNode : public ContinuousNode {
  public:
  EvidenceNode(double value);
  double GetConditional() const override;
  double GetSample() override;
};

class UniformNode : public ContinuousNode {
  public:
  UniformNode(double from, double to, const std::string& debug_name = "anon_uniform");
  double GetConditional() const override;
  double GetSample() override;
  
  private:
  double from_;
  double to_;
};

class ProposalDensity1D {
  public:
  virtual double Draw(double current_value) = 0; 
  virtual double GetUnnormalizedTransitionProbability(double from, double to) const = 0;
};

class GaussianProposalDensity1D : public ProposalDensity1D {
  private:
  double half_inv_sigma2_;
  double norm_;
  GaussianSource gaussian_source_;

  public:
  GaussianProposalDensity1D(double sigma2);
  double Draw(double current_value) override;
  double GetUnnormalizedTransitionProbability(double from, double to) const override;
};

class Sampler;

class Worker {
  public:
  virtual void Reset() = 0;
  virtual void Sample(Sampler* sampler) = 0;
};

class HistogramWorker : public Worker {
  sampler::Histogram histogram_;
  int node_idx_;
  public:
  HistogramWorker(double range_start, double range_end, int num_bins, int node_idx);

  void Sample(Sampler* sampler) override;
  void Reset() override;

  std::string ToJsonString() const;
};

class Sampler {
  private:
  std::vector<Node*> non_evidence_nodes_;
  // Sampler owns Register()'ed Nodes.
  std::vector<std::unique_ptr<Node>> all_nodes_;
  std::unique_ptr<Worker> worker_;
  bool is_initialized_;

  public:
  // Transfers ownership of Node to Sampler.
  int Register(Node* node);
  // Transfers ownership of Worker to Sampler.
  void Register(Worker* worker);
  void Reset();
  virtual void Infer(int num_iterations) = 0;
  Node* GetNode(int registration_idx);
  Worker* GetWorker();

  protected:
  const std::vector<Node*>& NonEvidenceNodes() const;
};

class MetroSampler : public Sampler {
  private:
  std::unique_ptr<ProposalDensity1D> proposal_density_;

  void MetroStep(Node* node);
  double GetLikelihoodRatio(Node* node, double proposal, double original);
  double GetTransitionProbabilityRatio(Node* node,
                                       double proposal,
                                       double original);
  double GetUnnormalizedLikelihood(Node* node, double value);
  
  public:
  MetroSampler(ProposalDensity1D* proposal);
  void Infer(int num_iterations) override;
};

class GibbsSampler : public Sampler {
};

#endif // FRAMEWORK_H
