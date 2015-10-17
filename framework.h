#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include <vector>
#include <random>

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
  const int GetNumParents() const;

  void AddChild(Node* node);
  void AddParent(Node* node);
};

class DiscreteNode : public Node {
  public:
  double GetConditional() const override;
};

class ContinuousNode : public Node {
  public:
  virtual double GetConditional() const = 0;
};

class GaussianNode : public ContinuousNode {
  private:
  std::vector<double> beta_;
  double sigma2_;
  double half_inv_sigma2_;
  GaussianSource gaussian_source_;

  public:
  GaussianNode(const std::vector<double>& beta, double sigma2);
  double GetConditional() const override;
  double GetSample() override;
  double GetMean() const;
};

class EvidenceNode : public ContinuousNode {
  public:
  EvidenceNode(double value);
  double GetSample() override;
  double GetConditional() const override;
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

class Worker {
  public:
  virtual void Sample(const std::vector<Node*>& nodes) = 0;
};

class HistogramWorker : public Worker {
  public:
  void Sample(const std::vector<Node*>& nodes) override;
};

class Sampler {
  private:
  std::vector<Node*> non_evidence_nodes_;
  std::vector<Node*> all_nodes_;
  bool is_initialized_;

  public:
  void Register(Node* node);
  const std::vector<Node*>& GetAllNodes() const;
  virtual void Infer(Worker* worker, int num_iterations) = 0;
  bool IsInitialized() const;
  void SetInitialized();
  void ClearInitialized();

  protected:
  void Initialize();
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
  void Infer(Worker* worker, int num_iterations) override;
};

class GibbsSampler : public Sampler {
};

#endif // FRAMEWORK_H
