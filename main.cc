#include "framework.h"

int main(int argc, char** argv) {
  const double x = 5.0;
  std::unique_ptr<Node> current_position(new EvidenceNode(x));

  const std::vector<double> beta {10.0, 1.0};
  const double sigma2 = 4.0; 
  std::unique_ptr<Node> next_position(new GaussianNode(beta, sigma2));

  next_position->AddParent(current_position.get());

  const double proposal_sigma2 = 10.0; 
  std::unique_ptr<Sampler> sampler(new MetroSampler(new GaussianProposalDensity1D(proposal_sigma2)));
  sampler->Register(current_position.get());
  sampler->Register(next_position.get());

  std::unique_ptr<Worker> worker(new HistogramWorker);
  sampler->Infer(worker.get(), 100);
}
