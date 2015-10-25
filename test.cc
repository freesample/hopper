#include <string>
#include <iostream>
#include "framework.h"

namespace sampler {

void TestMetroInitialize(Sampler* sampler) {
  const double v = 5.0;
  std::unique_ptr<Node> current_position(new EvidenceNode(v));

  const double x_0 = 10.0;
  const double delta_t = 1.0;
  const std::vector<double> beta {x_0, delta_t};
  const double sigma2 = 4.0; 
  std::unique_ptr<Node> next_position(new GaussianNode(beta, sigma2));

  next_position->EdgeFrom(current_position.get());

  sampler->Register(current_position.release());
  int node_idx = sampler->Register(next_position.release());

  const double histogram_range_min = 5.0;
  const double histogram_range_max = 20.0;
  const int histogram_num_bins = 20;
  std::unique_ptr<Worker> worker(new HistogramWorker(
      histogram_range_min, histogram_range_max, histogram_num_bins, node_idx));
  sampler->Register(worker.release());
  sampler->Reset();
}

void TestMetroInfer(Sampler* sampler, int num_iterations, std::string* result_json) {
  std::cerr << "sampler::TestMetroInfer before sampler->Infer()" << std::endl;
  sampler->Infer(num_iterations);
  std::cerr << "sampler::TestMetroInfer after sampler->Infer()" << std::endl;

  std::string histogram_json = "{values: [], data: []}";
  //std::string histogram_json = histogram_worker->ToJsonString();
  std::cout << "sampler::TestMetroInfer histogram:" << std::endl; 
  std::cout << histogram_json << std::endl; 
  *result_json = histogram_json;
}



}  // namespace sampler
