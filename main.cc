#include <string>
#include <iostream>
#include "framework.h"
#include "test.h"

using namespace std;

int main(int argc, char** argv) {
  string result_json;

  const double proposal_sigma2 = 10.0; 
  std::unique_ptr<Sampler> sampler(new MetroSampler(new GaussianProposalDensity1D(proposal_sigma2)));

  sampler::TestMetroInitialize(sampler.get());
  sampler::TestMetroInfer(sampler.get(), 20, &result_json);

  return 0;
}
