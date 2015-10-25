#ifndef SAMPLER_TEST_H
#define SAMPLER_TEST_H

#include <string>

class Sampler;

namespace sampler {

void TestMetroInitialize(Sampler* sampler);
void TestMetroInfer(Sampler* sampler, int num_iterations, std::string* result_json);

}  // namespace sampler

#endif  // SAMPLER_TEST_H
