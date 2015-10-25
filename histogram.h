#ifndef SAMPLER_HISTOGRAM_H_
#define SAMPLER_HISTOGRAM_H_

#include <vector>
#include <string>

namespace sampler {

class Histogram {
  public:
  Histogram(double range_start, double range_end, int num_bins);
  void Accumulate(double x);
  std::string ToString() const;
  std::string ToJsonString() const;
  void Reset();

  protected:
  double range_start_;
  double range_end_;
  int num_bins_;
  std::vector<int> counts_;
  double units_per_bin_;
};

}  // namespace sampler

#endif  // SAMPLER_HISTOGRAM_H_
