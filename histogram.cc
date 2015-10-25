#include <sstream>
#include <iostream>

#include "histogram.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

namespace sampler {

namespace json = rapidjson;

using namespace std;

Histogram::Histogram(double range_start, double range_end, int num_bins) :
  range_start_(range_start), range_end_(range_end), num_bins_(num_bins),
  counts_(num_bins + 2, 0), units_per_bin_((range_end_ - range_start_)/num_bins) {}

void Histogram::Accumulate(double x) {
  int bin = -1;
  if (x < range_start_) {
    bin = 0;
  } else if (x >= range_end_) {
    bin = num_bins_ + 1;
  } else {
    bin = ((x - range_start_) / units_per_bin_)+1;
  }
  ++counts_[bin];
}

string Histogram::ToString() const {
  ostringstream oss;
  oss << "< " << range_start_ << ": " << counts_[0] << endl;
  for (int i = 1; i <= num_bins_; ++i) {
    oss << "[" << range_start_ + (i - 1) * units_per_bin_ << "," << range_start_ + i * units_per_bin_ << ") : " << counts_[i] << endl;
  }
  oss << ">= " << range_end_ << ": " << counts_[num_bins_ + 1];
  return oss.str();
}

string Histogram::ToJsonString() const {
  json::Document doc;
  doc.SetObject();

  json::Value values_array(json::kArrayType);
  json::Value data_array(json::kArrayType);

  values_array.PushBack(range_start_, doc.GetAllocator());
  data_array.PushBack(counts_[0], doc.GetAllocator());
  double last = range_start_;
  for (int i = 1; i <= num_bins_; ++i) {
    double current = last + units_per_bin_;
    values_array.PushBack(0.5 * (last + current), doc.GetAllocator());
    data_array.PushBack(counts_[i], doc.GetAllocator());
    last = current;
  }
  values_array.PushBack(range_end_, doc.GetAllocator());
  data_array.PushBack(counts_[num_bins_ + 1], doc.GetAllocator());

  doc.AddMember("values", values_array, doc.GetAllocator());
  doc.AddMember("data", data_array, doc.GetAllocator());

  json::StringBuffer buffer;
  json::Writer<json::StringBuffer> writer(buffer);
  doc.Accept(writer);

  return buffer.GetString();
}

void Histogram::Reset() {
  for (int i = 0; i < counts_.size(); ++i) {
    counts_[i] = 0;
  }
}

}  // namespace sampler
