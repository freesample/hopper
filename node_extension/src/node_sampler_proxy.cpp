#include <iostream>
#include <string>
#include <v8.h>
#include <node.h>
#include <node_object_wrap.h>

// From hopper package
#include "hopper/framework.h"

using namespace v8;
using namespace std;

class SamplerProxy : public node::ObjectWrap {
  public:
  static void Init(Local<Object> exports);
  //static void NewInstance(const FunctionCallbackInfo<Value>& args);

  private:
  SamplerProxy() {}
  ~SamplerProxy() {}

  static void New(const FunctionCallbackInfo<Value>& args);
  static void SetupExperiment(const FunctionCallbackInfo<Value>& args);
  static void Reset(const FunctionCallbackInfo<Value>& args);
  static void TestMetroInfer(const FunctionCallbackInfo<Value>& args);

  void TestMetroInferInternal(int num_iterations, std::string* result_json);
  void SetupExperiment1Internal();
  void SetupExperiment2Internal();
  void ResetInternal();

  static Persistent<Function> ctor_tmpl_static_;
  std::unique_ptr<Sampler> sampler_;
};

v8::Persistent<Function> SamplerProxy::ctor_tmpl_static_;

extern "C" {
  static void init(Handle<Object> target) {
    SamplerProxy::Init(target);
  }
  NODE_MODULE(samplerproxy, init);
}

void SamplerProxy::Init(Local<Object> exports) {
  Isolate* isolate = exports->GetIsolate();

  Local<FunctionTemplate> tmpl = FunctionTemplate::New(isolate, SamplerProxy::New);
  tmpl->SetClassName(String::NewFromUtf8(isolate, "SamplerProxy"));
  tmpl->InstanceTemplate()->SetInternalFieldCount(1);

  NODE_SET_PROTOTYPE_METHOD(tmpl, "setupExperiment", SamplerProxy::SetupExperiment);
  NODE_SET_PROTOTYPE_METHOD(tmpl, "reset", SamplerProxy::Reset);
  NODE_SET_PROTOTYPE_METHOD(tmpl, "testMetroInfer", SamplerProxy::TestMetroInfer);

  ctor_tmpl_static_.Reset(isolate, tmpl->GetFunction());

  exports->Set(String::NewFromUtf8(isolate, "SamplerProxy"), tmpl->GetFunction());
}

void SamplerProxy::New(const FunctionCallbackInfo<Value>& args) {
  if (args.IsConstructCall()) {
    SamplerProxy* proxy = new SamplerProxy;
    proxy->Wrap(args.This());
    args.GetReturnValue().Set(args.This());
  } else {
    //NewInstance(args);
  }
}

/*
void SamplerProxy::NewInstance(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();

  const int argc = 0;
  Local<Value> argv = {};
  Local<Function> ctor = Local<Function>::New(isolate, ctor_tmpl_static_);
  args.GetReturnValue().Set(ctor->NewInstance(argc, argv));
}
*/

void SamplerProxy::Reset(const FunctionCallbackInfo<Value>& args) {
  SamplerProxy* proxy = ObjectWrap::Unwrap<SamplerProxy>(args.Holder());
  cerr << "SamplerProxy::Reset Internal commence" << endl;
  proxy->ResetInternal();
  cerr << "SamplerProxy::Reset Internal complete" << endl;
}

void SamplerProxy::SetupExperiment(const FunctionCallbackInfo<Value>& args) {
  SamplerProxy* proxy = ObjectWrap::Unwrap<SamplerProxy>(args.Holder());
  cerr << "SamplerProxy::SetupExperiment Internal commence" << endl;
  //proxy->SetupExperiment1Internal();
  proxy->SetupExperiment2Internal();
  cerr << "SamplerProxy::SetupExperiment Internal complete" << endl;
}

void SamplerProxy::TestMetroInfer(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();

  if(args.Length() == 0 || !args[0]->IsNumber()) {
    cout << "SamplerProxy::TestMetroInfer requires <num_iterations> arg" << endl;
  } else {
    int num_iterations = args[0]->ToUint32()->Value();

    SamplerProxy* proxy = ObjectWrap::Unwrap<SamplerProxy>(args.Holder());
    std::string result_json;
    proxy->TestMetroInferInternal(num_iterations, &result_json);

    cerr << "SamplerProxy::TestMetroInfer Internal complete" << endl;
    cerr << "SamplerProxy::TestMetroInfer result: " << endl;
    cerr << result_json << endl;
    cerr << "SamplerProxy::TestMetroInfer printed result" << endl;
    args.GetReturnValue().Set(String::NewFromUtf8(isolate, result_json.c_str()));
    cerr << "SamplerProxy::TestMetroInfer set return value" << endl;
  }
}

void SamplerProxy::SetupExperiment1Internal() {
  // Create nodes.
  // 1. Speed
  const double v = 5.0;
  std::unique_ptr<Node> speed(new EvidenceNode(v));

  // 2. Next position.
  const double x_0 = 10.0;
  const double delta_t = 1.0;
  const std::vector<double> beta {x_0, delta_t};
  const double sigma2 = 4.0; 
  std::unique_ptr<Node> next_position(new GaussianNode(beta, sigma2));

  // Create edges of Bayesian network.
  next_position->EdgeFrom(speed.get());

  // Create a Sampler to sample given evidence.
  // In this case, we use Metropolis-Hastings sampling with a Gaussian proposal density.
  const double proposal_sigma2 = 5.0; 
  std::unique_ptr<Sampler> sampler(new MetroSampler(new GaussianProposalDensity1D(proposal_sigma2)));

  // Register nodes of the graph with the Sampler. 
  // Transfers ownership of Nodes to Sampler.
  sampler->Register(speed.release());
  int node_idx = sampler->Register(next_position.release());

  // Create a worker that operates upon samples from the (approximated) posterior.
  // Here, the worker simply creates a histogram of the sampled values of the next position node.
  const double histogram_range_min = 5.0;
  const double histogram_range_max = 20.0;
  const int histogram_num_bins = 20;
  std::unique_ptr<HistogramWorker> worker(new HistogramWorker(
      histogram_range_min, histogram_range_max, histogram_num_bins, node_idx));
  // Register worker with the Sampler. 
  // Transfers ownership of Worker to Sampler.
  sampler->Register(worker.release());

  sampler_.reset(sampler.release());
}

// Infer velocity from Next position and mixing coefficients.
void SamplerProxy::SetupExperiment2Internal() {
  // Create nodes.
  // 1. Speed;
  const double min_u = 0.0;
  const double max_u = 10.0;
  std::unique_ptr<Node> u(new UniformNode(min_u, max_u));

  // 2. Next position.
  const double x_0 = 10.0;
  const double delta_t = 1.0;
  const double x_1 = 15.0;
  const std::vector<double> beta {x_0, delta_t};
  const double sigma2 = 4.0; 
  std::unique_ptr<Node> next_position(new GaussianEvidenceNode(beta, sigma2, x_1));

  // Create edges of Bayesian network.
  next_position->EdgeFrom(u.get());

  // Create a Sampler to sample given evidence.
  // In this case, we use Metropolis-Hastings sampling with a Gaussian proposal density.
  const double proposal_sigma2 = 5.0; 
  std::unique_ptr<Sampler> sampler(new MetroSampler(new GaussianProposalDensity1D(proposal_sigma2)));

  // Register nodes of the graph with the Sampler. 
  // Transfers ownership of Nodes to Sampler.
  int node_idx = sampler->Register(u.release());
  sampler->Register(next_position.release());

  // Create a worker that operates upon samples from the (approximated) posterior.
  // Here, the worker simply creates a histogram of the sampled values of the next position node.
  const double histogram_range_min = min_u;
  const double histogram_range_max = max_u;
  const int histogram_num_bins = 20;
  std::unique_ptr<HistogramWorker> worker(new HistogramWorker(
      histogram_range_min, histogram_range_max, histogram_num_bins, node_idx));
  // Register worker with the Sampler. 
  // Transfers ownership of Worker to Sampler.
  sampler->Register(worker.release());

  sampler_.reset(sampler.release());
}

void SamplerProxy::ResetInternal() {
  std::cerr << "SamplerProxy::ResetInternal before sampler->Reset()" << std::endl;
  sampler_->Reset();
  std::cerr << "SamplerProxy::TestMetroInferInternal after sampler->Reset()" << std::endl;
}

void SamplerProxy::TestMetroInferInternal(int num_iterations, std::string* result_json) {
  std::cerr << "SamplerProxy::TestMetroInferInternal before sampler->Infer()" << std::endl;
  sampler_->Infer(num_iterations);
  std::cerr << "SamplerProxy::TestMetroInferInternal after sampler->Infer()" << std::endl;

  std::string histogram_json = static_cast<HistogramWorker*>(sampler_->GetWorker())->ToJsonString();
  std::cout << "SamplerProxy::TestMetroInferInternal histogram:" << std::endl; 
  std::cout << histogram_json << std::endl; 
  *result_json = histogram_json;
}
